#!/usr/bin/env python3
"""
Whisper STT Daemon Server
Keeps Whisper model loaded in memory with Apple Silicon MLX acceleration.
Accepts base64-encoded audio and returns transcribed text in JSON responses.
"""
import argparse
import base64
import json
import os
import socket
import struct
import subprocess
import tempfile
import threading
import time
import warnings

warnings.filterwarnings("ignore")


class WhisperDaemon:
    def __init__(self, host="localhost", port=9998, idle_timeout=900, model="mlx-community/whisper-large-v3-turbo"):
        self.host = host
        self.port = port
        self.model_name = model
        self.model = None
        self.running = False
        self.sock = None
        self.idle_timeout = idle_timeout
        self.last_activity = None
        self.activity_lock = threading.Lock()

    def update_activity(self):
        with self.activity_lock:
            self.last_activity = time.time()

    def check_idle_timeout(self):
        if self.idle_timeout <= 0:
            return False
        with self.activity_lock:
            if self.last_activity is None:
                return False
            return (time.time() - self.last_activity) > self.idle_timeout

    def idle_monitor(self):
        if self.idle_timeout <= 0:
            return
        while self.running:
            if self.check_idle_timeout():
                print(f"Daemon idle for {self.idle_timeout // 60} minutes, shutting down...")
                self.running = False
                break
            time.sleep(30)

    def load_model(self):
        try:
            print(f"Loading Whisper model: {self.model_name}...")
            import mlx_whisper
            self.mlx_whisper = mlx_whisper
            # Warm up by transcribing a short silent audio
            self._warmup_model()
            print("Whisper model loaded and ready!")
            return True
        except Exception as e:
            print(f"Error loading Whisper model: {e}")
            return False

    def _warmup_model(self):
        try:
            print("Warming up model (first-run download + compilation)...")
            t0 = time.time()
            # Create a short silent WAV file for warmup
            import numpy as np
            import soundfile as sf
            import io
            silence = np.zeros(16000, dtype=np.float32)  # 1 second of silence at 16kHz
            buf = io.BytesIO()
            sf.write(buf, silence, 16000, format="WAV", subtype="PCM_16")
            buf.seek(0)
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp.write(buf.read())
            tmp.close()
            try:
                self.mlx_whisper.transcribe(tmp.name, path_or_hf_repo=self.model_name)
            finally:
                os.unlink(tmp.name)
            print(f"Warmup complete ({time.time() - t0:.1f}s)")
        except Exception as e:
            print(f"Warmup failed (non-fatal): {e}")

    def transcribe(self, audio_bytes, language=None):
        """Transcribe audio bytes and return text + metadata."""
        # Write audio to temp file (mlx_whisper expects a file path)
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        try:
            # If the audio is not WAV, convert with ffmpeg
            tmp.write(audio_bytes)
            tmp.close()

            # Probe format and convert to 16kHz mono WAV if needed
            wav_path = self._ensure_wav(tmp.name)

            t0 = time.time()
            kwargs = {"path_or_hf_repo": self.model_name}
            if language:
                kwargs["language"] = language

            result = self.mlx_whisper.transcribe(wav_path, **kwargs)
            transcription_time = time.time() - t0

            text = result.get("text", "").strip()
            detected_lang = result.get("language", language or "unknown")

            # Estimate audio duration from file
            try:
                import soundfile as sf
                info = sf.info(wav_path)
                duration = info.duration
            except Exception:
                duration = 0

            print(f"Transcribed: {duration:.1f}s audio in {transcription_time:.2f}s lang={detected_lang} text='{text[:80]}'")

            return {
                "text": text,
                "language": detected_lang,
                "duration": round(duration, 3),
                "transcription_time": round(transcription_time, 3),
            }
        finally:
            os.unlink(tmp.name)
            # Clean up converted file if different
            if 'wav_path' in locals() and wav_path != tmp.name and os.path.exists(wav_path):
                os.unlink(wav_path)

    def _ensure_wav(self, input_path):
        """Convert input audio to 16kHz mono WAV if needed. Returns path to WAV file."""
        # Check if it's already a valid WAV
        try:
            import soundfile as sf
            info = sf.info(input_path)
            if info.samplerate == 16000 and info.channels == 1:
                return input_path
        except Exception:
            pass

        # Convert with ffmpeg
        wav_path = input_path + ".converted.wav"
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", input_path, "-ar", "16000", "-ac", "1", "-f", "wav", wav_path],
                capture_output=True, check=True, timeout=30
            )
            return wav_path
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"ffmpeg conversion failed: {e}, trying raw input...")
            return input_path

    # -- TCP protocol (identical to Kokoro daemon) --

    def _recv_all(self, sock, length):
        chunks = []
        received = 0
        while received < length:
            chunk = sock.recv(min(length - received, 65536))
            if not chunk:
                raise ConnectionError("Connection closed")
            chunks.append(chunk)
            received += len(chunk)
        return b"".join(chunks)

    def _recv_request(self, sock):
        """Receive a length-prefixed JSON request (4-byte big-endian header)."""
        header = self._recv_all(sock, 4)
        payload_len = struct.unpack("!I", header)[0]

        # Detect legacy raw-JSON clients (first byte is '{' or '[')
        if header[0] in (0x7B, 0x5B):
            data = header
            while True:
                try:
                    return json.loads(data.decode("utf-8"))
                except json.JSONDecodeError:
                    pass
                chunk = sock.recv(65536)
                if not chunk:
                    return json.loads(data.decode("utf-8"))
                data += chunk

        if payload_len > 100 * 1024 * 1024:
            raise ValueError(f"Payload too large: {payload_len}")
        payload = self._recv_all(sock, payload_len)
        return json.loads(payload.decode("utf-8"))

    def _send_response(self, sock, response_dict):
        """Send a length-prefixed JSON response."""
        data = json.dumps(response_dict).encode("utf-8")
        sock.sendall(struct.pack("!I", len(data)) + data)

    # -- Client handling --

    def handle_client(self, client_socket, addr):
        try:
            self.update_activity()
            request = self._recv_request(client_socket)

            # Health check
            if request.get("action") == "ping":
                self._send_response(client_socket, {"success": True, "status": "ready"})
                return

            # Transcription request
            audio_b64 = request.get("audio_base64", "")
            if not audio_b64:
                self._send_response(client_socket, {"success": False, "error": "No audio_base64 provided"})
                return

            audio_bytes = base64.b64decode(audio_b64)
            language = request.get("language")

            result = self.transcribe(audio_bytes, language)

            self._send_response(client_socket, {
                "success": True,
                **result,
            })

        except Exception as e:
            print(f"Error handling client {addr}: {e}")
            try:
                self._send_response(client_socket, {"success": False, "error": str(e)})
            except:
                pass
        finally:
            client_socket.close()

    # -- Server lifecycle --

    def start(self):
        if not self.load_model():
            return False

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.settimeout(1.0)
        self.sock.bind((self.host, self.port))
        self.sock.listen(5)
        self.running = True

        print(f"Whisper STT Daemon started on {self.host}:{self.port}")
        if self.idle_timeout > 0:
            print(f"Auto-shutdown after {self.idle_timeout // 60} minutes idle")
        else:
            print("Idle timeout disabled")
        print("Using Apple Silicon MLX acceleration")

        self.update_activity()

        idle_thread = threading.Thread(target=self.idle_monitor, daemon=True)
        idle_thread.start()

        try:
            while self.running:
                try:
                    client_sock, addr = self.sock.accept()
                    t = threading.Thread(target=self.handle_client, args=(client_sock, addr), daemon=True)
                    t.start()
                except socket.timeout:
                    continue
                except socket.error:
                    if self.running:
                        print("Socket error")
                    break
        except KeyboardInterrupt:
            print("\nStopping daemon...")
        finally:
            self.stop()

    def stop(self):
        self.running = False
        if self.sock:
            self.sock.close()
        print("Daemon stopped")


def main():
    parser = argparse.ArgumentParser(description="Whisper STT Daemon")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=9998)
    parser.add_argument("--model", default="mlx-community/whisper-large-v3-turbo",
                        help="HuggingFace model repo for MLX Whisper")
    parser.add_argument("--idle-timeout", type=int, default=900,
                        help="Auto-shutdown after idle seconds (0 = disabled)")
    args = parser.parse_args()

    daemon = WhisperDaemon(host=args.host, port=args.port, idle_timeout=args.idle_timeout, model=args.model)
    daemon.start()


if __name__ == "__main__":
    main()
