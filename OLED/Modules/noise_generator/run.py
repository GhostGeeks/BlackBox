#!/usr/bin/env python3
"""
BlackBox Noise Generator module (headless)
- JSON-only stdout protocol
- Non-blocking stdin via selectors + os.read
- Heartbeat at least every 250ms
- Audio playback via background loop process (pw-play/paplay/aplay), WAV regenerated on changes
- Never touches OLED, never manages Bluetooth
"""

import errno
import json
import os
import selectors
import signal
import subprocess
import sys
import tempfile
import time
import traceback
import wave
import random
from typing import Optional, Dict, Any, List

MODULE_NAME = "noise_generator"
MODULE_VERSION = "ng_v1"

HEARTBEAT_S = 0.25
UI_TICK_S = 0.05  # internal tick to keep loop responsive without blocking
SAMPLE_RATE = 44100
CHANNELS = 2
SAMPLE_WIDTH = 2  # int16
WAV_SECONDS = 2.0  # short loop buffer; low CPU to regenerate
TMP_PREFIX = "blackbox_noise_"

MODES: List[str] = ["white", "pink", "brown"]
MODE_LABEL = {"white": "White", "pink": "Pink", "brown": "Brown"}


def _json_emit(obj: Dict[str, Any]) -> None:
    # JSON-only, single line
    sys.stdout.write(json.dumps(obj, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def _safe_message_from_exc(e: BaseException) -> str:
    # Keep short and user-meaningful; avoid huge tracebacks
    msg = str(e).strip()
    if not msg:
        msg = e.__class__.__name__
    return msg[:160]


def _which(cmd: str) -> Optional[str]:
    # Minimal which without importing shutil (fine either way, but this is tiny & explicit)
    paths = os.environ.get("PATH", "").split(":")
    for p in paths:
        fp = os.path.join(p, cmd)
        if os.path.isfile(fp) and os.access(fp, os.X_OK):
            return fp
    return None


class AudioBackend:
    """
    Spawns a background *looping* playback shell process:
      while true; <player> <wav>; done
    We bake volume into the WAV samples; no runtime mixer needed.
    """

    def __init__(self) -> None:
        self.player = None  # "pw-play" | "paplay" | "aplay"
        self.player_path = None
        self.proc: Optional[subprocess.Popen] = None
        self.wav_path: Optional[str] = None

        # Prefer PipeWire native; then Pulse; then ALSA
        for c in ("pw-play", "paplay", "aplay"):
            p = _which(c)
            if p:
                self.player = c
                self.player_path = p
                break

    def available(self) -> bool:
        return self.player_path is not None

    def start_loop(self, wav_path: str) -> None:
        self.stop()

        self.wav_path = wav_path
        if not self.available():
            raise RuntimeError("No audio player found (need pw-play or paplay or aplay)")

        # Loop in a shell so the player can exit naturally at end-of-file.
        # Redirect stdout/stderr to avoid contaminating module stdout (JSON-only).
        # Use setsid so we can terminate the whole process group cleanly.
        if self.player == "pw-play":
            cmd = ["/bin/sh", "-c", f'while true; "{self.player_path}" "{wav_path}" >/dev/null 2>&1; done']
        elif self.player == "paplay":
            cmd = ["/bin/sh", "-c", f'while true; "{self.player_path}" "{wav_path}" >/dev/null 2>&1; done']
        else:  # aplay
            cmd = ["/bin/sh", "-c", f'while true; "{self.player_path}" -q "{wav_path}" >/dev/null 2>&1; done']

        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            preexec_fn=os.setsid,
            close_fds=True,
        )

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            try:
                # kill the process group
                os.killpg(self.proc.pid, signal.SIGTERM)
            except Exception:
                pass
            # quick wait, then SIGKILL if needed
            t0 = time.monotonic()
            while time.monotonic() - t0 < 0.5:
                if self.proc.poll() is not None:
                    break
                time.sleep(0.02)
            if self.proc.poll() is None:
                try:
                    os.killpg(self.proc.pid, signal.SIGKILL)
                except Exception:
                    pass
        self.proc = None
        self.wav_path = None


def _clamp(v: int, lo: int, hi: int) -> int:
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v


def _int16_from_float(x: float) -> int:
    # clamp [-1.0, 1.0]
    if x < -1.0:
        x = -1.0
    elif x > 1.0:
        x = 1.0
    return int(x * 32767.0)


def _gen_white(n: int, rng: random.Random) -> List[float]:
    # Uniform white noise
    return [rng.uniform(-1.0, 1.0) for _ in range(n)]


def _gen_pink(n: int, rng: random.Random) -> List[float]:
    """
    Approx pink noise using Voss-McCartney (low CPU).
    """
    num_rows = 16
    rows = [0.0] * num_rows
    running_sum = 0.0
    # init rows
    for i in range(num_rows):
        rows[i] = rng.uniform(-1.0, 1.0)
        running_sum += rows[i]

    out: List[float] = []
    # counters
    counter = 0
    for _ in range(n):
        counter += 1
        # determine which row to update based on trailing zeros
        # (classic Voss approach)
        c = counter
        row = 0
        while (c & 1) == 0 and row < num_rows:
            c >>= 1
            row += 1
        if row < num_rows:
            running_sum -= rows[row]
            rows[row] = rng.uniform(-1.0, 1.0)
            running_sum += rows[row]
        # add an extra white component for smoother spectrum
        white = rng.uniform(-1.0, 1.0)
        sample = (running_sum + white) / (num_rows + 1)
        out.append(sample)
    return out


def _gen_brown(n: int, rng: random.Random) -> List[float]:
    """
    Brown(ian/red) noise by integrating white noise with gentle damping.
    """
    out: List[float] = []
    x = 0.0
    for _ in range(n):
        x += rng.uniform(-1.0, 1.0) * 0.02
        x *= 0.999  # slight damping to avoid runaway drift
        out.append(x)
    # normalize roughly
    mx = max(1e-9, max(abs(v) for v in out))
    scale = 1.0 / mx
    return [v * scale for v in out]


def _write_wav(path: str, mode: str, volume_pct: int, seconds: float) -> None:
    n = int(SAMPLE_RATE * seconds)
    rng = random.Random(0xB10C0 + volume_pct + (MODES.index(mode) * 1337))

    if mode == "white":
        mono = _gen_white(n, rng)
    elif mode == "pink":
        mono = _gen_pink(n, rng)
    else:
        mono = _gen_brown(n, rng)

    # volume baked into samples (0..100) -> amplitude scalar
    amp = (volume_pct / 100.0)
    # keep some headroom
    amp *= 0.8

    with wave.open(path, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(SAMPLE_WIDTH)
        wf.setframerate(SAMPLE_RATE)

        # interleave stereo (same signal both channels)
        frames = bytearray()
        for s in mono:
            v = _int16_from_float(s * amp)
            # little-endian int16, duplicated for stereo
            frames += int(v).to_bytes(2, byteorder="little", signed=True)
            frames += int(v).to_bytes(2, byteorder="little", signed=True)

        wf.writeframes(frames)


class NoiseModule:
    def __init__(self) -> None:
        self.sel = selectors.DefaultSelector()
        self.stdin_fd = sys.stdin.fileno()
        os.set_blocking(self.stdin_fd, False)
        self.sel.register(self.stdin_fd, selectors.EVENT_READ)

        self.backend = AudioBackend()

        self.page = "main"
        self.ready = True

        self.mode_idx = 0
        self.mode = MODES[self.mode_idx]

        self.playing = False
        self.volume = 70
        self.loop = True

        # select_hold toggles control focus
        self.focus = "mode"  # "mode" or "volume"

        self._stdin_buf = b""

        self._tmpdir = tempfile.gettempdir()
        self._wav_path = os.path.join(self._tmpdir, f"{TMP_PREFIX}{os.getpid()}.wav")

        self._last_state_emit = 0.0
        self._last_any_emit = 0.0

        self._stop_requested = False
        self._fatal: Optional[str] = None

    def emit_hello(self) -> None:
        _json_emit({"type": "hello", "module": MODULE_NAME, "version": MODULE_VERSION})
        _json_emit({"type": "page", "name": self.page})
        self.emit_state(force=True)

    def emit_state(self, force: bool = False) -> None:
        now = time.monotonic()
        if not force and (now - self._last_state_emit) < HEARTBEAT_S:
            return
        self._last_state_emit = now
        self._last_any_emit = now
        _json_emit({
            "type": "state",
            "ready": bool(self.ready and self._fatal is None),
            "mode": self.mode,
            "playing": bool(self.playing),
            "volume": int(self.volume),
            "duration_s": 0,
            "loop": bool(self.loop),
            "focus": self.focus,
        })

    def toast(self, msg: str) -> None:
        self._last_any_emit = time.monotonic()
        _json_emit({"type": "toast", "message": str(msg)[:160]})

    def fatal(self, msg: str) -> None:
        # Emit fatal but keep process alive if possible
        self._fatal = str(msg)[:160]
        self._last_any_emit = time.monotonic()
        _json_emit({"type": "fatal", "message": self._fatal})

    def _ensure_wav(self) -> None:
        _write_wav(self._wav_path, self.mode, self.volume, WAV_SECONDS)

    def _start_audio(self) -> None:
        if self.playing:
            return
        if not self.backend.available():
            raise RuntimeError("Audio backend not available (pw-play/paplay/aplay missing)")
        self._ensure_wav()
        self.backend.start_loop(self._wav_path)
        self.playing = True

    def _stop_audio(self) -> None:
        self.backend.stop()
        self.playing = False

    def _restart_audio_if_playing(self) -> None:
        if not self.playing:
            return
        # regenerate WAV and restart loop process
        self._stop_audio()
        self._start_audio()

    def _change_mode(self, delta: int) -> None:
        self.mode_idx = (self.mode_idx + delta) % len(MODES)
        self.mode = MODES[self.mode_idx]
        self.toast(f"Mode: {MODE_LABEL.get(self.mode, self.mode.title())}")
        self._restart_audio_if_playing()

    def _change_volume(self, delta: int) -> None:
        self.volume = _clamp(self.volume + delta, 0, 100)
        self.toast(f"Volume: {self.volume}")
        self._restart_audio_if_playing()

    def handle_button(self, name: str) -> None:
        if name == "back":
            # exit immediately; stop playback first
            try:
                self._stop_audio()
            except Exception:
                pass
            self._stop_requested = True
            _json_emit({"type": "exit"})
            return

        if self._fatal is not None:
            # After fatal, allow back to exit; ignore other inputs
            return

        if name == "select":
            try:
                if self.playing:
                    self._stop_audio()
                    self.toast("STOP")
                else:
                    self._start_audio()
                    self.toast("PLAY")
            except Exception as e:
                self.fatal(_safe_message_from_exc(e))
            finally:
                self.emit_state(force=True)
            return

        if name == "select_hold":
            # toggle focus between adjusting mode vs volume
            self.focus = "volume" if self.focus == "mode" else "mode"
            self.toast(f"Adjust: {self.focus}")
            self.emit_state(force=True)
            return

        if name == "up":
            if self.focus == "mode":
                self._change_mode(+1)
            else:
                self._change_volume(+5)
            self.emit_state(force=True)
            return

        if name == "down":
            if self.focus == "mode":
                self._change_mode(-1)
            else:
                self._change_volume(-5)
            self.emit_state(force=True)
            return

        # ignore unknown button names silently

    def _read_stdin_nonblocking(self) -> None:
        # Pull available bytes; split lines; handle complete commands
        try:
            while True:
                try:
                    chunk = os.read(self.stdin_fd, 4096)
                except OSError as e:
                    if e.errno in (errno.EAGAIN, errno.EWOULDBLOCK):
                        break
                    raise
                if not chunk:
                    # stdin closed -> exit cleanly
                    self._stop_requested = True
                    return
                self._stdin_buf += chunk

                while b"\n" in self._stdin_buf:
                    line, self._stdin_buf = self._stdin_buf.split(b"\n", 1)
                    cmd = line.decode("utf-8", errors="ignore").strip()
                    if cmd:
                        self.handle_button(cmd)
                        if self._stop_requested:
                            return
        except Exception as e:
            # stdin errors should not crash; emit fatal and keep looping
            self.fatal(f"stdin error: {_safe_message_from_exc(e)}")

    def run(self) -> int:
        self.emit_hello()

        # If no audio backend, we can still let UI run and only fatal when user hits play.
        # Heartbeat continues regardless.

        last_tick = time.monotonic()
        while not self._stop_requested:
            try:
                # selectors wait a short time; never block long
                events = self.sel.select(timeout=UI_TICK_S)
                for key, mask in events:
                    if key.fileobj == self.stdin_fd and (mask & selectors.EVENT_READ):
                        self._read_stdin_nonblocking()

                # Heartbeat (even if idle)
                self.emit_state(force=False)

                # If our playback loop died unexpectedly, reflect it without hanging
                if self.playing and self.backend.proc is not None and (self.backend.proc.poll() is not None):
                    # player died -> stop state; mark fatal
                    self.playing = False
                    self.fatal("Audio playback stopped unexpectedly")
                    self.emit_state(force=True)

                # lightweight tick pacing
                now = time.monotonic()
                if (now - last_tick) < UI_TICK_S:
                    # already paced by select timeout; nothing needed
                    pass
                last_tick = now

            except Exception as e:
                # Catch-all: never dump to stdout; report fatal
                self.fatal(_safe_message_from_exc(e))
                self.emit_state(force=True)

        return 0


def main() -> int:
    # Ensure signals allow clean exit
    mod = NoiseModule()

    def _sigterm(_signum, _frame):
        try:
            mod._stop_audio()
        except Exception:
            pass
        mod._stop_requested = True
        try:
            _json_emit({"type": "exit"})
        except Exception:
            pass

    signal.signal(signal.SIGTERM, _sigterm)
    signal.signal(signal.SIGINT, _sigterm)

    try:
        rc = mod.run()
    finally:
        try:
            mod._stop_audio()
        except Exception:
            pass
        # best-effort cleanup wav
        try:
            if mod._wav_path and os.path.exists(mod._wav_path):
                os.remove(mod._wav_path)
        except Exception:
            pass

    return rc


if __name__ == "__main__":
    # Never allow tracebacks to stdout
    try:
        sys.exit(main())
    except BaseException as e:
        try:
            _json_emit({"type": "fatal", "message": _safe_message_from_exc(e)})
            _json_emit({"type": "exit"})
        except Exception:
            pass
        sys.exit(1)
