#!/usr/bin/env python3
"""
BlackBox - Noise Generator (headless module)

RULES:
- Never touch OLED / luma.oled
- Never print non-JSON to stdout (no debug prints, no tracebacks)
- Non-blocking stdin (selectors + os.read)
- Heartbeat JSON "state" at least every 250ms
- Audio playback via background process (pw-play/paplay/aplay) using a short looped WAV
- Exit immediately on 'back' (stop playback first)
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
import wave
import random
from typing import Optional, Dict, Any, List

MODULE_NAME = "noise_generator"
MODULE_VERSION = "ng_v1"

HEARTBEAT_S = 0.25
TICK_S = 0.05

SAMPLE_RATE = 44100
CHANNELS = 2
SAMPLE_WIDTH = 2  # int16
WAV_SECONDS = 2.0
TMP_PREFIX = "blackbox_noise_"

MODES: List[str] = ["white", "pink", "brown"]
MODE_LABEL = {"white": "White", "pink": "Pink", "brown": "Brown"}


def _emit(obj: Dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(obj, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def _safe_err(e: BaseException) -> str:
    s = (str(e) or e.__class__.__name__).strip()
    return s[:160]


def _which(cmd: str) -> Optional[str]:
    for p in os.environ.get("PATH", "").split(":"):
        fp = os.path.join(p, cmd)
        if os.path.isfile(fp) and os.access(fp, os.X_OK):
            return fp
    return None


def _clamp(v: int, lo: int, hi: int) -> int:
    return lo if v < lo else hi if v > hi else v


def _int16(x: float) -> int:
    if x < -1.0:
        x = -1.0
    elif x > 1.0:
        x = 1.0
    return int(x * 32767.0)


def _gen_white(n: int, rng: random.Random) -> List[float]:
    return [rng.uniform(-1.0, 1.0) for _ in range(n)]


def _gen_pink(n: int, rng: random.Random) -> List[float]:
    # Voss-McCartney style; lightweight
    rows_n = 16
    rows = [rng.uniform(-1.0, 1.0) for _ in range(rows_n)]
    s = sum(rows)
    out: List[float] = []
    counter = 0

    for _ in range(n):
        counter += 1
        c = counter
        row = 0
        while (c & 1) == 0 and row < rows_n:
            c >>= 1
            row += 1
        if row < rows_n:
            s -= rows[row]
            rows[row] = rng.uniform(-1.0, 1.0)
            s += rows[row]
        white = rng.uniform(-1.0, 1.0)
        out.append((s + white) / (rows_n + 1))
    return out


def _gen_brown(n: int, rng: random.Random) -> List[float]:
    out: List[float] = []
    x = 0.0
    for _ in range(n):
        x += rng.uniform(-1.0, 1.0) * 0.02
        x *= 0.999
        out.append(x)
    mx = max(1e-9, max(abs(v) for v in out))
    scale = 1.0 / mx
    return [v * scale for v in out]


def _write_wav(path: str, mode: str, volume_pct: int, seconds: float) -> None:
    n = int(SAMPLE_RATE * seconds)
    # deterministic seed, but changes across mode/volume for variety
    rng = random.Random(0xBB0X if False else (0xB10C0 + volume_pct + (MODES.index(mode) * 1337)))

    if mode == "white":
        mono = _gen_white(n, rng)
    elif mode == "pink":
        mono = _gen_pink(n, rng)
    else:
        mono = _gen_brown(n, rng)

    amp = (volume_pct / 100.0) * 0.8  # headroom

    with wave.open(path, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(SAMPLE_WIDTH)
        wf.setframerate(SAMPLE_RATE)

        frames = bytearray()
        for s in mono:
            v = _int16(s * amp)
            b = int(v).to_bytes(2, "little", signed=True)
            frames += b
            frames += b
        wf.writeframes(frames)


class AudioLoop:
    """
    Loop a WAV via pw-play, paplay, or aplay.
    Looping is handled by /bin/sh while true; player wav; done
    Stdout/stderr are suppressed to keep module stdout JSON-only.
    """

    def __init__(self) -> None:
        self.player = None
        self.player_path = None
        for c in ("pw-play", "paplay", "aplay"):
            p = _which(c)
            if p:
                self.player = c
                self.player_path = p
                break
        self.proc: Optional[subprocess.Popen] = None
        self.wav_path: Optional[str] = None

    def available(self) -> bool:
        return self.player_path is not None

    def start(self, wav_path: str) -> None:
        self.stop()
        self.wav_path = wav_path
        if not self.available():
            raise RuntimeError("No audio player found (pw-play/paplay/aplay)")

        if self.player == "pw-play":
            loop_cmd = f'while true; "{self.player_path}" "{wav_path}" >/dev/null 2>&1; done'
        elif self.player == "paplay":
            loop_cmd = f'while true; "{self.player_path}" "{wav_path}" >/dev/null 2>&1; done'
        else:
            loop_cmd = f'while true; "{self.player_path}" -q "{wav_path}" >/dev/null 2>&1; done'

        self.proc = subprocess.Popen(
            ["/bin/sh", "-c", loop_cmd],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            preexec_fn=os.setsid,
            close_fds=True,
        )

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, signal.SIGTERM)
            except Exception:
                pass
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


class NoiseModule:
    def __init__(self) -> None:
        self.sel = selectors.DefaultSelector()
        self.stdin_fd = sys.stdin.fileno()
        os.set_blocking(self.stdin_fd, False)
        self.sel.register(self.stdin_fd, selectors.EVENT_READ)

        self.audio = AudioLoop()
        self._stop = False
        self._fatal: Optional[str] = None

        self.page = "main"
        self.mode_idx = 0
        self.mode = MODES[self.mode_idx]
        self.playing = False
        self.volume = 70
        self.loop = True

        self.focus = "mode"  # "mode" or "volume"
        self._stdin_buf = b""

        self._wav_path = os.path.join(tempfile.gettempdir(), f"{TMP_PREFIX}{os.getpid()}.wav")

        self._last_state_emit = 0.0

    def hello(self) -> None:
        _emit({"type": "hello", "module": MODULE_NAME, "version": MODULE_VERSION})
        _emit({"type": "page", "name": self.page})
        self.emit_state(force=True)

    def toast(self, msg: str) -> None:
        _emit({"type": "toast", "message": str(msg)[:160]})

    def fatal(self, msg: str) -> None:
        self._fatal = str(msg)[:160]
        _emit({"type": "fatal", "message": self._fatal})

    def emit_state(self, force: bool = False) -> None:
        now = time.monotonic()
        if not force and (now - self._last_state_emit) < HEARTBEAT_S:
            return
        self._last_state_emit = now
        _emit({
            "type": "state",
            "ready": bool(self._fatal is None),
            "mode": self.mode,
            "playing": bool(self.playing),
            "volume": int(self.volume),
            "duration_s": 0,
            "loop": bool(self.loop),
            "focus": self.focus,
        })

    def _ensure_wav(self) -> None:
        _write_wav(self._wav_path, self.mode, self.volume, WAV_SECONDS)

    def _start_audio(self) -> None:
        if self.playing:
            return
        if not self.audio.available():
            raise RuntimeError("Audio backend not available (need pw-play/paplay/aplay)")
        self._ensure_wav()
        self.audio.start(self._wav_path)
        self.playing = True

    def _stop_audio(self) -> None:
        self.audio.stop()
        self.playing = False

    def _restart_if_playing(self) -> None:
        if not self.playing:
            return
        self._stop_audio()
        self._start_audio()

    def _change_mode(self, delta: int) -> None:
        self.mode_idx = (self.mode_idx + delta) % len(MODES)
        self.mode = MODES[self.mode_idx]
        self.toast(f"Mode: {MODE_LABEL.get(self.mode, self.mode.title())}")
        self._restart_if_playing()

    def _change_volume(self, delta: int) -> None:
        self.volume = _clamp(self.volume + delta, 0, 100)
        self.toast(f"Volume: {self.volume}")
        self._restart_if_playing()

    def handle(self, cmd: str) -> None:
        if cmd == "back":
            try:
                self._stop_audio()
            except Exception:
                pass
            self._stop = True
            _emit({"type": "exit"})
            return

        # after fatal, ignore everything except back
        if self._fatal is not None:
            return

        if cmd == "select":
            try:
                if self.playing:
                    self._stop_audio()
                    self.toast("STOP")
                else:
                    self._start_audio()
                    self.toast("PLAY")
            except Exception as e:
                self.fatal(_safe_err(e))
            finally:
                self.emit_state(force=True)
            return

        if cmd == "select_hold":
            self.focus = "volume" if self.focus == "mode" else "mode"
            self.toast(f"Adjust: {self.focus}")
            self.emit_state(force=True)
            return

        if cmd == "up":
            if self.focus == "mode":
                self._change_mode(+1)
            else:
                self._change_volume(+5)
            self.emit_state(force=True)
            return

        if cmd == "down":
            if self.focus == "mode":
                self._change_mode(-1)
            else:
                self._change_volume(-5)
            self.emit_state(force=True)
            return

        # ignore unknown cmd

    def _read_stdin(self) -> None:
        try:
            while True:
                try:
                    chunk = os.read(self.stdin_fd, 4096)
                except OSError as e:
                    if e.errno in (errno.EAGAIN, errno.EWOULDBLOCK):
                        break
                    raise
                if not chunk:
                    # stdin closed
                    self._stop = True
                    return
                self._stdin_buf += chunk
                while b"\n" in self._stdin_buf:
                    line, self._stdin_buf = self._stdin_buf.split(b"\n", 1)
                    cmd = line.decode("utf-8", errors="ignore").strip()
                    if cmd:
                        self.handle(cmd)
                        if self._stop:
                            return
        except Exception as e:
            self.fatal(f"stdin error: {_safe_err(e)}")

    def run(self) -> int:
        self.hello()

        while not self._stop:
            try:
                events = self.sel.select(timeout=TICK_S)
                for key, mask in events:
                    if (mask & selectors.EVENT_READ) and key.fileobj == self.stdin_fd:
                        self._read_stdin()

                # detect unexpected audio death
                if self.playing and self.audio.proc is not None and self.audio.proc.poll() is not None:
                    self.playing = False
                    self.fatal("Audio playback stopped unexpectedly")
                    self.emit_state(force=True)

                # heartbeat
                self.emit_state(force=False)

            except Exception as e:
                # never crash to stdout traceback
                self.fatal(_safe_err(e))
                self.emit_state(force=True)

        return 0


def main() -> int:
    mod = NoiseModule()

    def _sig(_signum, _frame):
        try:
            mod._stop_audio()
        except Exception:
            pass
        mod._stop = True
        try:
            _emit({"type": "exit"})
        except Exception:
            pass

    signal.signal(signal.SIGTERM, _sig)
    signal.signal(signal.SIGINT, _sig)

    try:
        return mod.run()
    finally:
        try:
            mod._stop_audio()
        except Exception:
            pass
        try:
            if os.path.exists(mod._wav_path):
                os.remove(mod._wav_path)
        except Exception:
            pass


if __name__ == "__main__":
    try:
        sys.exit(main())
    except BaseException as e:
        try:
            _emit({"type": "fatal", "message": _safe_err(e)})
            _emit({"type": "exit"})
        except Exception:
            pass
        sys.exit(1)
