#!/usr/bin/env python3
"""
BlackBox - Noise Generator (headless module) - pulsed spirit-box style (no volume)

Changes in this revision:
- Volume removed completely (speaker handles volume)
- Clean 3-row main UI: Noise Type / Sweep Rate / Play
- Noise selection is now a real menu page:
    * select on Noise Type opens noise_menu_cycle (select cycles + applies live)
    * select_hold on Noise Type opens noise_menu_scroll (up/down scroll, select confirms)
- No toast overlays required for normal operation
- JSON-only stdout; non-blocking stdin; heartbeat <=250ms
- Audio uses pattern.wav loop (single background /bin/sh loop)
- shell+player stderr -> /tmp/blackbox_noise_audio.err
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
MODULE_VERSION = "ng_v5_menus_novol"

HEARTBEAT_S = 0.25
TICK_S = 0.05

PATTERN_SECONDS = 8.0
PULSE_DUTY = 0.90  # higher = denser sweep

SAMPLE_RATE = 44100
CHANNELS = 2
SAMPLE_WIDTH = 2  # int16
AMP = 0.85  # fixed; speaker handles volume

AUDIO_ERR_LOG = "/tmp/blackbox_noise_audio.err"
TMP_PREFIX = "blackbox_noise_"

PULSE_OPTIONS_MS = [150, 200, 250, 300]
NOISE_TYPES: List[str] = ["white", "pink", "brown"]
NOISE_LABEL = {"white": "White", "pink": "Pink", "brown": "Brown"}

# Main rows
ROWS = ["noise", "rate", "play"]  # Noise Type, Sweep Rate, Play


def _emit(obj: Dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(obj, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def _safe_err(e: BaseException) -> str:
    s = (str(e) or e.__class__.__name__).strip()
    return s[:220]


def _which(cmd: str) -> Optional[str]:
    for p in os.environ.get("PATH", "").split(":"):
        fp = os.path.join(p, cmd)
        if os.path.isfile(fp) and os.access(fp, os.X_OK):
            return fp
    return None


def _int16(x: float) -> int:
    if x < -1.0:
        x = -1.0
    elif x > 1.0:
        x = 1.0
    return int(x * 32767.0)


def _gen_white(n: int, rng: random.Random) -> List[float]:
    return [rng.uniform(-1.0, 1.0) for _ in range(n)]


def _gen_pink(n: int, rng: random.Random) -> List[float]:
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


def _apply_fade(mono: List[float], fade_ms: int) -> List[float]:
    if not mono:
        return mono
    fade_n = int(SAMPLE_RATE * (fade_ms / 1000.0))
    fade_n = max(1, min(fade_n, len(mono) // 3))
    N = len(mono)
    out = mono[:]
    for i in range(N):
        g = 1.0
        if i < fade_n:
            g = i / float(fade_n)
        elif i >= (N - fade_n):
            g = (N - 1 - i) / float(fade_n)
            if g < 0.0:
                g = 0.0
        out[i] = out[i] * g
    return out


def _write_wav(path: str, mono: List[float]) -> None:
    with wave.open(path, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(SAMPLE_WIDTH)
        wf.setframerate(SAMPLE_RATE)

        frames = bytearray()
        for s in mono:
            v = _int16(s * AMP)
            b = int(v).to_bytes(2, "little", signed=True)
            frames += b
            frames += b
        wf.writeframes(frames)


class AudioLoop:
    """
    Plays a single WAV file in a forever loop using one /bin/sh process.
    Captures shell+player stderr to AUDIO_ERR_LOG.
    """

    def __init__(self) -> None:
        for c in ("paplay", "pw-play", "aplay"):
            p = _which(c)
            if p:
                self.player = c
                self.player_path = p
                break
        else:
            self.player = None
            self.player_path = None

        self.proc: Optional[subprocess.Popen] = None
        self.last_exit_code: Optional[int] = None

    def available(self) -> bool:
        return self.player_path is not None

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

        if self.proc and self.proc.poll() is not None:
            self.last_exit_code = self.proc.returncode
        self.proc = None

    def died(self) -> bool:
        return self.proc is not None and (self.proc.poll() is not None)

    def _err_tail(self) -> str:
        try:
            if os.path.exists(AUDIO_ERR_LOG):
                with open(AUDIO_ERR_LOG, "rb") as f:
                    data = f.read()[-2048:]
                txt = data.decode("utf-8", errors="ignore").strip()
                return txt[-240:]
        except Exception:
            pass
        return ""

    def died_reason(self) -> str:
        rc = self.proc.returncode if self.proc else self.last_exit_code
        tail = self._err_tail()
        if rc is None:
            return "unknown"
        return f"rc={rc} {tail}".strip() if tail else f"rc={rc}"

    def _player_cmd(self, wav_path: str) -> str:
        if self.player == "pw-play":
            return f'"{self.player_path}" "{wav_path}"'
        if self.player == "paplay":
            return f'"{self.player_path}" "{wav_path}"'
        return f'"{self.player_path}" -q "{wav_path}"'

    def start_continuous(self, wav_path: str) -> None:
        self.stop()
        self.last_exit_code = None

        if not self.available():
            raise RuntimeError("No audio player found (paplay/pw-play/aplay)")

        # clear error log
        try:
            with open(AUDIO_ERR_LOG, "w") as f:
                f.write("")
        except Exception:
            pass

        play_cmd = self._player_cmd(wav_path)

        # capture shell stderr too
        loop_cmd = (
            f'exec 2>>"{AUDIO_ERR_LOG}"; '
            f'while true; do {play_cmd} 1>/dev/null; done'
        )

        self.proc = subprocess.Popen(
            ["/bin/sh", "-c", loop_cmd],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            preexec_fn=os.setsid,
            close_fds=True,
        )

        time.sleep(0.05)
        if self.proc.poll() is not None:
            self.last_exit_code = self.proc.returncode
            raise RuntimeError(f"Audio loop failed ({self.died_reason()})")


class NoiseModule:
    def __init__(self) -> None:
        self.sel = selectors.DefaultSelector()
        self.stdin_fd = sys.stdin.fileno()
        os.set_blocking(self.stdin_fd, False)
        self.sel.register(self.stdin_fd, selectors.EVENT_READ)

        self.audio = AudioLoop()
        self._stop = False
        self._fatal: Optional[str] = None

        self.page = "main"  # main | noise_menu_cycle | noise_menu_scroll | fatal

        self.noise_idx = 0
        self.noise_type = NOISE_TYPES[self.noise_idx]

        self.pulse_idx = 1  # 200ms
        self.pulse_ms = PULSE_OPTIONS_MS[self.pulse_idx]

        self.playing = False

        self.cursor_idx = 0
        self.cursor = ROWS[self.cursor_idx]

        # menu selection index (for scroll menu)
        self.menu_noise_idx = self.noise_idx

        self._stdin_buf = b""
        self._last_state_emit = 0.0

        self._tmpdir = tempfile.gettempdir()
        self._pattern_path = os.path.join(self._tmpdir, f"{TMP_PREFIX}{os.getpid()}_pattern.wav")

    def hello(self) -> None:
        _emit({"type": "hello", "module": MODULE_NAME, "version": MODULE_VERSION})
        _emit({"type": "page", "name": self.page})
        self.emit_state(force=True)

    def set_page(self, name: str) -> None:
        self.page = name
        _emit({"type": "page", "name": self.page})
        self.emit_state(force=True)

    def fatal(self, msg: str) -> None:
        self._fatal = str(msg)[:220]
        _emit({"type": "fatal", "message": self._fatal})
        self.set_page("fatal")

    def emit_state(self, force: bool = False) -> None:
        now = time.monotonic()
        if not force and (now - self._last_state_emit) < HEARTBEAT_S:
            return
        self._last_state_emit = now
        _emit({
            "type": "state",
            "ready": (self._fatal is None),
            "page": self.page,
            "noise_type": self.noise_type,
            "pulse_ms": int(self.pulse_ms),
            "playing": bool(self.playing),
            "cursor": self.cursor,             # main page cursor: noise|rate|play
            "menu_noise_idx": int(self.menu_noise_idx),  # for noise_menu_scroll rendering
        })

    def _write_pattern(self) -> None:
        total_n = int(SAMPLE_RATE * PATTERN_SECONDS)
        pulse_n = max(1, int(SAMPLE_RATE * (self.pulse_ms / 1000.0)))
        on_n = max(1, int(pulse_n * PULSE_DUTY))

        # seed stable per noise_type + pulse_ms so changing selections changes texture
        rng = random.Random(0x515745 + (NOISE_TYPES.index(self.noise_type) * 1337) + (self.pulse_ms * 17))

        mono = [0.0] * total_n
        i = 0
        while i < total_n:
            n = min(on_n, total_n - i)
            if self.noise_type == "white":
                seg = _gen_white(n, rng)
            elif self.noise_type == "pink":
                seg = _gen_pink(n, rng)
            else:
                seg = _gen_brown(n, rng)

            seg = [max(-1.0, min(1.0, x * 0.92)) for x in seg]
            seg = _apply_fade(seg, fade_ms=8)

            mono[i:i+n] = seg
            i += pulse_n  # silence between pulses
        _write_wav(self._pattern_path, mono)

    def _start_audio(self) -> None:
        if self.playing:
            return
        if not self.audio.available():
            raise RuntimeError("Audio backend not available (need paplay/pw-play/aplay)")
        self._write_pattern()
        self.audio.start_continuous(self._pattern_path)
        self.playing = True

    def _stop_audio(self) -> None:
        self.audio.stop()
        self.playing = False

    def _restart_if_playing(self) -> None:
        if not self.playing:
            return
        self._stop_audio()
        self._start_audio()

    def _move_cursor(self, delta: int) -> None:
        self.cursor_idx = (self.cursor_idx + delta) % len(ROWS)
        self.cursor = ROWS[self.cursor_idx]
        self.emit_state(force=True)

    def _cycle_rate(self, delta: int) -> None:
        self.pulse_idx = (self.pulse_idx + delta) % len(PULSE_OPTIONS_MS)
        self.pulse_ms = PULSE_OPTIONS_MS[self.pulse_idx]
        self._restart_if_playing()
        self.emit_state(force=True)

    def _apply_noise_idx(self, idx: int) -> None:
        self.noise_idx = idx % len(NOISE_TYPES)
        self.noise_type = NOISE_TYPES[self.noise_idx]
        self.menu_noise_idx = self.noise_idx
        self._restart_if_playing()
        self.emit_state(force=True)

    # ---- Page handlers ----

    def open_noise_menu_cycle(self) -> None:
        # live-apply cycling with select; no up/down
        self.menu_noise_idx = self.noise_idx
        self.set_page("noise_menu_cycle")

    def open_noise_menu_scroll(self) -> None:
        # up/down scroll; select confirm; back cancel
        self.menu_noise_idx = self.noise_idx
        self.set_page("noise_menu_scroll")

    def handle_main(self, cmd: str) -> None:
        if cmd == "up":
            self._move_cursor(-1)
            return
        if cmd == "down":
            self._move_cursor(+1)
            return

        if cmd == "select":
            if self.cursor == "noise":
                self.open_noise_menu_cycle()
                return
            if self.cursor == "rate":
                self._cycle_rate(+1)
                return
            # play
            if self.playing:
                self._stop_audio()
            else:
                self._start_audio()
            self.emit_state(force=True)
            return

        if cmd == "select_hold":
            if self.cursor == "noise":
                self.open_noise_menu_scroll()
                return
            if self.cursor == "rate":
                self._cycle_rate(-1)
                return
            # play row hold = stop
            self._stop_audio()
            self.emit_state(force=True)
            return

    def handle_noise_menu_cycle(self, cmd: str) -> None:
        # select cycles & applies immediately; back returns to main; hold = open scroll menu
        if cmd == "select":
            self._apply_noise_idx(self.noise_idx + 1)
            return
        if cmd == "select_hold":
            # user requested hold = open menu scrollable; allow switching without leaving menu layer
            self.open_noise_menu_scroll()
            return
        if cmd == "back":
            self.set_page("main")
            return
        # ignore up/down in cycle menu to prevent accidental navigation

    def handle_noise_menu_scroll(self, cmd: str) -> None:
        if cmd == "up":
            self.menu_noise_idx = (self.menu_noise_idx - 1) % len(NOISE_TYPES)
            self.emit_state(force=True)
            return
        if cmd == "down":
            self.menu_noise_idx = (self.menu_noise_idx + 1) % len(NOISE_TYPES)
            self.emit_state(force=True)
            return
        if cmd == "select":
            self._apply_noise_idx(self.menu_noise_idx)
            self.set_page("main")
            return
        if cmd == "back":
            # cancel
            self.menu_noise_idx = self.noise_idx
            self.set_page("main")
            return
        if cmd == "select_hold":
            # quick cancel back to main
            self.menu_noise_idx = self.noise_idx
            self.set_page("main")
            return

    def handle(self, cmd: str) -> None:
        if cmd == "back" and self.page == "main":
            try:
                self._stop_audio()
            except Exception:
                pass
            self._stop = True
            _emit({"type": "exit"})
            return

        if self._fatal is not None and cmd != "back":
            return

        try:
            if self.page == "main":
                self.handle_main(cmd)
            elif self.page == "noise_menu_cycle":
                self.handle_noise_menu_cycle(cmd)
            elif self.page == "noise_menu_scroll":
                self.handle_noise_menu_scroll(cmd)
            elif self.page == "fatal":
                if cmd == "back":
                    try:
                        self._stop_audio()
                    except Exception:
                        pass
                    self._stop = True
                    _emit({"type": "exit"})
        except Exception as e:
            self.fatal(_safe_err(e))

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

                # watchdog audio
                if self.playing and self.audio.died():
                    reason = self.audio.died_reason()
                    self.playing = False
                    recovered = False
                    for _ in range(3):
                        try:
                            time.sleep(0.15)
                            self._start_audio()
                            recovered = True
                            break
                        except Exception:
                            recovered = False
                    if not recovered:
                        self.fatal(f"Audio playback stopped ({reason})")
                    self.emit_state(force=True)

                self.emit_state(force=False)

            except Exception as e:
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
            if mod._pattern_path and os.path.exists(mod._pattern_path):
                os.remove(mod._pattern_path)
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
