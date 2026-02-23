#!/usr/bin/env python3
"""
BlackBox Spirit Box (headless JSON stdout protocol) - sb_v2 (REAL)

Based on sb_v1 structure:
- NO OLED access
- JSON-only stdout (never print debug)
- Non-blocking stdin (selectors + os.read)
- REAL tuner sweep (TEA controller; includes TEA5767 I2C implementation + safe stub)
- REAL ALSA capture from PCM1808 (hw:0,0) @ 48k stereo S16_LE
- Ring buffer + simple level meter for OLED activity
- Optional snippet record to WAV from ring buffer
- Detect arecord crash + restart 2-3 times; then fatal
- Heartbeat state <=250ms
- Clean exit on back, and on SIGTERM

NEW in this replacement:
- Optional Bluetooth playback via PipeWire/Pulse default sink:
  arecord (raw) -> paplay --raw ... /dev/stdin
- Robust fallback: if BT disconnects / paplay dies, capture continues.

UI surface (handled by app.py):
Main page:
  Spirit Box
  Sweep Rate: 150/200/250/300 ms
  Direction: FWD/REV
  Mode: Scan/Burst (Burst = future; state supported)
  Play: PLAY/STOP

Controls:
  up/down: move cursor
  select: change value (forward)
  select_hold: change value (reverse)
  back: exit immediately

Note:
- "Play" toggles Sweep+Capture (+ Playback if enabled).
- Adds state fields: freq_mhz, level, tea_ok, alsa_ok
"""

import os
import sys
import json
import time
import wave
import errno
import signal
import selectors
import subprocess
import threading
from dataclasses import dataclass
from typing import Optional, List, Deque
from collections import deque

MODULE_NAME = "spirit_box"
MODULE_VERSION = "sb_v2_real_bt"

MODULE_ERR = "/tmp/blackbox_spirit_module.err"
AUDIO_ERR  = "/tmp/blackbox_spirit_audio.err"
SNAP_WAV   = "/tmp/blackbox_spirit_snap.wav"

HEARTBEAT_S = 0.25
TICK_S = 0.02  # <=50ms tick

SWEEP_MS_CHOICES = [150, 200, 250, 300]
DIR_CHOICES = ["fwd", "rev"]
MODE_CHOICES = ["scan", "burst"]  # burst is future expansion
CURSOR_CHOICES = ["rate", "direction", "mode", "play"]

# Real capture
ALSA_DEVICE = "hw:CARD=Device,DEV=0"  # or "hw:2,0"
CAP_RATE = 48000
CAP_CHANNELS = 2
CAP_FORMAT = "S16_LE"
SAMPLE_BYTES = 2
FRAME_BYTES = CAP_CHANNELS * SAMPLE_BYTES

CHUNK_FRAMES = 1024
CHUNK_BYTES = CHUNK_FRAMES * FRAME_BYTES

RING_SECONDS = 5  # keep last N seconds for snapshots

# Tuner sweep defaults (FM band)
FM_MIN = 87.5
FM_MAX = 108.0
FM_STEP = 0.2  # MHz per step; tweak to match your desired "chop"


# ---------------- logging (file only) ----------------
def _log_err(msg: str) -> None:
    try:
        with open(MODULE_ERR, "a", encoding="utf-8") as f:
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")
    except Exception:
        pass


# ---------------- strict JSON stdout ----------------
def _emit(obj: dict) -> None:
    try:
        sys.stdout.write(json.dumps(obj, separators=(",", ":")) + "\n")
        sys.stdout.flush()
    except Exception:
        pass

def _toast(msg: str) -> None:
    _emit({"type": "toast", "message": msg})

def _fatal(msg: str) -> None:
    _emit({"type": "fatal", "message": msg})


# ---------------- stdin reader (non-blocking) ----------------
class StdinReader:
    def __init__(self) -> None:
        self.fd = sys.stdin.fileno()
        os.set_blocking(self.fd, False)
        self.sel = selectors.DefaultSelector()
        self.sel.register(self.fd, selectors.EVENT_READ)
        self.buf = bytearray()

    def close(self) -> None:
        try:
            self.sel.unregister(self.fd)
        except Exception:
            pass
        try:
            self.sel.close()
        except Exception:
            pass

    def read_commands(self, max_bytes: int = 4096) -> List[str]:
        out: List[str] = []
        if not self.sel.select(timeout=0):
            return out

        drained = 0
        while drained < max_bytes:
            try:
                chunk = os.read(self.fd, min(1024, max_bytes - drained))
            except BlockingIOError:
                break
            except OSError as e:
                if e.errno in (errno.EAGAIN, errno.EWOULDBLOCK):
                    break
                break

            if not chunk:
                break

            drained += len(chunk)
            self.buf.extend(chunk)

            while b"\n" in self.buf:
                line, _, rest = self.buf.partition(b"\n")
                self.buf = bytearray(rest)
                try:
                    s = line.decode("utf-8", errors="ignore").strip().lower()
                except Exception:
                    continue
                if s:
                    out.append(s)

        return out


# ---------------- ring buffer (raw PCM) ----------------
class PCMRing:
    def __init__(self, max_bytes: int):
        self.max_bytes = max_bytes
        self.q: Deque[bytes] = deque()
        self.size = 0
        self.lock = threading.Lock()

    def push(self, data: bytes) -> None:
        with self.lock:
            self.q.append(data)
            self.size += len(data)
            while self.size > self.max_bytes and self.q:
                d = self.q.popleft()
                self.size -= len(d)

    def snapshot_last_seconds(self, seconds: float) -> bytes:
        want = int(seconds * CAP_RATE * FRAME_BYTES)
        with self.lock:
            if want <= 0:
                return b""
            out = bytearray()
            for block in reversed(self.q):
                out[:0] = block  # prepend
                if len(out) >= want:
                    break
            if len(out) > want:
                out = out[-want:]
            return bytes(out)


# ---------------- Pulse playback (PipeWire/PulseAudio) ----------------
class PulsePlayback:
    """
    Plays raw PCM to the default Pulse sink (PipeWire's PulseAudio server).
    Your system default sink is BT (bluez_output...), so this becomes BT playback automatically.
    """
    def __init__(self):
        self.proc: Optional[subprocess.Popen] = None
        self.ok = False
        self._last_start = 0.0

    def start(self) -> None:
        # simple backoff to avoid tight restart loops if BT is missing
        now = time.monotonic()
        if now - self._last_start < 0.5:
            return
        self._last_start = now

        if self.proc and self.proc.poll() is None:
            self.ok = True
            return

        # IMPORTANT: paplay expects a path; /dev/stdin works for piped audio.
        cmd = [
            "paplay",
            "--raw",
            "--rate=48000",
            "--channels=2",
            "--format=s16le",
            "/dev/stdin",
        ]
        try:
            self.proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                bufsize=0,
                close_fds=True,
            )
            self.ok = True
        except Exception as e:
            _log_err(f"paplay_start_failed: {e!r}")
            self.proc = None
            self.ok = False

    def stop(self) -> None:
        p = self.proc
        self.proc = None
        self.ok = False
        if not p:
            return
        try:
            if p.poll() is None:
                try:
                    if p.stdin:
                        p.stdin.close()
                except Exception:
                    pass
                try:
                    p.terminate()
                except Exception:
                    pass
        except Exception:
            pass

    def write(self, data: bytes) -> None:
        p = self.proc
        if not p or p.poll() is not None:
            self.ok = False
            return
        try:
            if not p.stdin:
                self.ok = False
                return
            p.stdin.write(data)
            # no need to flush every chunk; pipe is unbuffered-ish with bufsize=0
        except Exception:
            self.ok = False


# ---------------- ALSA capture (arecord raw) ----------------
class ALSACapture:
    def __init__(self):
        self.proc: Optional[subprocess.Popen] = None
        self.t: Optional[threading.Thread] = None
        self.stop_ev = threading.Event()

        max_bytes = int(RING_SECONDS * CAP_RATE * FRAME_BYTES)
        self.ring = PCMRing(max_bytes=max_bytes)

        self.level = 0       # 0..100
        self.alsa_ok = False
        self._restart_tries = 0

        # Playback (optional)
        self.playback = PulsePlayback()
        self.play_audio = True  # set False if you want capture-only

    def start(self) -> None:
        if self.t and self.t.is_alive():
            return
        self.stop_ev.clear()
        self._spawn()
        if self.play_audio:
            self.playback.start()
        self.t = threading.Thread(target=self._loop, daemon=True)
        self.t.start()

    def stop(self) -> None:
        self.stop_ev.set()
        try:
            self.playback.stop()
        except Exception:
            pass
        self._stop_proc()
        if self.t:
            self.t.join(timeout=1.0)
        self.proc = None
        self.t = None
        self.alsa_ok = False

    def _spawn(self) -> None:
        try:
            open(AUDIO_ERR, "a").close()
        except Exception:
            pass
        cmd = [
            "arecord",
            "-D", ALSA_DEVICE,
            "-f", CAP_FORMAT,
            "-r", str(CAP_RATE),
            "-c", str(CAP_CHANNELS),
            "-t", "raw",
        ]
        self.proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=open(AUDIO_ERR, "ab", buffering=0),
            bufsize=0,
            close_fds=True,
        )

    def _stop_proc(self) -> None:
        p = self.proc
        if not p:
            return
        try:
            if p.poll() is None:
                try:
                    p.terminate()
                except Exception:
                    pass
                t0 = time.time()
                while time.time() - t0 < 0.20:
                    if p.poll() is not None:
                        break
                    time.sleep(0.01)
            if p.poll() is None:
                try:
                    p.kill()
                except Exception:
                    pass
        except Exception:
            pass

    def _loop(self) -> None:
        # keep capturing; restart arecord up to 3 times if it dies
        self._restart_tries = 0

        while not self.stop_ev.is_set():
            if not self.proc:
                self.alsa_ok = False
                return

            try:
                out = self.proc.stdout
                if out is None:
                    raise RuntimeError("arecord stdout missing")

                self.alsa_ok = True
                data = out.read(CHUNK_BYTES)
                if not data:
                    raise RuntimeError("arecord produced no data")

                self.level = _compute_level_fast(data)
                self.ring.push(data)

                # ---- BT playback (PipeWire/Pulse default sink) ----
                if self.play_audio:
                    if not self.playback.ok:
                        self.playback.start()
                    if self.playback.ok:
                        self.playback.write(data)

            except Exception as e:
                self.alsa_ok = False
                rc = self.proc.poll() if self.proc else None
                _log_err(f"arecord_error rc={rc} err={e!r}")

                try:
                    self.playback.stop()
                except Exception:
                    pass

                self._stop_proc()
                self.proc = None

                self._restart_tries += 1
                if self._restart_tries <= 3 and not self.stop_ev.is_set():
                    time.sleep(0.10)
                    try:
                        self._spawn()
                        if self.play_audio:
                            self.playback.start()
                    except Exception as e2:
                        _log_err(f"arecord_respawn_failed: {e2!r}")
                        break
                else:
                    break


def _compute_level_fast(pcm: bytes) -> int:
    # peak meter from int16, stride for low CPU
    if len(pcm) < 4:
        return 0
    peak = 0
    step = 8 * FRAME_BYTES
    for i in range(0, len(pcm) - 1, step):
        s = int.from_bytes(pcm[i:i+2], "little", signed=True)
        a = -s if s < 0 else s
        if a > peak:
            peak = a
    return int(min(100, (peak / 32767.0) * 100))


def _write_wav(path: str, pcm: bytes) -> None:
    tmp = path + ".tmp"
    with wave.open(tmp, "wb") as wf:
        wf.setnchannels(CAP_CHANNELS)
        wf.setsampwidth(SAMPLE_BYTES)
        wf.setframerate(CAP_RATE)
        wf.writeframes(pcm)
    os.replace(tmp, path)


# ---------------- TEA tuner (real impl + stub) ----------------
class TEATunerBase:
    def probe(self) -> bool:
        return False
    def set_freq_mhz(self, mhz: float) -> bool:
        return False
    def mute(self, on: bool) -> None:
        pass

class TEATunerStub(TEATunerBase):
    def probe(self) -> bool:
        return False
    def set_freq_mhz(self, mhz: float) -> bool:
        return True  # pretend success

class TEA5767I2C(TEATunerBase):
    """
    TEA5767 FM tuner is commonly I2C addr 0x60.
    This uses /dev/i2c-1. If your TEA is different, keep the interface and swap the backend.
    """
    def __init__(self, bus: int = 1, addr: int = 0x60):
        self.bus = bus
        self.addr = addr
        self.fd: Optional[int] = None

    def probe(self) -> bool:
        try:
            self._open()
            os.read(self.fd, 5)
            return True
        except Exception:
            self._close()
            return False

    def set_freq_mhz(self, mhz: float) -> bool:
        try:
            self._open()
            f_rf = float(mhz) * 1_000_000.0
            IF = 225_000.0
            f_ref = 32_768.0
            pll = int((4.0 * (f_rf + IF)) / f_ref) & 0x3FFF

            b0 = ((pll >> 8) & 0x3F)
            b1 = pll & 0xFF
            b0 |= 0x00
            b2 = 0xB0
            b3 = 0x10
            b4 = 0x00

            os.write(self.fd, bytes([b0, b1, b2, b3, b4]))
            return True
        except Exception as e:
            _log_err(f"tea_set_freq_failed: {e!r}")
            self._close()
            return False

    def mute(self, on: bool) -> None:
        pass

    def _open(self) -> None:
        if self.fd is not None:
            return
        path = f"/dev/i2c-{self.bus}"
        self.fd = os.open(path, os.O_RDWR)
        import fcntl
        fcntl.ioctl(self.fd, 0x0703, self.addr)

    def _close(self) -> None:
        if self.fd is not None:
            try:
                os.close(self.fd)
            except Exception:
                pass
        self.fd = None


# ---------------- state ----------------
@dataclass
class UIState:
    page: str = "main"
    ready: bool = False

    sweep_ms: int = 200
    direction: str = "fwd"
    mode: str = "scan"
    playing: bool = False
    cursor: str = "rate"

    # new fields
    freq_mhz: float = 99.5
    level: int = 0
    tea_ok: bool = False
    alsa_ok: bool = False

    # internal
    _last_toast_t: float = 0.0
    _fatal_active: bool = False
    _last_step_t: float = 0.0


def _emit_page(st: UIState) -> None:
    _emit({"type": "page", "name": st.page})


def _emit_state(st: UIState) -> None:
    _emit({
        "type": "state",
        "ready": bool(st.ready),
        "sweep_ms": int(st.sweep_ms),
        "direction": str(st.direction),
        "mode": str(st.mode),
        "playing": bool(st.playing),
        "cursor": str(st.cursor),

        "freq_mhz": round(float(st.freq_mhz), 1),
        "level": int(st.level),
        "tea_ok": bool(st.tea_ok),
        "alsa_ok": bool(st.alsa_ok),
    })


def _toast_throttle(st: UIState, msg: str, min_interval_s: float = 0.10) -> None:
    now = time.monotonic()
    if now - st._last_toast_t >= min_interval_s:
        st._last_toast_t = now
        _toast(msg)


def _cycle_choice(cur: str, choices: List[str], delta: int) -> str:
    cur = str(cur).lower().strip()
    try:
        idx = choices.index(cur)
    except Exception:
        idx = 0
    idx = (idx + delta) % len(choices)
    return choices[idx]


def _cycle_int_choice(cur: int, choices: List[int], delta: int) -> int:
    try:
        idx = choices.index(int(cur))
    except Exception:
        idx = 0
    idx = (idx + delta) % len(choices)
    return int(choices[idx])


def _step_freq(freq: float, direction: str) -> float:
    if direction == "rev":
        f = freq - FM_STEP
        if f < FM_MIN:
            f = FM_MAX
        return f
    else:
        f = freq + FM_STEP
        if f > FM_MAX:
            f = FM_MIN
        return f


def main() -> int:
    exiting = {"flag": False}

    def _sig_handler(_signo, _frame):
        exiting["flag"] = True

    try:
        signal.signal(signal.SIGTERM, _sig_handler)
        signal.signal(signal.SIGINT, _sig_handler)
    except Exception:
        pass

    reader = StdinReader()
    st = UIState()

    _emit({"type": "hello", "module": MODULE_NAME, "version": MODULE_VERSION})
    _emit_page(st)

    # -------- initialize tuner backend --------
    tuner: TEATunerBase = TEA5767I2C(bus=1, addr=0x60)
    st.tea_ok = tuner.probe()
    if not st.tea_ok:
        tuner = TEATunerStub()
        st.tea_ok = False
        _log_err("TEA probe failed; using stub backend")

    # -------- initialize capture --------
    cap = ALSACapture()

    st.ready = True
    _emit_state(st)

    def start_real() -> None:
        try:
            ok = tuner.set_freq_mhz(st.freq_mhz)
            if st.tea_ok and not ok:
                _toast_throttle(st, "TEA tune failed")
        except Exception as e:
            _log_err(f"tuner_set_exc: {e!r}")
        try:
            cap.start()
        except Exception as e:
            _log_err(f"cap_start_exc: {e!r}")
        time.sleep(0.02)

    def stop_real() -> None:
        try:
            cap.stop()
        except Exception:
            pass

    last_hb = 0.0
    last_tick = time.monotonic()

    try:
        while not exiting["flag"]:
            now = time.monotonic()

            # ---------- process stdin (never blocking) ----------
            cmds = reader.read_commands()
            for cmd in cmds:
                if cmd == "back":
                    exiting["flag"] = True
                    break

                if cmd == "up":
                    try:
                        idx = CURSOR_CHOICES.index(st.cursor)
                    except Exception:
                        idx = 0
                    st.cursor = CURSOR_CHOICES[(idx - 1) % len(CURSOR_CHOICES)]
                    _emit_state(st)

                elif cmd == "down":
                    try:
                        idx = CURSOR_CHOICES.index(st.cursor)
                    except Exception:
                        idx = 0
                    st.cursor = CURSOR_CHOICES[(idx + 1) % len(CURSOR_CHOICES)]
                    _emit_state(st)

                elif cmd in ("select", "select_hold"):
                    delta = +1 if cmd == "select" else -1

                    if st.cursor == "rate":
                        st.sweep_ms = _cycle_int_choice(st.sweep_ms, SWEEP_MS_CHOICES, delta)
                        _toast_throttle(st, f"Sweep: {st.sweep_ms}ms")
                        _emit_state(st)

                    elif st.cursor == "direction":
                        st.direction = _cycle_choice(st.direction, DIR_CHOICES, delta)
                        _toast_throttle(st, f"Direction: {'REV' if st.direction=='rev' else 'FWD'}")
                        _emit_state(st)

                    elif st.cursor == "mode":
                        st.mode = _cycle_choice(st.mode, MODE_CHOICES, delta)
                        _toast_throttle(st, f"Mode: {st.mode.upper()}")
                        _emit_state(st)

                    elif st.cursor == "play":
                        if st.playing:
                            st.playing = False
                            _emit_state(st)
                            stop_real()
                            _toast_throttle(st, "STOP")
                        else:
                            st.playing = True
                            st._last_step_t = 0.0
                            _emit_state(st)
                            start_real()

                            if not cap.t or not cap.t.is_alive():
                                st.playing = False
                                _emit_state(st)
                                _fatal("ALSA capture failed to start (arecord)")
                            else:
                                _toast_throttle(st, "PLAY")

                    # Optional quick snapshot shortcut:
                    if st.cursor == "play" and cmd == "select_hold" and not st.playing:
                        try:
                            pcm = cap.ring.snapshot_last_seconds(2.0)
                            if pcm:
                                _write_wav(SNAP_WAV, pcm)
                                _toast_throttle(st, "Saved 2s snapshot WAV")
                            else:
                                _toast_throttle(st, "No audio in buffer")
                        except Exception as e:
                            _log_err(f"snapshot_failed: {e!r}")
                            _toast_throttle(st, "Snapshot failed")

            # ---------- sweep step timing (non-blocking) ----------
            if st.playing:
                st.level = int(cap.level)
                st.alsa_ok = bool(cap.alsa_ok)

                if (now - st._last_step_t) >= (st.sweep_ms / 1000.0):
                    st._last_step_t = now
                    st.freq_mhz = _step_freq(st.freq_mhz, st.direction)
                    try:
                        tuner.set_freq_mhz(st.freq_mhz)
                    except Exception as e:
                        _log_err(f"tune_exc: {e!r}")

                if cap.t and (not cap.t.is_alive()) and not st._fatal_active:
                    st._fatal_active = True
                    st.ready = False
                    st.playing = False
                    _emit_state(st)
                    _fatal("ALSA capture crashed repeatedly")
            else:
                st.level = int(cap.level)
                st.alsa_ok = bool(cap.alsa_ok)

            # ---------- heartbeat (<=250ms) ----------
            if (now - last_hb) >= HEARTBEAT_S:
                _emit_state(st)
                last_hb = now

            # ---------- tick pacing ----------
            elapsed = now - last_tick
            last_tick = now
            if elapsed < TICK_S:
                time.sleep(TICK_S - elapsed)

    finally:
        try:
            reader.close()
        except Exception:
            pass
        try:
            stop_real()
        except Exception:
            pass
        _emit({"type": "exit"})

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
