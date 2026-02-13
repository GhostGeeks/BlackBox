#!/usr/bin/env python3
import os
import sys
import time
import json
import wave
import shutil
import signal
import threading
import subprocess
import selectors
import errno
from pathlib import Path
from typing import Optional

# -----------------------------
# Cache paths
# -----------------------------
HERE = Path(__file__).resolve().parent
CACHE_DIR = HERE / ".cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

OUT_WAV = CACHE_DIR / "uap3_signature.wav"
OUT_TMP = CACHE_DIR / "uap3_signature.wav.tmp"
META_JSON = CACHE_DIR / "uap3_signature.meta.json"

# -----------------------------
# Audio config (Pi Zero 2W safe)
# -----------------------------
SAMPLE_RATE = 44100
CHANNELS = 1
SAMPWIDTH_BYTES = 2

DEFAULT_DURATION_S = 180
DURATION_S = int(os.environ.get("UAP_DURATION_S", str(DEFAULT_DURATION_S)))

SCHUMANN_FREQ = 7.83
CARRIER_FREQ = 100.0
HARMONIC_BASE_FREQ = 528.0
AMBIENT_BASE_FREQ = 432.0

AMP_SCHUMANN = 0.15
AMP_HARMONIC = 0.15
AMP_PING = 0.10
AMP_CHIRP = 0.10
AMP_AMBIENT = 0.20
AMP_BREATH = 0.15

BREATH_SEED = 1337
CHUNK_SECONDS = float(os.environ.get("UAP_CHUNK_S", "0.75"))

SIGNATURE_VERSION = "uap3_signature_v7_nonblocking_stdin"

running = True
stop_now = threading.Event()

play_proc: Optional[subprocess.Popen] = None
started_at: Optional[float] = None

_build_thread: Optional[threading.Thread] = None
_build_cancel = threading.Event()
_build_lock = threading.Lock()
_building = False
_last_build_progress_emit = 0.0


# -----------------------------
# JSON-only output
# -----------------------------
def emit(obj: dict) -> None:
    try:
        sys.stdout.write(json.dumps(obj, separators=(",", ":")) + "\n")
        sys.stdout.flush()
    except Exception:
        pass


# -----------------------------
# Playback (no BT management)
# -----------------------------
def _pick_player():
    if shutil.which("aplay"):
        return ["aplay", "-q"]
    if shutil.which("paplay"):
        return ["paplay"]
    raise RuntimeError("No audio player found (need aplay or paplay).")


def is_playing() -> bool:
    return play_proc is not None and play_proc.poll() is None


def playback_elapsed() -> int:
    if started_at is None:
        return 0
    return int(time.time() - started_at)


def stop_playback() -> None:
    global play_proc, started_at
    if play_proc and play_proc.poll() is None:
        try:
            play_proc.terminate()
            try:
                play_proc.wait(timeout=1.0)
            except Exception:
                play_proc.kill()
        except Exception:
            pass
    play_proc = None
    started_at = None


def start_playback_loop(path: Path) -> None:
    global play_proc, started_at
    stop_playback()
    player = _pick_player()

    if "aplay" in player[0]:
        play_proc = subprocess.Popen(
            player + ["--loop=0", str(path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    else:
        play_proc = subprocess.Popen(
            player + [str(path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    started_at = time.time()


# -----------------------------
# Cache/meta
# -----------------------------
def signature_is_ready() -> bool:
    if not OUT_WAV.exists() or not META_JSON.exists():
        return False
    try:
        meta = json.loads(META_JSON.read_text())
        return (
            meta.get("version") == SIGNATURE_VERSION
            and int(meta.get("sample_rate", -1)) == SAMPLE_RATE
            and int(meta.get("duration_s", -1)) == DURATION_S
            and int(meta.get("channels", -1)) == CHANNELS
        )
    except Exception:
        return False


def write_meta() -> None:
    META_JSON.write_text(
        json.dumps(
            {
                "version": SIGNATURE_VERSION,
                "sample_rate": SAMPLE_RATE,
                "duration_s": DURATION_S,
                "channels": CHANNELS,
                "created_at": int(time.time()),
            },
            indent=2,
        )
    )


def _safe_unlink(p: Path) -> None:
    try:
        if p.exists():
            p.unlink()
    except Exception:
        pass


# -----------------------------
# Build (threaded + cancelable)
# -----------------------------
def _throttled_build_emit(pct: float, step: str, start_time: float, force: bool = False) -> None:
    global _last_build_progress_emit
    now = time.time()
    if (not force) and (now - _last_build_progress_emit) < 0.10:
        return
    _last_build_progress_emit = now
    emit(
        {
            "type": "build",
            "pct": max(0.0, min(1.0, float(pct))),
            "step": str(step),
            "elapsed_s": int(now - start_time),
        }
    )


def build_uap_signature(cancel_evt: threading.Event) -> None:
    try:
        import numpy as np
    except Exception as e:
        raise RuntimeError("numpy missing (pip install numpy)") from e

    try:
        os.nice(5)
    except Exception:
        pass

    total_samples = int(DURATION_S * SAMPLE_RATE)
    chunk_size = max(512, int(CHUNK_SECONDS * SAMPLE_RATE))
    chunks = total_samples // chunk_size
    remainder = total_samples % chunk_size

    rng = np.random.default_rng(BREATH_SEED)
    klen = 64

    def moving_average_same(x_f32, win: int):
        if win <= 1:
            return x_f32
        pad_left = win // 2
        pad_right = win - 1 - pad_left
        xpad = np.pad(x_f32, (pad_left, pad_right), mode="edge").astype(np.float64, copy=False)
        c = np.cumsum(np.concatenate(([0.0], xpad)), dtype=np.float64)
        y = (c[win:] - c[:-win]) / float(win)
        return y.astype(np.float32, copy=False)

    steps = [
        "Allocating spectrum",
        "Installing harmonics",
        "Tuning resonance",
        "Calibrating pings",
        "Shaping chirps",
        "Breathing envelope",
        "Final mixdown",
        "Normalizing output",
    ]

    _safe_unlink(OUT_TMP)

    start_time = time.time()
    _throttled_build_emit(0.0, "Starting build", start_time, force=True)

    with wave.open(str(OUT_TMP), "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(SAMPWIDTH_BYTES)
        wf.setframerate(SAMPLE_RATE)

        total_parts = chunks + (1 if remainder else 0)

        for c in range(total_parts):
            if cancel_evt.is_set() or stop_now.is_set():
                raise RuntimeError("Build cancelled")

            cur_size = remainder if (c == chunks and remainder) else chunk_size
            chunk_start = c * chunk_size
            t0 = chunk_start / SAMPLE_RATE

            step = steps[min(len(steps) - 1, (c * len(steps)) // max(1, total_parts))]
            _throttled_build_emit(chunk_start / float(total_samples), step, start_time)

            t = (np.arange(cur_size, dtype=np.float64) / SAMPLE_RATE) + t0

            carrier = np.sin(2 * np.pi * CARRIER_FREQ * t)
            modulator = 0.5 * (1.0 + np.sin(2 * np.pi * SCHUMANN_FREQ * t))
            layer1 = (modulator * carrier * AMP_SCHUMANN).astype(np.float64, copy=False)

            sig = np.sin(2 * np.pi * HARMONIC_BASE_FREQ * t)
            sig += 0.3 * np.sin(2 * np.pi * (HARMONIC_BASE_FREQ * 2.0) * t)
            sig += 0.1 * np.sin(2 * np.pi * (HARMONIC_BASE_FREQ * 3.0) * t)
            wobble = 1.0 + 0.001 * np.sin(2 * np.pi * 0.1 * t)
            layer2 = (sig * wobble * AMP_HARMONIC).astype(np.float64, copy=False)

            ping_freq = 17000.0
            ping_dur = 0.1
            cycle5 = np.mod(t, 5.0)
            ping_mask = cycle5 < ping_dur
            ping_env = np.zeros_like(t, dtype=np.float64)
            if np.any(ping_mask):
                ping_env[ping_mask] = np.sin(np.pi * (cycle5[ping_mask] / ping_dur)) ** 2
            layer3 = (np.sin(2 * np.pi * ping_freq * t) * ping_env * AMP_PING).astype(np.float64, copy=False)

            chirp_dur = 0.2
            cycle10 = np.mod(t, 10.0)
            chirp_mask = cycle10 < chirp_dur
            chirp = np.zeros_like(t, dtype=np.float64)
            if np.any(chirp_mask):
                f0 = 2000.0
                f1 = 3000.0
                tr = cycle10[chirp_mask]
                k = (f1 - f0) / chirp_dur
                phase = 2 * np.pi * (f0 * tr + 0.5 * k * tr * tr)
                env = np.sin(np.pi * (tr / chirp_dur)) ** 2
                chirp[chirp_mask] = np.sin(phase) * env
            layer4 = (chirp * AMP_CHIRP).astype(np.float64, copy=False)

            pad = np.sin(2 * np.pi * AMBIENT_BASE_FREQ * t)
            pad += 0.5 * np.sin(2 * np.pi * (AMBIENT_BASE_FREQ * 1.5) * t + 0.3)
            pad += 0.25 * np.sin(2 * np.pi * (AMBIENT_BASE_FREQ * 2.0) * t + 0.7)
            pad += 0.125 * np.sin(2 * np.pi * (AMBIENT_BASE_FREQ * 2.5) * t + 1.1)
            mod = 0.8 + 0.2 * np.sin(2 * np.pi * 0.1 * t)
            layer5 = (pad * mod * AMP_AMBIENT).astype(np.float64, copy=False)

            noise = rng.normal(0.0, 1.0, size=cur_size).astype(np.float32, copy=False)
            filtered = moving_average_same(noise, klen)
            cycleB = np.mod(t, 5.0)
            envB = np.zeros_like(t, dtype=np.float64)
            inhale = cycleB < 2.0
            envB[inhale] = (np.sin(np.pi * cycleB[inhale] / 4.0) ** 2)
            envB[~inhale] = (np.cos(np.pi * (cycleB[~inhale] - 2.0) / 6.0) ** 2)
            layer6 = (filtered.astype(np.float64, copy=False) * envB * AMP_BREATH).astype(np.float64, copy=False)

            mixed = layer1 + layer2 + layer3 + layer4 + layer5 + layer6

            max_amp = float(np.max(np.abs(mixed))) if mixed.size else 0.0
            if max_amp > 0.95:
                mixed = mixed * (0.95 / max_amp)

            pcm = (mixed * 32767.0).astype(np.int16, copy=False)
            wf.writeframes(pcm.tobytes())

            done = chunk_start + cur_size
            _throttled_build_emit(done / float(total_samples), step, start_time)

            if (c % 2) == 0:
                time.sleep(0.001)

    OUT_TMP.replace(OUT_WAV)
    write_meta()
    _throttled_build_emit(1.0, "Build complete", start_time, force=True)


def send_state() -> None:
    emit(
        {
            "type": "state",
            "ready": signature_is_ready(),
            "playing": is_playing(),
            "elapsed_s": playback_elapsed(),
            "duration_s": DURATION_S,
            "building": _building,
        }
    )


def _set_building(val: bool) -> None:
    global _building
    with _build_lock:
        _building = val


def start_build_async(force_rebuild: bool) -> None:
    global _build_thread

    with _build_lock:
        if _build_thread is not None and _build_thread.is_alive():
            return

        _build_cancel.clear()
        _set_building(True)

        if force_rebuild:
            stop_playback()
            _safe_unlink(OUT_WAV)
            _safe_unlink(META_JSON)
            _safe_unlink(OUT_TMP)

        emit({"type": "page", "name": "build"})

        def _runner():
            try:
                build_uap_signature(_build_cancel)
            except Exception as e:
                # If we were cancelled as part of shutdown, don't scream fatal
                if not (stop_now.is_set() or _build_cancel.is_set()):
                    emit({"type": "fatal", "message": f"Build failed: {e}"})
            finally:
                _set_building(False)
                if signature_is_ready() and not (_build_cancel.is_set() or stop_now.is_set()):
                    emit({"type": "page", "name": "playback"})
                    send_state()

        _build_thread = threading.Thread(target=_runner, name="uap_build", daemon=True)
        _build_thread.start()


# -----------------------------
# Command handling
# -----------------------------
def handle_cmd(cmd: str) -> None:
    cmd = (cmd or "").strip().lower()
    if not cmd:
        return

    if cmd == "select":
        if _building:
            send_state()
            return

        if is_playing():
            stop_playback()
        else:
            if not signature_is_ready():
                emit({"type": "fatal", "message": "Signature not ready"})
                return
            start_playback_loop(OUT_WAV)

        send_state()
        return

    if cmd == "select_hold":
        start_build_async(force_rebuild=True)
        return

    if cmd == "back":
        _build_cancel.set()
        stop_playback()
        emit({"type": "exit"})
        stop_now.set()
        return

    # up/down ignored
    return


# -----------------------------
# Signals (must stop promptly)
# -----------------------------
def _sig_handler(signum, frame):
    global running
    running = False
    stop_now.set()
    _build_cancel.set()
    stop_playback()
    # Do NOT block here. Loop will exit quickly.
    # Also emit exit so parent can clean up.
    emit({"type": "exit"})


signal.signal(signal.SIGINT, _sig_handler)
signal.signal(signal.SIGTERM, _sig_handler)


# -----------------------------
# Non-blocking stdin loop
# -----------------------------
def _setup_nonblocking_stdin(sel: selectors.BaseSelector):
    fd = sys.stdin.fileno()
    os.set_blocking(fd, False)
    sel.register(fd, selectors.EVENT_READ)
    return fd


def main() -> int:
    emit({"type": "hello", "module": "uap_caller", "version": SIGNATURE_VERSION})

    try:
        _pick_player()
    except Exception as e:
        emit({"type": "fatal", "message": str(e)})
        emit({"type": "exit"})
        return 2

    if not signature_is_ready():
        start_build_async(force_rebuild=False)
    else:
        emit({"type": "page", "name": "playback"})
        send_state()

    sel = selectors.DefaultSelector()
    fd = _setup_nonblocking_stdin(sel)
    buf = bytearray()

    last_state_emit = 0.0

    while running and not stop_now.is_set():
        # Periodic state updates so launcher UI stays fresh during playback
        now = time.time()
        if (now - last_state_emit) >= 0.25:
            send_state()
            last_state_emit = now

        events = sel.select(timeout=0.1)
        if not events:
            continue

        for key, _mask in events:
            if key.fd != fd:
                continue
            try:
                chunk = os.read(fd, 4096)
            except BlockingIOError:
                continue
            except OSError as e:
                if e.errno in (errno.EAGAIN, errno.EWOULDBLOCK):
                    continue
                stop_now.set()
                break

            if not chunk:
                # stdin closed -> exit
                stop_now.set()
                break

            buf.extend(chunk)
            while b"\n" in buf:
                line, _, rest = buf.partition(b"\n")
                buf = bytearray(rest)
                try:
                    handle_cmd(line.decode("utf-8", errors="ignore"))
                except Exception as e:
                    emit({"type": "fatal", "message": f"Runtime error: {e}"})
                    stop_now.set()
                    break

    # shutdown
    stop_now.set()
    _build_cancel.set()
    stop_playback()
    emit({"type": "exit"})
    return 0


if __name__ == "__main__":
    sys.exit(main())
