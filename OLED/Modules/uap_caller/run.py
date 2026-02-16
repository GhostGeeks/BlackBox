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

HERE = Path(__file__).resolve().parent
CACHE_DIR = HERE / ".cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

OUT_WAV = CACHE_DIR / "uap3_signature.wav"
OUT_TMP = CACHE_DIR / "uap3_signature.wav.tmp"
META_JSON = CACHE_DIR / "uap3_signature.meta.json"

SAMPLE_RATE = 44100
CHANNELS = 1
SAMPWIDTH_BYTES = 2

DEFAULT_DURATION_S = 180
DURATION_S = int(os.environ.get("UAP_DURATION_S", str(DEFAULT_DURATION_S)))

# synthesis params
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

SIGNATURE_VERSION = "uap3_signature_v12_builder_subprocess"

stop_now = threading.Event()

play_proc: Optional[subprocess.Popen] = None
started_at: Optional[float] = None

_building_lock = threading.Lock()
_building = False
_builder_proc: Optional[subprocess.Popen] = None


# -----------------------------
# JSON-only output
# -----------------------------
def emit(obj: dict) -> None:
    try:
        sys.stdout.write(json.dumps(obj, separators=(",", ":")) + "\n")
        sys.stdout.flush()
    except Exception:
        pass


def _hard_exit(code: int = 0) -> None:
    try:
        emit({"type": "exit"})
    finally:
        os._exit(code)


# -----------------------------
# Playback
# -----------------------------
def _pick_player():
    # Prefer PipeWire/Pulse path for Bluetooth
    if shutil.which("paplay"):
        return ["paplay"]
    if shutil.which("aplay"):
        return ["aplay", "-q"]
    raise RuntimeError("No audio player found (need paplay or aplay).")


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
    """
    Never raise out of here. If playback fails, emit fatal and keep module alive.
    """
    global play_proc, started_at

    if not path.exists():
        emit({"type": "fatal", "message": f"WAV missing: {path.name}"})
        return

    stop_playback()

    try:
        player = _pick_player()

        # paplay plays once; aplay can loop with --loop=0
        if os.path.basename(player[0]) == "aplay":
            args = player + ["--loop=0", str(path)]
        else:
            args = player + [str(path)]

        play_proc = subprocess.Popen(
            args,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        started_at = time.time()

    except Exception as e:
        play_proc = None
        started_at = None
        emit({"type": "fatal", "message": f"Playback failed: {e.__class__.__name__}: {e}"})
        return


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


def _set_building(v: bool) -> None:
    global _building
    with _building_lock:
        _building = v


def _get_building() -> bool:
    with _building_lock:
        return _building


# -----------------------------
# Builder subprocess (numpy lives there)
# -----------------------------
BUILDER_CODE = r"""
import os, sys, time, json, wave
from pathlib import Path

def emit(o):
    sys.stdout.write(json.dumps(o, separators=(",",":")) + "\n")
    sys.stdout.flush()

out_tmp = Path(sys.argv[1])
sr = int(sys.argv[2])
dur_s = int(sys.argv[3])
chunk_seconds = float(sys.argv[4])

# Immediate feedback before numpy import
t0 = time.time()
emit({"type":"build","pct":0.01,"step":"Loading numpy","elapsed_s":int(time.time()-t0)})

import numpy as np

try:
    os.nice(5)
except Exception:
    pass

CHANNELS=1
SAMPWIDTH_BYTES=2

SCHUMANN_FREQ=7.83
CARRIER_FREQ=100.0
HARMONIC_BASE_FREQ=528.0
AMBIENT_BASE_FREQ=432.0

AMP_SCHUMANN=0.15
AMP_HARMONIC=0.15
AMP_PING=0.10
AMP_CHIRP=0.10
AMP_AMBIENT=0.20
AMP_BREATH=0.15

BREATH_SEED=1337

total_samples = int(dur_s * sr)
chunk_size = max(512, int(chunk_seconds * sr))
chunks = total_samples // chunk_size
remainder = total_samples % chunk_size

rng = np.random.default_rng(BREATH_SEED)
klen = 64

def moving_average_same(x_f32, win):
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

emit({"type":"build","pct":0.02,"step":"Starting build","elapsed_s":int(time.time()-t0)})

out_tmp.parent.mkdir(parents=True, exist_ok=True)
try:
    if out_tmp.exists():
        out_tmp.unlink()
except Exception:
    pass

with wave.open(str(out_tmp), "wb") as wf:
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(SAMPWIDTH_BYTES)
    wf.setframerate(sr)

    total_parts = chunks + (1 if remainder else 0)

    for c in range(total_parts):
        cur_size = remainder if (c == chunks and remainder) else chunk_size
        chunk_start = c * chunk_size
        t_start = chunk_start / sr

        step = steps[min(len(steps)-1, (c * len(steps)) // max(1, total_parts))]
        emit({"type":"build","pct":chunk_start/float(total_samples),"step":step,"elapsed_s":int(time.time()-t0)})

        t = (np.arange(cur_size, dtype=np.float64) / sr) + t_start

        carrier = np.sin(2*np.pi*CARRIER_FREQ*t)
        modulator = 0.5*(1.0 + np.sin(2*np.pi*SCHUMANN_FREQ*t))
        layer1 = (modulator*carrier*AMP_SCHUMANN).astype(np.float64, copy=False)

        sig = np.sin(2*np.pi*HARMONIC_BASE_FREQ*t)
        sig += 0.3*np.sin(2*np.pi*(HARMONIC_BASE_FREQ*2.0)*t)
        sig += 0.1*np.sin(2*np.pi*(HARMONIC_BASE_FREQ*3.0)*t)
        wobble = 1.0 + 0.001*np.sin(2*np.pi*0.1*t)
        layer2 = (sig*wobble*AMP_HARMONIC).astype(np.float64, copy=False)

        ping_freq = 17000.0
        ping_dur = 0.1
        cycle5 = np.mod(t, 5.0)
        ping_mask = cycle5 < ping_dur
        ping_env = np.zeros_like(t, dtype=np.float64)
        if np.any(ping_mask):
            ping_env[ping_mask] = (np.sin(np.pi*(cycle5[ping_mask]/ping_dur))**2)
        layer3 = (np.sin(2*np.pi*ping_freq*t)*ping_env*AMP_PING).astype(np.float64, copy=False)

        chirp_dur = 0.2
        cycle10 = np.mod(t, 10.0)
        chirp_mask = cycle10 < chirp_dur
        chirp = np.zeros_like(t, dtype=np.float64)
        if np.any(chirp_mask):
            f0 = 2000.0
            f1 = 3000.0
            tr = cycle10[chirp_mask]
            k = (f1-f0)/chirp_dur
            phase = 2*np.pi*(f0*tr + 0.5*k*tr*tr)
            env = (np.sin(np.pi*(tr/chirp_dur))**2)
            chirp[chirp_mask] = np.sin(phase)*env
        layer4 = (chirp*AMP_CHIRP).astype(np.float64, copy=False)

        pad = np.sin(2*np.pi*AMBIENT_BASE_FREQ*t)
        pad += 0.5*np.sin(2*np.pi*(AMBIENT_BASE_FREQ*1.5)*t + 0.3)
        pad += 0.25*np.sin(2*np.pi*(AMBIENT_BASE_FREQ*2.0)*t + 0.7)
        pad += 0.125*np.sin(2*np.pi*(AMBIENT_BASE_FREQ*2.5)*t + 1.1)
        mod = 0.8 + 0.2*np.sin(2*np.pi*0.1*t)
        layer5 = (pad*mod*AMP_AMBIENT).astype(np.float64, copy=False)

        noise = rng.normal(0.0, 1.0, size=cur_size).astype("float32", copy=False)
        filtered = moving_average_same(noise, klen)
        cycleB = np.mod(t, 5.0)
        envB = np.zeros_like(t, dtype=np.float64)
        inhale = cycleB < 2.0
        envB[inhale] = (np.sin(np.pi*cycleB[inhale]/4.0)**2)
        envB[~inhale] = (np.cos(np.pi*(cycleB[~inhale]-2.0)/6.0)**2)
        layer6 = (filtered.astype(np.float64, copy=False)*envB*AMP_BREATH).astype(np.float64, copy=False)

        mixed = layer1+layer2+layer3+layer4+layer5+layer6
        max_amp = float(np.max(np.abs(mixed))) if mixed.size else 0.0
        if max_amp > 0.95:
            mixed = mixed * (0.95/max_amp)

        pcm = (mixed*32767.0).astype("int16", copy=False)
        wf.writeframes(pcm.tobytes())

        done = chunk_start + cur_size
        emit({"type":"build","pct":done/float(total_samples),"step":step,"elapsed_s":int(time.time()-t0)})

        if (c % 2) == 0:
            time.sleep(0.001)

emit({"type":"build","pct":1.0,"step":"Build complete","elapsed_s":int(time.time()-t0)})
"""

def _spawn_builder() -> subprocess.Popen:
    # Spawn with -u so builder progress is flushed immediately
    cmd = [
        sys.executable, "-u", "-c", BUILDER_CODE,
        str(OUT_TMP), str(SAMPLE_RATE), str(DURATION_S), str(CHUNK_SECONDS)
    ]
    return subprocess.Popen(
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        bufsize=1,
    )


def _stop_builder() -> None:
    global _builder_proc
    if _builder_proc and _builder_proc.poll() is None:
        try:
            _builder_proc.terminate()
            try:
                _builder_proc.wait(timeout=1.0)
            except Exception:
                _builder_proc.kill()
        except Exception:
            pass
    _builder_proc = None


def start_build(force_rebuild: bool) -> None:
    global _builder_proc
    if _get_building():
        return

    _set_building(True)
    stop_playback()

    if force_rebuild:
        _safe_unlink(OUT_WAV)
        _safe_unlink(META_JSON)
        _safe_unlink(OUT_TMP)

    emit({"type": "page", "name": "build"})
    emit({"type": "build", "pct": 0.0, "step": "Starting builder", "elapsed_s": 0})

    _builder_proc = _spawn_builder()


# -----------------------------
# Heartbeat (keeps launcher watchdog happy)
# -----------------------------
def _heartbeat():
    while not stop_now.is_set():
        emit({
            "type": "state",
            "ready": signature_is_ready(),
            "playing": is_playing(),
            "elapsed_s": playback_elapsed(),
            "duration_s": DURATION_S,
            "building": _get_building(),
        })
        time.sleep(0.25)


# -----------------------------
# Commands
# -----------------------------
def handle_cmd(cmd: str) -> None:
    cmd = (cmd or "").strip().lower()
    if not cmd:
        return

    if cmd == "select":
        if _get_building():
            return
        if is_playing():
            stop_playback()
        else:
            if not signature_is_ready():
                emit({"type": "fatal", "message": "Signature not ready"})
                return
            start_playback_loop(OUT_WAV)
        return

    if cmd == "select_hold":
        start_build(force_rebuild=True)
        return

    if cmd == "back":
        stop_now.set()
        _stop_builder()
        stop_playback()
        _hard_exit(0)


def _sig_handler(_signum, _frame):
    stop_now.set()
    _stop_builder()
    stop_playback()
    _hard_exit(0)


signal.signal(signal.SIGINT, _sig_handler)
signal.signal(signal.SIGTERM, _sig_handler)


# -----------------------------
# Main loop: pump stdin + builder stdout nonblocking
# -----------------------------
def main() -> int:
    emit({"type": "hello", "module": "uap_caller", "version": SIGNATURE_VERSION})

    threading.Thread(target=_heartbeat, name="uap_heartbeat", daemon=True).start()

    # Player preflight early (fatal if missing)
    try:
        _pick_player()
    except Exception as e:
        emit({"type": "fatal", "message": str(e)})
        _hard_exit(2)

    if signature_is_ready():
        emit({"type": "page", "name": "playback"})
    else:
        start_build(force_rebuild=False)

    sel = selectors.DefaultSelector()

    # stdin nonblocking
    stdin_fd = sys.stdin.fileno()
    os.set_blocking(stdin_fd, False)
    sel.register(stdin_fd, selectors.EVENT_READ, data="stdin")
    stdin_buf = bytearray()

    while not stop_now.is_set():
        # also watch builder stdout if running
        if _builder_proc and _builder_proc.stdout:
            try:
                sel.register(_builder_proc.stdout, selectors.EVENT_READ, data="builder")
            except KeyError:
                pass  # already registered

        events = sel.select(timeout=0.1)

        # handle builder completion
        if _builder_proc and _builder_proc.poll() is not None:
            rc = _builder_proc.returncode
            _stop_builder()
            if rc == 0 and OUT_TMP.exists():
                try:
                    OUT_TMP.replace(OUT_WAV)
                    write_meta()
                    _set_building(False)
                    emit({"type": "page", "name": "playback"})
                except Exception as e:
                    _set_building(False)
                    emit({"type": "fatal", "message": f"Finalize failed: {e}"})
            else:
                _set_building(False)
                emit({"type": "fatal", "message": "Build failed (builder exited non-zero)"})

        if not events:
            continue

        for key, _mask in events:
            tag = key.data

            if tag == "stdin":
                try:
                    chunk = os.read(stdin_fd, 4096)
                except BlockingIOError:
                    continue
                except OSError as e:
                    if e.errno in (errno.EAGAIN, errno.EWOULDBLOCK):
                        continue
                    _hard_exit(0)

                if not chunk:
                    _hard_exit(0)

                stdin_buf.extend(chunk)
                while b"\n" in stdin_buf:
                    line, _, rest = stdin_buf.partition(b"\n")
                    stdin_buf = bytearray(rest)
                    handle_cmd(line.decode("utf-8", errors="ignore"))

            elif tag == "builder":
                # read lines without blocking
                try:
                    line = key.fileobj.readline()
                except Exception:
                    continue
                if not line:
                    continue
                line = line.strip()
                if not line:
                    continue
                # builder guarantees JSON; but be defensive
                try:
                    msg = json.loads(line)
                except Exception:
                    continue
                # forward build messages
                if msg.get("type") == "build":
                    emit(msg)

    _hard_exit(0)


if __name__ == "__main__":
    main()
