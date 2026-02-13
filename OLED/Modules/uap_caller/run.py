#!/usr/bin/env python3
import os
import sys
import time
import json
import wave
import shutil
import signal
import subprocess
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

DEFAULT_DURATION_S = 180  # looped signature
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
CHUNK_SECONDS = 1

SIGNATURE_VERSION = "uap3_signature_v4_headless"

running = True
play_proc: Optional[subprocess.Popen] = None
started_at: Optional[float] = None


def emit(obj: dict) -> None:
    """Send status to app.py (one JSON object per line)."""
    try:
        sys.stdout.write(json.dumps(obj) + "\n")
        sys.stdout.flush()
    except Exception:
        pass


# -----------------------------
# Playback (no BT management here)
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

    if player and "aplay" in player[0]:
        play_proc = subprocess.Popen(
            player + ["--loop=0", str(path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    else:
        # paplay doesn't loop; app can re-trigger, but we keep it simple for now
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


# -----------------------------
# Build crafted signature
# -----------------------------
def build_uap3_signature() -> None:
    try:
        import numpy as np
    except Exception as e:
        raise RuntimeError("numpy missing (install python3-numpy)") from e

    total_samples = int(DURATION_S * SAMPLE_RATE)
    chunk_size = int(CHUNK_SECONDS * SAMPLE_RATE)
    chunks = total_samples // chunk_size
    remainder = total_samples % chunk_size

    rng = np.random.default_rng(BREATH_SEED)

    # FAST smoothing for breathing noise (O(n))
    klen = 64

    def moving_average_same(x, win):
        """Moving average returning same-length output (len(y) == len(x))."""
        if win <= 1:
            return x.astype(np.float32, copy=False)

        pad_left = win // 2
        pad_right = win - 1 - pad_left

        # pad to keep output == input length
        xpad = np.pad(x, (pad_left, pad_right), mode="edge").astype(np.float64, copy=False)

        # prepend 0 so the windowed diff yields len(x) samples
        c = np.cumsum(np.concatenate(([0.0], xpad)), dtype=np.float64)

        y = (c[win:] - c[:-win]) / float(win)  # length == len(x)
        return y.astype(np.float32, copy=False)

    steps = [
        "Installing harmonics",
        "Tuning resonance",
        "Calibrating pings",
        "Shaping chirps",
        "Breathing envelope",
        "Final mixdown",
        "Normalizing output",
    ]

    start_time = time.time()

    with wave.open(str(OUT_TMP), "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(SAMPWIDTH_BYTES)
        wf.setframerate(SAMPLE_RATE)

        total_parts = chunks + (1 if remainder else 0)

        for c in range(total_parts):
            cur_size = remainder if (c == chunks and remainder) else chunk_size
            chunk_start = c * chunk_size
            t0 = chunk_start / SAMPLE_RATE
            t = (np.arange(cur_size, dtype=np.float64) / SAMPLE_RATE) + t0

            # Layer 1: Schumann AM on 100Hz carrier
            carrier = np.sin(2 * np.pi * CARRIER_FREQ * t)
            modulator = 0.5 * (1.0 + np.sin(2 * np.pi * SCHUMANN_FREQ * t))
            layer1 = modulator * carrier * AMP_SCHUMANN

            # Layer 2: 528Hz + harmonics
            sig = np.sin(2 * np.pi * HARMONIC_BASE_FREQ * t)
            sig += 0.3 * np.sin(2 * np.pi * (HARMONIC_BASE_FREQ * 2.0) * t)
            sig += 0.1 * np.sin(2 * np.pi * (HARMONIC_BASE_FREQ * 3.0) * t)
            wobble = 1.0 + 0.001 * np.sin(2 * np.pi * 0.1 * t)
            layer2 = sig * wobble * AMP_HARMONIC

            # Layer 3: 17kHz pings every 5s
            ping_freq = 17000.0
            ping_dur = 0.1
            cycle5 = np.mod(t, 5.0)
            ping_mask = cycle5 < ping_dur
            ping_env = np.zeros_like(t, dtype=np.float64)
            ping_env[ping_mask] = np.sin(np.pi * (cycle5[ping_mask] / ping_dur)) ** 2
            layer3 = np.sin(2 * np.pi * ping_freq * t) * ping_env * AMP_PING

            # Layer 4: chirps every 10s (2k->3k over 0.2s)
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
            layer4 = chirp * AMP_CHIRP

            # Layer 5: ambient pad at 432Hz with harmonics
            pad = np.sin(2 * np.pi * AMBIENT_BASE_FREQ * t)
            pad += 0.5 * np.sin(2 * np.pi * (AMBIENT_BASE_FREQ * 1.5) * t + 0.3)
            pad += 0.25 * np.sin(2 * np.pi * (AMBIENT_BASE_FREQ * 2.0) * t + 0.7)
            pad += 0.125 * np.sin(2 * np.pi * (AMBIENT_BASE_FREQ * 2.5) * t + 1.1)
            mod = 0.8 + 0.2 * np.sin(2 * np.pi * 0.1 * t)
            layer5 = pad * mod * AMP_AMBIENT

            # Layer 6: breathing noise (deterministic)
            noise = rng.normal(0.0, 1.0, size=cur_size).astype(np.float32)
            filtered = moving_average_same(noise, klen)
            cycleB = np.mod(t, 5.0)
            envB = np.zeros_like(t, dtype=np.float64)
            inhale = cycleB < 2.0
            envB[inhale] = np.sin(np.pi * cycleB[inhale] / 4.0) ** 2
            exhale = ~inhale
            envB[exhale] = np.cos(np.pi * (cycleB[exhale] - 2.0) / 6.0) ** 2
            layer6 = (filtered.astype(np.float64) * envB) * AMP_BREATH

            mixed = layer1 + layer2 + layer3 + layer4 + layer5 + layer6

            # Normalize if needed
            max_amp = float(np.max(np.abs(mixed))) if mixed.size else 0.0
            if max_amp > 0.95:
                mixed = mixed * (0.95 / max_amp)

            pcm = (mixed * 32767.0).astype(np.int16)
            wf.writeframes(pcm.tobytes())

            done = chunk_start + cur_size
            pct = done / float(total_samples)
            elapsed = int(time.time() - start_time)
            step = steps[(c // 2) % len(steps)]
            emit({"type": "build", "pct": pct, "step": step, "elapsed_s": elapsed})

    OUT_TMP.replace(OUT_WAV)
    write_meta()


def send_state() -> None:
    emit(
        {
            "type": "state",
            "ready": signature_is_ready(),
            "playing": is_playing(),
            "elapsed_s": playback_elapsed(),
            "duration_s": DURATION_S,
        }
    )


# -----------------------------
# Button protocol (from app.py)
# up, down, select, select_hold, back
# -----------------------------
def handle_cmd(cmd: str) -> None:
    cmd = (cmd or "").strip().lower()
    if not cmd:
        return

    if cmd == "select":
        if is_playing():
            stop_playback()
        else:
            if not signature_is_ready():
                emit({"type": "error", "message": "Signature not ready"})
            else:
                start_playback_loop(OUT_WAV)
        send_state()
        return

    if cmd == "back":
        stop_playback()
        emit({"type": "exit"})
        raise SystemExit(0)

    if cmd == "select_hold":
        stop_playback()
        try:
            if OUT_WAV.exists():
                OUT_WAV.unlink()
            if META_JSON.exists():
                META_JSON.unlink()
        except Exception:
            pass
        emit({"type": "page", "name": "build"})
        try:
            build_uap3_signature()
        except Exception as e:
            emit({"type": "fatal", "message": f"Build failed: {e}"})
            return
        emit({"type": "page", "name": "playback"})
        send_state()
        return

    if cmd in ("up", "down"):
        emit({"type": "noop", "cmd": cmd})
        return

    emit({"type": "error", "message": f"Unknown cmd: {cmd}"})


# -----------------------------
# Signals
# -----------------------------
def _sig_handler(signum, frame):
    global running
    running = False
    stop_playback()


signal.signal(signal.SIGINT, _sig_handler)
signal.signal(signal.SIGTERM, _sig_handler)


def main():
    emit({"type": "hello", "module": "uap_caller", "version": SIGNATURE_VERSION})

    # Preflight player
    try:
        _pick_player()
    except Exception as e:
        emit({"type": "fatal", "message": str(e)})
        return 2

    # Build on initial load
    if not signature_is_ready():
        emit({"type": "page", "name": "build"})
        try:
            build_uap3_signature()
        except Exception as e:
            emit({"type": "fatal", "message": f"Build failed: {e}"})
            return 3

    emit({"type": "page", "name": "playback"})
    send_state()

    while running:
        line = sys.stdin.readline()
        if line == "":
            break
        try:
            handle_cmd(line)
        except SystemExit:
            break
        except Exception as e:
            emit({"type": "error", "message": str(e)})

    stop_playback()
    emit({"type": "exit"})
    return 0


if __name__ == "__main__":
    sys.exit(main())
