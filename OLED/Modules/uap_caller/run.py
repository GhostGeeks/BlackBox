#!/usr/bin/env python3
import os
import sys
import time
import math
import wave
import random
import signal
import shutil
import subprocess
from pathlib import Path
from typing import Optional, List, Dict
import selectors

from luma.core.interface.serial import i2c
from luma.oled.device import ssd1306
from luma.core.render import canvas


# -----------------------------
# OLED config
# -----------------------------
I2C_PORT = 1
I2C_ADDR = 0x3C
OLED_W, OLED_H = 128, 64

serial = i2c(port=I2C_PORT, address=I2C_ADDR)
device = ssd1306(serial, width=OLED_W, height=OLED_H)

# -----------------------------
# Paths
# -----------------------------
HERE = Path(__file__).resolve().parent
CACHE_DIR = HERE / ".cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
OUT_WAV = CACHE_DIR / "uap_output.wav"
OUT_WAV_TMP = CACHE_DIR / "uap_output.wav.tmp"

# -----------------------------
# Audio config
# -----------------------------
SAMPLE_RATE = 44100
CHANNELS = 1
SAMPWIDTH_BYTES = 2  # 16-bit PCM

# Reliability knobs for Pi Zero 2W
WRITE_CHUNK_FRAMES = 1024  # small, steady chunk writes
DEFAULT_DURATION_S = 60     # per run (preset can override)
FADE_MS = 20               # click-free start/stop

running = True


# -----------------------------
# Presets (audio-first, timed, predictable)
# -----------------------------
# All presets are "generators" that create a single wav for reliable playback.
# Keep computations light; avoid heavy filters.
PRESETS: List[Dict] = [
    {
        "name": "SCAN SWEEP",
        "subtitle": "Slow sweep + noise",
        "duration_s": 75,
        "base_amp": 0.28,
        "noise_amp": 0.05,
        "kind": "sweep",
        "f0": 180.0,
        "f1": 980.0,
        "sweep_period_s": 9.0,   # how long for up/down
        "pulse_hz": 0.0,
    },
    {
        "name": "PULSE BEACON",
        "subtitle": "Pulsed tone bursts",
        "duration_s": 60,
        "base_amp": 0.30,
        "noise_amp": 0.03,
        "kind": "pulse",
        "tone_hz": 432.0,
        "pulse_hz": 0.8,         # pulses per second
        "duty": 0.25,            # pulse on fraction
    },
    {
        "name": "CHIRP PINGS",
        "subtitle": "Short chirps + hiss",
        "duration_s": 60,
        "base_amp": 0.26,
        "noise_amp": 0.05,
        "kind": "chirp",
        "chirp_every_s": 1.5,
        "chirp_len_s": 0.12,
        "chirp_f0": 900.0,
        "chirp_f1": 1800.0,
    },
    {
        "name": "DRONE HUM",
        "subtitle": "Thick multi-tone",
        "duration_s": 90,
        "base_amp": 0.23,
        "noise_amp": 0.02,
        "kind": "drone",
        "tones": [110.0, 220.0, 440.0],
        "wobble_hz": 0.18,       # slow amplitude wobble
    },
    {
        "name": "RANDOM HOPS",
        "subtitle": "Freq hops + noise",
        "duration_s": 75,
        "base_amp": 0.28,
        "noise_amp": 0.05,
        "kind": "hop",
        "hop_every_s": 1.0,
        "hop_band": (250.0, 1400.0),
    },
]


# -----------------------------
# OLED helpers
# -----------------------------
def oled_message(title: str, lines=None, footer: str = ""):
    lines = lines or []
    with canvas(device) as draw:
        draw.text((0, 0), title[:21], fill=255)
        draw.line((0, 12, 127, 12), fill=255)
        y = 16
        for ln in lines[:3]:
            draw.text((0, y), str(ln)[:21], fill=255)
            y += 12
        if footer:
            draw.text((0, 54), footer[:21], fill=255)


def oled_status(preset_name: str, subtitle: str, state: str, t_left: int, anim: int, level: float):
    """
    World-class OLED: always answers:
      - what mode/preset?
      - what is it doing now?
      - how long left?
      - subtle motion (heartbeat) + level meter
    """
    # level 0..1
    level = max(0.0, min(1.0, level))
    bars = int(level * 10)

    with canvas(device) as draw:
        draw.text((0, 0), preset_name[:21], fill=255)
        draw.line((0, 12, 127, 12), fill=255)
        draw.text((0, 16), subtitle[:21], fill=255)
        draw.text((0, 28), state[:21], fill=255)

        mm = t_left // 60
        ss = t_left % 60
        draw.text((0, 40), f"Time: {mm:02d}:{ss:02d}", fill=255)

        # Level meter right side
        x0, y0 = 86, 40
        draw.rectangle((x0, y0, 127, 52), outline=255, fill=0)
        # bars inside
        for i in range(bars):
            bx0 = x0 + 2 + i * 4
            draw.rectangle((bx0, y0 + 2, bx0 + 2, 50), outline=255, fill=255)

        # tiny "heartbeat" animation bottom right
        dot_x = 120 + (anim % 2)
        draw.text((dot_x, 54), "•", fill=255)


# -----------------------------
# Playback helpers (no BT management)
# -----------------------------
def _pick_player() -> List[str]:
    """
    Prefer simple, reliable players available on Pi OS.
    - aplay (ALSA) is very reliable for wav
    - paplay is fine if PulseAudio/PipeWire is present
    """
    if shutil.which("aplay"):
        # -q = quiet, -N disable ALSA resampling? (not always supported); keep minimal
        return ["aplay", "-q"]
    if shutil.which("paplay"):
        return ["paplay"]
    raise RuntimeError("No audio player found (need aplay or paplay).")


def play_wav_blocking(path: Path) -> int:
    player = _pick_player()
    try:
        p = subprocess.run(player + [str(path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return p.returncode
    except Exception:
        return 1


# -----------------------------
# Audio generation (click-free, lightweight DSP)
# -----------------------------
def _clamp16(x: float) -> int:
    if x > 1.0:
        x = 1.0
    elif x < -1.0:
        x = -1.0
    return int(x * 32767)


def _fade_gain(n: int, total: int, fade_frames: int) -> float:
    """Simple linear fade in/out for click-free edges."""
    if total <= 0:
        return 1.0
    if n < fade_frames:
        return n / max(1, fade_frames)
    if n > (total - fade_frames):
        return max(0.0, (total - n) / max(1, fade_frames))
    return 1.0


def generate_uap_wav(path: Path, preset: Dict, seed: Optional[int] = None):
    """
    Generates a single PCM wav file based on the selected preset.
    Uses an atomic write+rename to avoid partial/corrupt outputs.
    """
    duration_s = int(preset.get("duration_s", DEFAULT_DURATION_S))
    total_frames = duration_s * SAMPLE_RATE
    fade_frames = int((FADE_MS / 1000.0) * SAMPLE_RATE)

    base_amp = float(preset.get("base_amp", 0.28))
    noise_amp = float(preset.get("noise_amp", 0.05))

    if seed is None:
        seed = int(time.time() * 1000) & 0xFFFFFFFF
    rng = random.Random(seed)

    path.parent.mkdir(parents=True, exist_ok=True)

    with wave.open(str(OUT_WAV_TMP), "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(SAMPWIDTH_BYTES)
        wf.setframerate(SAMPLE_RATE)

        phase = 0.0
        twopi = 2.0 * math.pi

        # For hop preset
        hop_next = 0.0
        hop_freq = 600.0

        # For chirp preset
        chirp_next = 0.0

        # For sweep preset
        sweep_period = float(preset.get("sweep_period_s", 8.0))
        f0 = float(preset.get("f0", 220.0))
        f1 = float(preset.get("f1", 880.0))

        # For pulse preset
        pulse_hz = float(preset.get("pulse_hz", 0.0))
        duty = float(preset.get("duty", 0.25))
        tone_hz = float(preset.get("tone_hz", 432.0))

        # For drone preset
        drone_tones = preset.get("tones", [110.0, 220.0, 440.0])
        wobble_hz = float(preset.get("wobble_hz", 0.15))

        kind = preset.get("kind", "sweep")

        n = 0
        while n < total_frames:
            frames = min(WRITE_CHUNK_FRAMES, total_frames - n)
            buf = bytearray()

            for i in range(frames):
                t = (n + i) / SAMPLE_RATE
                g = _fade_gain(n + i, total_frames, fade_frames)

                # base signal
                y = 0.0

                if kind == "sweep":
                    # triangle sweep 0..1..0 over period
                    u = (t % sweep_period) / sweep_period
                    tri = 1.0 - abs(2.0 * u - 1.0)
                    freq = f0 + (f1 - f0) * tri
                    phase += (twopi * freq) / SAMPLE_RATE
                    y += math.sin(phase) * base_amp

                elif kind == "pulse":
                    # pulse envelope
                    if pulse_hz <= 0.0:
                        env = 1.0
                    else:
                        u = (t * pulse_hz) % 1.0
                        env = 1.0 if u < duty else 0.0
                    phase += (twopi * tone_hz) / SAMPLE_RATE
                    # soften edges a little (no heavy filter, just a smoothstep-ish)
                    env = env * env * (3.0 - 2.0 * env)
                    y += math.sin(phase) * base_amp * env

                elif kind == "chirp":
                    every = float(preset.get("chirp_every_s", 1.5))
                    clen = float(preset.get("chirp_len_s", 0.12))
                    cf0 = float(preset.get("chirp_f0", 900.0))
                    cf1 = float(preset.get("chirp_f1", 1800.0))

                    if t >= chirp_next:
                        chirp_next = t + every

                    dt = t - (chirp_next - every)
                    if 0.0 <= dt <= clen:
                        u = dt / max(1e-6, clen)
                        freq = cf0 + (cf1 - cf0) * u
                        phase += (twopi * freq) / SAMPLE_RATE
                        # fast fade in/out on the chirp itself
                        chirp_env = math.sin(math.pi * u)
                        y += math.sin(phase) * base_amp * chirp_env
                    else:
                        # quiet in-between (still a bit of baseline tone to avoid dead air)
                        phase += (twopi * 220.0) / SAMPLE_RATE
                        y += math.sin(phase) * (base_amp * 0.12)

                elif kind == "drone":
                    wob = (math.sin(twopi * wobble_hz * t) * 0.5 + 0.5)  # 0..1
                    wob = 0.75 + 0.25 * wob
                    for j, hz in enumerate(drone_tones):
                        y += math.sin(twopi * hz * t + (j * 0.7)) * (base_amp / max(1, len(drone_tones))) * wob

                elif kind == "hop":
                    every = float(preset.get("hop_every_s", 1.0))
                    lo, hi = preset.get("hop_band", (250.0, 1400.0))
                    if t >= hop_next:
                        hop_next = t + every
                        hop_freq = rng.uniform(float(lo), float(hi))
                    phase += (twopi * hop_freq) / SAMPLE_RATE
                    y += math.sin(phase) * base_amp

                else:
                    # fallback simple tone + noise
                    phase += (twopi * 432.0) / SAMPLE_RATE
                    y += math.sin(phase) * base_amp

                # noise floor
                y += (rng.random() * 2.0 - 1.0) * noise_amp

                # global fade to avoid clicks
                y *= g

                s = _clamp16(y)
                buf += int(s).to_bytes(2, "little", signed=True)

            wf.writeframes(buf)
            n += frames

    # Atomic replace
    OUT_WAV_TMP.replace(path)
    return duration_s


# -----------------------------
# Input handling (stdin keys)
# -----------------------------
sel = selectors.DefaultSelector()
try:
    sel.register(sys.stdin, selectors.EVENT_READ)
except Exception:
    pass


def read_key_nonblocking(timeout: float = 0.0) -> Optional[str]:
    try:
        events = sel.select(timeout)
        if not events:
            return None
        data = sys.stdin.read(1)
        if data == "":
            return None
        return data
    except Exception:
        return None


# -----------------------------
# Signals
# -----------------------------
def _sig_handler(signum, frame):
    global running
    running = False


signal.signal(signal.SIGINT, _sig_handler)
signal.signal(signal.SIGTERM, _sig_handler)


# -----------------------------
# Main UX loop
# -----------------------------
def main():
    global running

    preset_idx = 0
    state = "IDLE"
    last_anim = 0
    wav_seed = None

    # Preflight: ensure player exists early, so UX is clean
    try:
        _ = _pick_player()
    except Exception as e:
        oled_message("UAP Caller", ["Audio player missing", "Install aplay/paplay"], "Exit")
        print(f"[uap_caller] ERROR: {e}", file=sys.stderr)
        return 2

    oled_message("UAP Caller", ["Ready", "1-5 preset | Space run"], "q=quit")
    time.sleep(0.4)

    while running:
        preset = PRESETS[preset_idx]
        name = preset["name"]
        subtitle = preset.get("subtitle", "")
        key = read_key_nonblocking(0.05)

        # Simple animated idle meter
        last_anim += 1
        idle_level = (math.sin(time.time() * 3.0) * 0.5 + 0.5) * 0.35

        if state == "IDLE":
            oled_status(name, subtitle, "IDLE  (select / start)", 0, last_anim, idle_level)

        if key:
            key = key.lower()

            if key == "q":
                running = False
                break

            if key in ["\n", "\r", " "]:  # start/stop
                if state == "IDLE":
                    state = "GENERATE"
                else:
                    # If you later make playback async, stop it here.
                    state = "IDLE"

            elif key == "n":
                preset_idx = (preset_idx + 1) % len(PRESETS)

            elif key in ["1", "2", "3", "4", "5"]:
                preset_idx = int(key) - 1

        if state == "GENERATE":
            wav_seed = int(time.time() * 1000) & 0xFFFFFFFF
            oled_status(name, subtitle, "Building signal…", 0, last_anim, 0.6)

            try:
                dur = generate_uap_wav(OUT_WAV, preset, seed=wav_seed)
            except Exception as e:
                oled_message("UAP Caller", ["Generate failed", str(e)[:21]], "Back")
                time.sleep(1.2)
                state = "IDLE"
                continue

            state = "PLAY"
            t_end = time.time() + dur

        if state == "PLAY":
            # We do blocking playback, but keep OLED updated using a crude "phase meter".
            # To keep it responsive, we *could* switch to Popen + polling; blocking keeps it most reliable.
            # So: show a short countdown screen, then play.
            remaining = max(0, int(t_end - time.time()))
            oled_status(name, subtitle, "Playing…", remaining, last_anim, 0.8)
            rc = play_wav_blocking(OUT_WAV)

            if rc != 0:
                oled_message("UAP Caller", ["Playback error", "Check audio output"], "Back")
                time.sleep(1.2)

            state = "IDLE"

    oled_message("UAP Caller", ["Shutting down…"], "")
    time.sleep(0.3)
    return 0


if __name__ == "__main__":
    sys.exit(main())
