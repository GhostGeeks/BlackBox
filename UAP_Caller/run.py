#!/usr/bin/env python3
import os
import sys
import time
import math
import wave
import random
import signal
import subprocess
from pathlib import Path
from typing import Optional

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
# Audio config / file locations
# -----------------------------
HOME = Path("/home/ghostgeeks01")
MOD_DIR = HOME / "oled" / "modules" / "uap_caller"
OUT_WAV = MOD_DIR / "uap3_output.wav"

SAMPLE_RATE = 44100
CHANNELS = 1

# Output modes
MODE_LOCAL = "LOCAL"
MODE_BT = "BT"
MODE_AUTO = "AUTO"  # plays regardless; assumes system routes audio (BT if connected, else local)

running = True

def oled_message(title: str, lines, footer: str = ""):
    with canvas(device) as draw:
        draw.text((0, 0), title[:21], fill=255)
        draw.line((0, 12, 127, 12), fill=255)
        y = 16
        for ln in (lines or [])[:3]:
            draw.text((0, y), str(ln)[:21], fill=255)
            y += 12
        if footer:
            draw.text((0, 54), footer[:21], fill=255)

def bt_connected() -> bool:
    """
    More reliable than 'bluetoothctl info' with no device.
    Returns True if any connected device is listed.
    """
    try:
        r = subprocess.run(
            "bluetoothctl devices Connected | grep -q .",
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return r.returncode == 0
    except Exception:
        return False

def generate_uap_wav(path: Path, duration_s: int = 60):
    """
    Lightweight generator (no numpy/scipy). Creates a UAP-style layered signal.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    total_frames = duration_s * SAMPLE_RATE

    def clamp16(x: float) -> int:
        x = max(-1.0, min(1.0, x))
        return int(x * 32767)

    # Layer params (inspired by your original generator description)
    schumann = 7.83
    carrier = 100.0
    harmonic = 528.0
    ambient = 432.0
    ping_freq = 17000.0
    chirp_f0 = 2000.0
    chirp_f1 = 3000.0

    # amplitudes
    A_sch = 0.14
    A_har = 0.14
    A_amb = 0.20
    A_ping = 0.08
    A_chirp = 0.08
    A_breath = 0.14

    # helper for chirp
    def chirp(t_rel: float, dur: float) -> float:
        # linear sweep instantaneous phase approximation
        k = (chirp_f1 - chirp_f0) / dur
        phase = 2 * math.pi * (chirp_f0 * t_rel + 0.5 * k * t_rel * t_rel)
        return math.sin(phase)

    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(SAMPLE_RATE)

        # write in chunks
        chunk = 1024
        n = 0
        while n < total_frames:
            frames = min(chunk, total_frames - n)
            buf = bytearray()

            for i in range(frames):
                t = (n + i) / SAMPLE_RATE
                # 1) Schumann AM over 100 Hz carrier
                mod = 0.5 * (1.0 + math.sin(2 * math.pi * schumann * t))
                sch = math.sin(2 * math.pi * carrier * t) * mod * A_sch

                # 2) 528 Hz + harmonics (slight wobble)
                wobble = 1.0 + 0.001 * math.sin(2 * math.pi * 0.1 * t)
                har = (math.sin(2 * math.pi * harmonic * t) +
                       0.3 * math.sin(2 * math.pi * (harmonic * 2) * t) +
                       0.1 * math.sin(2 * math.pi * (harmonic * 3) * t)) * wobble * A_har

                # 3) Ambient pad
                amb = (math.sin(2 * math.pi * ambient * t) +
                       0.5 * math.sin(2 * math.pi * (ambient * 1.5) * t + 0.3) +
                       0.25 * math.sin(2 * math.pi * (ambient * 2.0) * t + 0.7)) * (0.8 + 0.2 * math.sin(2 * math.pi * 0.1 * t)) * A_amb

                # 4) Pings every 5 seconds, 100ms
                ping = 0.0
                cycle5 = t % 5.0
                if cycle5 < 0.10:
                    env = math.sin(math.pi * (cycle5 / 0.10)) ** 2
                    ping = math.sin(2 * math.pi * ping_freq * t) * env * A_ping

                # 5) Chirps every 10 seconds, 200ms
                chirp_sig = 0.0
                cycle10 = t % 10.0
                if cycle10 < 0.20:
                    env = math.sin(math.pi * (cycle10 / 0.20)) ** 2
                    chirp_sig = chirp(cycle10, 0.20) * env * A_chirp

                # 6) “Breathing” noise (simple shaped noise)
                breath_cycle = 5.0
                pos = t % breath_cycle
                if pos < 2.0:
                    env = math.sin(math.pi * pos / 4.0) ** 2
                else:
                    env = math.cos(math.pi * (pos - 2.0) / 6.0) ** 2
                noise = (random.random() * 2.0 - 1.0) * env * A_breath * 0.6

                y = sch + har + amb + ping + chirp_sig + noise
                s = clamp16(y)
                buf += int(s).to_bytes(2, byteorder="little", signed=True)

            wf.writeframes(buf)
            n += frames

def start_playback(path: Path) -> Optional[subprocess.Popen]:
    """
    Plays WAV via pw-play. We use signals for pause/resume (SIGSTOP/SIGCONT).
    """
    try:
        return subprocess.Popen(
            ["pw-play", "-q", str(path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return None

def stop_process(p: Optional[subprocess.Popen]):
    if not p:
        return
    try:
        if p.poll() is None:
            p.terminate()
            time.sleep(0.3)
        if p.poll() is None:
            p.kill()
    except Exception:
        pass

def pause_process(p: Optional[subprocess.Popen]):
    if not p or p.poll() is not None:
        return
    try:
        p.send_signal(signal.SIGSTOP)
    except Exception:
        pass

def resume_process(p: Optional[subprocess.Popen]):
    if not p or p.poll() is not None:
        return
    try:
        p.send_signal(signal.SIGCONT)
    except Exception:
        pass

def handle_sigterm(sig, frame):
    global running
    running = False

signal.signal(signal.SIGTERM, handle_sigterm)
signal.signal(signal.SIGINT, handle_sigterm)

def main():
    """
    Controls:
      - Receives commands from stdin (sent by app.py): up, down, select, select_hold, back
      - select toggles play/pause
      - up/down changes output mode
      - back exits cleanly
    """
    mode_idx = 0
    modes = [MODE_AUTO, MODE_LOCAL, MODE_BT]

    playing = False
    paused = False
    proc = None

    MOD_DIR.mkdir(parents=True, exist_ok=True)

    oled_message("UAP CALLER", ["Loading...", "", ""], "BACK = exit")
    time.sleep(0.4)

    # Make/generate file if missing
    if not OUT_WAV.exists():
        oled_message("UAP CALLER", ["Generating audio", "Please wait...", ""], "BACK = exit")
        generate_uap_wav(OUT_WAV, duration_s=60)

    oled_message("UAP CALLER", [f"Mode: {modes[mode_idx]}", "SEL = play/pause", ""], "UP/DN mode")

    # stdin command loop
    global running
    while running:
        # Read a line if available (non-blocking-ish)
        line = sys.stdin.readline()
        if not line:
            time.sleep(0.05)
            continue

        cmd = line.strip().lower()

        if cmd == "up":
            mode_idx = (mode_idx - 1) % len(modes)
        elif cmd == "down":
            mode_idx = (mode_idx + 1) % len(modes)
        elif cmd == "select":
            # Toggle play/pause
            if not playing:
                # If BT mode requested but no BT connected, show warning (still can try)
                if modes[mode_idx] == MODE_BT and not bt_connected():
                    oled_message("UAP CALLER", ["No BT connected", "Playing anyway...", ""], "BACK = exit")
                    time.sleep(0.8)

                proc = start_playback(OUT_WAV)
                playing = proc is not None
                paused = False
            else:
                # playing -> pause/resume
                if not paused:
                    pause_process(proc)
                    paused = True
                else:
                    resume_process(proc)
                    paused = False

        elif cmd == "select_hold":
            # Optional: regenerate a new file quickly
            oled_message("UAP CALLER", ["Regenerating", "audio...", ""], "BACK = exit")
            stop_process(proc)
            proc = None
            playing = False
            paused = False
            generate_uap_wav(OUT_WAV, duration_s=60)

        elif cmd == "back":
            break

        # Update UI
        status = "STOPPED"
        if playing and proc and proc.poll() is None:
            status = "PAUSED" if paused else "PLAYING"
        elif playing:
            # playback ended
            playing = False
            paused = False
            proc = None

        extra = ""
        if modes[mode_idx] == MODE_AUTO:
            extra = "BT" if bt_connected() else "LOCAL"
        elif modes[mode_idx] == MODE_BT:
            extra = "BT OK" if bt_connected() else "NO BT"

        oled_message(
            "UAP CALLER",
            [f"Mode: {modes[mode_idx]} {extra}", f"State: {status}", "SEL toggle"],
            "BACK exit",
        )

    # Cleanup
    stop_process(proc)
    oled_message("UAP CALLER", ["Exiting...", "", ""], "")
    time.sleep(0.3)

if __name__ == "__main__":
    main()
