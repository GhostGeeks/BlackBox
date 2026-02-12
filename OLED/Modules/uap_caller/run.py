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
from typing import Optional, List
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
# Module-local file paths (FIXED)
# -----------------------------
HERE = Path(__file__).resolve().parent
OUT_WAV = HERE / "uap3_output.wav"

SAMPLE_RATE = 44100
CHANNELS = 1

MODE_LOCAL = "LOCAL"
MODE_BT = "BT"
MODE_AUTO = "AUTO"

running = True


# -----------------------------
# OLED helpers
# -----------------------------
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


# -----------------------------
# Signal generator
# -----------------------------
def generate_uap_wav(path: Path, duration_s: int = 60):
    path.parent.mkdir(parents=True, exist_ok=True)
    total_frames = duration_s * SAMPLE_RATE

    def clamp16(x: float) -> int:
        return int(max(-1.0, min(1.0, x)) * 32767)

    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)

        chunk = 1024
        n = 0
        while n < total_frames:
            frames = min(chunk, total_frames - n)
            buf = bytearray()

            for i in range(frames):
                t = (n + i) / SAMPLE_RATE
                y = math.sin(2 * math.pi * 432.0 * t) * 0.3
                y += (random.random() * 2.0 - 1.0) * 0.05
                s = clamp16(y)
                buf += int(s).to_bytes(2, "little", signed=True)

            wf.writeframes(buf)
            n += frames


# -----------------------------
# Playback helpers
# -----------------------------
def _pick_player() -> List[str]:
    if shutil.which("paplay"):
        return ["paplay", "--client]()
