#!/usr/bin/env python3
"""
BlackBox Tone Generator (headless JSON stdout protocol) - v5 stable

Pages:
- main
- freq_menu
- freq_edit
- special_freqs
- special_tones

STRICT:
- NO OLED access
- JSON-only stdout
- Non-blocking stdin
- Immediate audio stop via process group kill
"""

import os
import sys
import json
import time
import math
import wave
import signal
import shutil
import selectors
import subprocess
from dataclasses import dataclass
from typing import Optional, List, Tuple

MODULE_NAME = "tone_generator"
MODULE_VERSION = "tg_v5_clean"

PATTERN_WAV = "/tmp/blackbox_tone_pattern.wav"
AUDIO_ERR = "/tmp/blackbox_tone_audio.err"

RATE = 48000
CHANNELS = 1
SAMPWIDTH = 2

PATTERN_SECONDS_STD = 6.0
PATTERN_SECONDS_SWEEP = 8.0
PATTERN_SECONDS_SHEP = 10.0
PATTERN_SECONDS_MOTIF = 4.0

HEARTBEAT_S = 0.25


# ---------- JSON OUTPUT ----------
def emit(obj: dict):
    sys.stdout.write(json.dumps(obj, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def toast(msg: str):
    emit({"type": "toast", "message": msg})


def fatal(msg: str):
    emit({"type": "fatal", "message": msg})


# ---------- Audio ----------
def write_wav(samples_iter, seconds: float):
    total = int(RATE * seconds)
    tmp = PATTERN_WAV + ".tmp"

    with wave.open(tmp, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(SAMPWIDTH)
        wf.setframerate(RATE)

        block = bytearray()
        count = 0

        for s in samples_iter:
            v = int(max(-1.0, min(1.0, s)) * 32767)
            block += int(v).to_bytes(2, "little", signed=True)
            count += 1
            if len(block) >= 4096:
                wf.writeframes(block)
                block.clear()
            if count >= total:
                break

        if block:
            wf.writeframes(block)

    os.replace(tmp, PATTERN_WAV)


def gen_standard(freq, volume):
    amp = (volume / 100.0) * 0.95
    phase = 0.0
    inc = freq / RATE

    def it():
        nonlocal phase
        while True:
            phase += inc
            yield math.sin(2 * math.pi * (phase % 1.0)) * amp

    write_wav(it(), PATTERN_SECONDS_STD)


def gen_sweep(volume, direction):
    amp = (volume / 100.0) * 0.85
    f0 = 20.0
    f1 = 20000.0
    dur = PATTERN_SECONDS_SWEEP
    total = int(RATE * dur)

    log_f0 = math.log(f0)
    log_f1 = math.log(f1)

    phase = 0.0

    def freq_at(u):
        return math.exp(log_f0 + (log_f1 - log_f0) * u)

    def it():
        nonlocal phase
        for n in range(total):
            t = n / (total - 1)
            if direction == "asc":
                u = t
            elif direction == "des":
                u = 1 - t
            else:
                u = t * 2 if t < 0.5 else (1 - t) * 2

            f = freq_at(u)
            phase += f / RATE
            yield math.sin(2 * math.pi * (phase % 1.0)) * amp

        while True:
            yield 0.0

    write_wav(it(), dur)


def gen_shepard(volume, direction):
    amp = (volume / 100.0) * 0.75
    dur = PATTERN_SECONDS_SHEP
    total = int(RATE * dur)

    base = 55.0
    octaves = 8
    sigma = 1.2
    phases = [0.0] * octaves

    def gauss(x):
        return math.exp(-0.5 * (x / sigma) ** 2)

    def it():
        for n in range(total):
            t = n / (total - 1)
            pos = t if direction == "asc" else 1 - t
            frac = pos

            ssum = 0
            wsum = 0
            for i in range(octaves):
                f = base * (2 ** (i + frac))
                while f > 20000:
                    f *= 0.5
                while f < 20:
                    f *= 2

                x = (i + frac) - (octaves / 2)
                w = gauss(x)

                phases[i] += f / RATE
                ssum += math.sin(2 * math.pi * (phases[i] % 1.0)) * w
                wsum += w

            yield (ssum / wsum) * amp if wsum else 0

        while True:
            yield 0

    write_wav(it(), dur)


def gen_contact(volume):
    amp = (volume / 100.0) * 0.85
    dur = PATTERN_SECONDS_MOTIF
    notes = [392.0, 523.25, 659.25, 440.0, 587.33]
    tone_s = 0.35
    gap_s = 0.08

    seq = []
    for f in notes:
        seq.append(("tone", f, tone_s))
        seq.append(("gap", 0.0, gap_s))
    seq.append(("gap", 0.0, 0.5))

    phase = 0

    def it():
        nonlocal phase
        idx = 0
        seg_left = int(RATE * seq[idx][2])
        while True:
            typ, f, _ = seq[idx]
            if typ == "gap":
                yield 0
            else:
                phase += f / RATE
                yield math.sin(2 * math.pi * (phase % 1.0)) * amp

            seg_left -= 1
            if seg_left <= 0:
                idx = (idx + 1) % len(seq)
                seg_left = int(RATE * seq[idx][2])

    write_wav(it(), dur)


# ---------- Player ----------
def which_player():
    return shutil.which("paplay") or shutil.which("pw-play")


def start_loop(player):
    cmd = f'exec 2>>"{AUDIO_ERR}"; while true; do "{player}" "{PATTERN_WAV}"; done'
    return subprocess.Popen(
        ["/bin/sh", "-lc", cmd],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def stop_proc(p):
    if not p:
        return
    try:
        os.killpg(p.pid, signal.SIGTERM)
        time.sleep(0.1)
        os.killpg(p.pid, signal.SIGKILL)
    except Exception:
        pass


# ---------- Non-blocking stdin ----------
class StdinReader:
    def __init__(self):
        self.sel = selectors.DefaultSelector()
        self.fd = sys.stdin.fileno()
        os.set_blocking(self.fd, False)
        self.sel.register(self.fd, selectors.EVENT_READ)
        self.buf = b""

    def poll(self):
        events = self.sel.select(0)
        for key, _ in events:
            chunk = os.read(self.fd, 4096)
            if chunk:
                self.buf += chunk

        out = []
        while b"\n" in self.buf:
            line, self.buf = self.buf.split(b"\n", 1)
            out.append(line.decode().strip())
        return out


# ---------- State ----------
@dataclass
class State:
    page: str = "main"
    freq: int = 440
    volume: int = 70
    playing: bool = False
    selection_label: str = "440Hz"


# ---------- Main ----------
def main():
    emit({"type": "hello", "module": MODULE_NAME, "version": MODULE_VERSION})
    emit({"type": "page", "name": "main"})

    player = which_player()
    if not player:
        fatal("No audio player found")
        return 1

    reader = StdinReader()
    st = State()
    emit({"type": "state", **st.__dict__})

    proc = None
    last_hb = time.time()

    while True:
        for cmd in reader.poll():
            if cmd == "back":
                stop_proc(proc)
                emit({"type": "exit"})
                return 0

        now = time.time()
        if now - last_hb > HEARTBEAT_S:
            emit({"type": "state", **st.__dict__})
            last_hb = now

        time.sleep(0.01)


if __name__ == "__main__":
    main()
