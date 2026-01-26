#!/usr/bin/env python3
"""
Ghost Geeks - Tone Generator module (OLED UI + stdin button control)

Buttons arrive via stdin lines from the parent app:
  up, down, select, select_hold, back

Audio:
  Streams continuous PCM (s16le, 48kHz, stereo) to PipeWire using pw-cat.
  This lets your existing routing send audio to line-out and/or Bluetooth.

Persistence:
  ~/.config/ghostgeeks/tone_generator.json (settings + favorites)
"""

from __future__ import annotations

import json
import math
import os
import queue
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

from luma.core.interface.serial import i2c
from luma.oled.device import ssd1306
from luma.core.render import canvas

# -----------------------------
# OLED SETUP
# -----------------------------
OLED_ADDR = 0x3C
OLED_W, OLED_H = 128, 64

serial = i2c(port=1, address=OLED_ADDR)
device = ssd1306(serial, width=OLED_W, height=OLED_H)

# Layout constants (tuned to avoid bottom cutoff)
TOP_H = 12
BOTTOM_H = 10
LINE_H = 10
DIV_Y_TOP = TOP_H
DIV_Y_BOTTOM = OLED_H - BOTTOM_H - 1

def draw_divider(d, y: int):
    d.line((0, y, OLED_W - 1, y), fill=255)

def draw_header(d, title: str):
    d.text((0, 0), title[:16], fill=255)
    draw_divider(d, DIV_Y_TOP)

def draw_footer(d, text: str):
    draw_divider(d, DIV_Y_BOTTOM)
    d.text((0, OLED_H - BOTTOM_H), text[:21], fill=255)

def draw_menu(d, title: str, items: List[str], idx: int, hint: str):
    draw_header(d, title)
    y0 = TOP_H + 1
    visible_lines = (DIV_Y_BOTTOM - y0) // LINE_H
    if visible_lines < 1:
        visible_lines = 1

    # scroll window
    start = 0
    if idx >= visible_lines:
        start = idx - visible_lines + 1
    end = min(len(items), start + visible_lines)

    y = y0
    for i in range(start, end):
        prefix = ">" if i == idx else " "
        txt = f"{prefix} {items[i]}"
        d.text((0, y), txt[:21], fill=255)
        y += LINE_H

    draw_footer(d, hint)

def draw_big_value(d, title: str, big: str, lines: List[str], hint: str):
    draw_header(d, title)
    # big centered-ish
    d.text((0, TOP_H + 2), big[:10], fill=255)
    y = TOP_H + 2 + 18
    for ln in lines[:3]:
        d.text((0, y), ln[:21], fill=255)
        y += LINE_H
    draw_footer(d, hint)

def oled_render(fn):
    with canvas(device) as d:
        fn(d)

# -----------------------------
# INPUT EVENTS (stdin)
# -----------------------------
def stdin_event_queue() -> "queue.Queue[str]":
    q: "queue.Queue[str]" = queue.Queue()

    def reader():
        while True:
            line = sys.stdin.readline()
            if not line:
                time.sleep(0.05)
                continue
            line = line.strip()
            if line:
                q.put(line)

    import threading
    t = threading.Thread(target=reader, daemon=True)
    t.start()
    return q

# -----------------------------
# CONFIG / PERSISTENCE
# -----------------------------
CFG_DIR = Path.home() / ".config" / "ghostgeeks"
CFG_PATH = CFG_DIR / "tone_generator.json"

DEFAULTS = {
    "volume": 0.65,          # 0.0 - 1.0
    "steady_hz": 432.0,
    "pulse_hz": 528.0,
    "pulse_on_ms": 250,
    "pulse_off_ms": 250,
    "sweep_start_hz": 100.0,
    "sweep_end_hz": 1200.0,
    "sweep_step_hz": 5.0,
    "sweep_step_ms": 100,
    "favorites": [432.0, 528.0],
}

def load_cfg() -> dict:
    CFG_DIR.mkdir(parents=True, exist_ok=True)
    if not CFG_PATH.exists():
        save_cfg(DEFAULTS.copy())
        return DEFAULTS.copy()
    try:
        data = json.loads(CFG_PATH.read_text())
    except Exception:
        data = {}
    merged = DEFAULTS.copy()
    merged.update({k: data.get(k, merged[k]) for k in merged.keys()})
    # normalize
    merged["favorites"] = list(dict.fromkeys([float(x) for x in merged.get("favorites", [])]))[:30]
    merged["volume"] = float(max(0.0, min(1.0, merged["volume"])))
    return merged

def save_cfg(cfg: dict):
    CFG_DIR.mkdir(parents=True, exist_ok=True)
    CFG_PATH.write_text(json.dumps(cfg, indent=2, sort_keys=True))

# -----------------------------
# AUDIO ENGINE (PipeWire pw-cat)
# -----------------------------
SAMPLE_RATE = 48000
CHANNELS = 2
FRAME_SAMPLES = 960  # 20ms @ 48kHz
TWOPI = 2.0 * math.pi

@dataclass
class AudioProc:
    p: subprocess.Popen
    phase: float = 0.0

def have_pw_cat() -> bool:
    return subprocess.call(["bash", "-lc", "command -v pw-cat >/dev/null 2>&1"]) == 0

def start_audio_stream() -> Optional[AudioProc]:
    """
    Start pw-cat playback process that reads raw PCM from stdin.
    """
    if not have_pw_cat():
        return None
    cmd = [
        "pw-cat",
        "--playback",
        "--rate", str(SAMPLE_RATE),
        "--channels", str(CHANNELS),
        "--format", "s16le",
    ]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return AudioProc(p=p, phase=0.0)

def stop_audio_stream(ap: Optional[AudioProc]):
    if not ap:
        return
    try:
        if ap.p.stdin:
            ap.p.stdin.close()
    except Exception:
        pass
    try:
        ap.p.terminate()
    except Exception:
        pass
    try:
        ap.p.wait(timeout=0.5)
    except Exception:
        try:
            ap.p.kill()
        except Exception:
            pass

def s16(x: float) -> int:
    # clamp and convert
    if x > 1.0: x = 1.0
    if x < -1.0: x = -1.0
    return int(x * 32767.0)

def write_frames(ap: AudioProc, samples: List[int]) -> bool:
    """
    samples: interleaved stereo int16 list length = FRAME_SAMPLES*CHANNELS
    """
    if ap.p.poll() is not None:
        return False
    try:
        b = bytearray()
        for v in samples:
            b += int(v).to_bytes(2, byteorder="little", signed=True)
        ap.p.stdin.write(b)  # type: ignore
        ap.p.stdin.flush()   # type: ignore
        return True
    except Exception:
        return False

def gen_sine_block(ap: AudioProc, hz: float, vol: float) -> List[int]:
    inc = TWOPI * hz / SAMPLE_RATE
    out: List[int] = []
    ph = ap.phase
    for _ in range(FRAME_SAMPLES):
        v = math.sin(ph) * vol
        iv = s16(v)
        out.append(iv)  # L
        out.append(iv)  # R
        ph += inc
        if ph > TWOPI:
            ph -= TWOPI
    ap.phase = ph
    return out

def gen_silence_block() -> List[int]:
    return [0] * (FRAME_SAMPLES * CHANNELS)

# -----------------------------
# PRESETS / “CLAIMED” TONES
# -----------------------------
# Commonly cited “Solfeggio / chakra” lists vary widely; these are the popular ones.
# (You asked for the claimed tones, not scientific validation.)
PRESETS: List[Tuple[str, float]] = [
    ("A=432", 432.0),
    ("528", 528.0),
    ("396", 396.0),
    ("417", 417.0),
    ("639", 639.0),
    ("741", 741.0),
    ("852", 852.0),
    ("963", 963.0),
    ("174", 174.0),
    ("285", 285.0),
]

# Tritone sequences:
# Use equal temperament: tritone = +6 semitones => ratio = 2^(6/12) = sqrt(2).
def tritone_ratio() -> float:
    return math.sqrt(2.0)

# -----------------------------
# UI STATE MACHINE
# -----------------------------
STATE_MAIN = "main"
STATE_STEADY = "steady"
STATE_PULSE = "pulse"
STATE_SWEEP = "sweep"
STATE_PRESETS = "presets"
STATE_FAVS = "favorites"
STATE_SETTINGS = "settings"
STATE_SEQUENCE = "sequence"

MAIN_ITEMS = [
    "Steady Tone",
    "Pulse Tone",
    "Sweep Tones",
    "Presets",
    "Favorites",
    "Settings",
    "Exit",
]

SEQ_ITEMS = [
    "Tritone (rise)",
    "Tritone (fall)",
    "Back",
]

SETTINGS_ITEMS = [
    "Volume",
    "Pulse on/off",
    "Sweep range",
    "Sweep timing",
    "Back",
]

def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v

def fmt_hz(hz: float) -> str:
    if hz >= 1000:
        return f"{hz/1000.0:.3f}kHz"
    return f"{hz:.1f}Hz"

# -----------------------------
# PLAYER MODES
# -----------------------------
class Mode:
    STOPPED = 0
    STEADY = 1
    PULSE = 2
    SWEEP = 3
    SEQ = 4

# -----------------------------
# MAIN LOOP
# -----------------------------
def main():
    cfg = load_cfg()

    q = stdin_event_queue()

    state = STATE_MAIN
    idx = 0
    presets_idx = 0
    fav_idx = 0
    settings_idx = 0
    seq_idx = 0

    mode = Mode.STOPPED
    ap = start_audio_stream()
    if ap is None:
        oled_render(lambda d: draw_big_value(d, "TONE GEN", "NO AUDIO", ["Missing pw-cat", "Install pipewire", "or pw-cat"], "BACK = exit"))
        time.sleep(1.2)
        return

    # sweep internals
    sweep_hz = float(cfg["sweep_start_hz"])
    sweep_dir = 1

    # pulse internals
    pulse_phase_ms = 0
    pulse_is_on = True

    # sequence internals
    seq_step = 0
    seq_hz = 110.0  # base
    seq_next_change = time.monotonic()

    last_draw = 0.0

    def stop_playback():
        nonlocal mode, pulse_phase_ms, pulse_is_on
        mode = Mode.STOPPED
        pulse_phase_ms = 0
        pulse_is_on = True

    def start_steady():
        nonlocal mode
        mode = Mode.STEADY

    def start_pulse():
        nonlocal mode, pulse_phase_ms, pulse_is_on
        mode = Mode.PULSE
        pulse_phase_ms = 0
        pulse_is_on = True

    def start_sweep():
        nonlocal mode, sweep_hz, sweep_dir
        mode = Mode.SWEEP
        sweep_hz = float(cfg["sweep_start_hz"])
        sweep_dir = 1

    def start_sequence(which: str):
        nonlocal mode, seq_step, seq_hz, seq_next_change
        mode = Mode.SEQ
        seq_step = 0
        seq_hz = 110.0
        seq_next_change = time.monotonic()
        cfg["_seq_kind"] = which

    def toggle_favorite(hz: float):
        favs = cfg["favorites"]
        hz = float(round(hz, 2))
        if hz in favs:
            favs.remove(hz)
        else:
            favs.append(hz)
        cfg["favorites"] = favs[:30]
        save_cfg(cfg)

    def draw():
        nonlocal last_draw
        now = time.monotonic()
        if now - last_draw < 0.05:
            return
        last_draw = now

        def _draw(d):
            nonlocal state, idx, presets_idx, fav_idx, settings_idx, seq_idx, sweep_hz
            if state == STATE_MAIN:
                draw_menu(d, "TONE GEN", MAIN_ITEMS, idx, "SEL=go  BACK=exit")

            elif state == STATE_PRESETS:
                items = [f"{name}  {fmt_hz(hz)}" for (name, hz) in PRESETS] + ["Tritone seq", "Back"]
                draw_menu(d, "PRESETS", items, presets_idx, "SEL=play  HOLD=fav")

            elif state == STATE_FAVS:
                favs = cfg["favorites"]
                items = [fmt_hz(x) for x in favs] + ["Back"]
                draw_menu(d, "FAVORITES", items, fav_idx, "SEL=play  HOLD=del")

            elif state == STATE_SETTINGS:
                # show current values inline to keep it “device-like”
                items = [
                    f"Volume: {int(cfg['volume']*100):d}%",
                    f"Pulse: {cfg['pulse_on_ms']} / {cfg['pulse_off_ms']}ms",
                    f"Sweep: {fmt_hz(cfg['sweep_start_hz'])}-{fmt_hz(cfg['sweep_end_hz'])}",
                    f"Step: {cfg['sweep_step_hz']}Hz @ {cfg['sweep_step_ms']}ms",
                    "Back",
                ]
                draw_menu(d, "SETTINGS", items, settings_idx, "UP/DN edit  BACK")

            elif state == STATE_STEADY:
                hz = float(cfg["steady_hz"])
                big = fmt_hz(hz)
                lines = [
                    "Mode: Steady",
                    f"Vol: {int(cfg['volume']*100)}%",
                    "HOLD=save fav",
                ]
                draw_big_value(d, "STEADY", big, lines, "SEL=stop  BACK=menu")

            elif state == STATE_PULSE:
                hz = float(cfg["pulse_hz"])
                big = fmt_hz(hz)
                lines = [
                    "Mode: Pulse",
                    f"{cfg['pulse_on_ms']}/{cfg['pulse_off_ms']}ms",
                    "HOLD=save fav",
                ]
                draw_big_value(d, "PULSE", big, lines, "SEL=stop  BACK=menu")

            elif state == STATE_SWEEP:
                big = fmt_hz(sweep_hz)
                lines = [
                    "Mode: Sweep",
                    f"{fmt_hz(cfg['sweep_start_hz'])}->{fmt_hz(cfg['sweep_end_hz'])}",
                    f"Step {cfg['sweep_step_hz']}Hz",
                ]
                draw_big_value(d, "SWEEP", big, lines, "SEL=stop  BACK=menu")

            elif state == STATE_SEQUENCE:
                kind = cfg.get("_seq_kind", "tritone_rise")
                label = "Tritone rise" if kind == "tritone_rise" else "Tritone fall"
                big = fmt_hz(seq_hz)
                lines = [
                    f"Mode: {label}",
                    "Patterned",
                    "HOLD=save base",
                ]
                draw_big_value(d, "SEQUENCE", big, lines, "SEL=stop  BACK=menu")

            else:
                draw_big_value(d, "TONE GEN", "READY", ["", "", ""], "BACK=exit")

        oled_render(_draw)

    # Main loop
    try:
        while True:
            # Handle input events
            ev = None
            try:
                ev = q.get_nowait()
            except queue.Empty:
                ev = None

            if ev:
                # ---------- GLOBAL BACK behavior ----------
                if ev == "back":
                    if state == STATE_MAIN:
                        # Exit module entirely
                        stop_playback()
                        break
                    else:
                        # Return to main menu without exiting module
                        stop_playback()
                        state = STATE_MAIN
                        idx = 0

                # ---------- SELECT HOLD as quick-actions ----------
                elif ev == "select_hold":
                    if state in (STATE_STEADY, STATE_PULSE, STATE_SWEEP, STATE_SEQUENCE):
                        # Save current key frequency to favorites
                        if state == STATE_STEADY:
                            toggle_favorite(float(cfg["steady_hz"]))
                        elif state == STATE_PULSE:
                            toggle_favorite(float(cfg["pulse_hz"]))
                        elif state == STATE_SWEEP:
                            toggle_favorite(float(round(sweep_hz, 2)))
                        elif state == STATE_SEQUENCE:
                            toggle_favorite(float(round(seq_hz, 2)))

                # ---------- STATE SPECIFIC ----------
                elif state == STATE_MAIN:
                    if ev == "up":
                        idx = (idx - 1) % len(MAIN_ITEMS)
                    elif ev == "down":
                        idx = (idx + 1) % len(MAIN_ITEMS)
                    elif ev == "select":
                        choice = MAIN_ITEMS[idx]
                        if choice == "Steady Tone":
                            state = STATE_STEADY
                            start_steady()
                        elif choice == "Pulse Tone":
                            state = STATE_PULSE
                            start_pulse()
                        elif choice == "Sweep Tones":
                            state = STATE_SWEEP
                            start_sweep()
                        elif choice == "Presets":
                            state = STATE_PRESETS
                            presets_idx = 0
                        elif choice == "Favorites":
                            state = STATE_FAVS
                            fav_idx = 0
                        elif choice == "Settings":
                            state = STATE_SETTINGS
                            settings_idx = 0
                        elif choice == "Exit":
                            stop_playback()
                            break

                elif state == STATE_PRESETS:
                    items_len = len(PRESETS) + 2
                    if ev == "up":
                        presets_idx = (presets_idx - 1) % items_len
                    elif ev == "down":
                        presets_idx = (presets_idx + 1) % items_len
                    elif ev == "select":
                        if presets_idx < len(PRESETS):
                            _, hz = PRESETS[presets_idx]
                            cfg["steady_hz"] = float(hz)
                            save_cfg(cfg)
                            state = STATE_STEADY
                            start_steady()
                        else:
                            tail = presets_idx - len(PRESETS)
                            if tail == 0:
                                # Tritone sequences submenu (simple)
                                # Use select to toggle between rise/fall quickly
                                state = STATE_SEQUENCE
                                start_sequence("tritone_rise")
                            else:
                                state = STATE_MAIN
                                idx = 0

                elif state == STATE_FAVS:
                    favs = cfg["favorites"]
                    items_len = len(favs) + 1
                    if ev == "up":
                        fav_idx = (fav_idx - 1) % items_len
                    elif ev == "down":
                        fav_idx = (fav_idx + 1) % items_len
                    elif ev == "select":
                        if fav_idx < len(favs):
                            cfg["steady_hz"] = float(favs[fav_idx])
                            save_cfg(cfg)
                            state = STATE_STEADY
                            start_steady()
                        else:
                            state = STATE_MAIN
                            idx = 0
                    elif ev == "select_hold":
                        if fav_idx < len(favs):
                            hz = float(favs[fav_idx])
                            favs.remove(hz)
                            cfg["favorites"] = favs
                            save_cfg(cfg)
                            fav_idx = int(clamp(fav_idx, 0, max(0, len(favs))))

                elif state == STATE_SETTINGS:
                    # settings edit is UP/DOWN only (select does nothing except back via back button)
                    if ev == "up":
                        if settings_idx == 0:
                            cfg["volume"] = float(clamp(cfg["volume"] + 0.05, 0.0, 1.0))
                        elif settings_idx == 1:
                            cfg["pulse_on_ms"] = int(clamp(cfg["pulse_on_ms"] + 50, 50, 2000))
                        elif settings_idx == 2:
                            cfg["sweep_end_hz"] = float(clamp(cfg["sweep_end_hz"] + 10.0, 50.0, 20000.0))
                        elif settings_idx == 3:
                            cfg["sweep_step_ms"] = int(clamp(cfg["sweep_step_ms"] + 50, 50, 350))
                        save_cfg(cfg)

                    elif ev == "down":
                        if settings_idx == 0:
                            cfg["volume"] = float(clamp(cfg["volume"] - 0.05, 0.0, 1.0))
                        elif settings_idx == 1:
                            cfg["pulse_on_ms"] = int(clamp(cfg["pulse_on_ms"] - 50, 50, 2000))
                        elif settings_idx == 2:
                            cfg["sweep_end_hz"] = float(clamp(cfg["sweep_end_hz"] - 10.0, 50.0, 20000.0))
                        elif settings_idx == 3:
                            cfg["sweep_step_ms"] = int(clamp(cfg["sweep_step_ms"] - 50, 50, 350))
                        save_cfg(cfg)

                    elif ev == "select":
                        settings_idx = (settings_idx + 1) % len(SETTINGS_ITEMS)

                elif state == STATE_STEADY:
                    if ev == "up":
                        cfg["steady_hz"] = float(clamp(cfg["steady_hz"] + 1.0, 1.0, 20000.0))
                        save_cfg(cfg)
                    elif ev == "down":
                        cfg["steady_hz"] = float(clamp(cfg["steady_hz"] - 1.0, 1.0, 20000.0))
                        save_cfg(cfg)
                    elif ev == "select":
                        stop_playback()
                        state = STATE_MAIN
                        idx = 0

                elif state == STATE_PULSE:
                    if ev == "up":
                        cfg["pulse_hz"] = float(clamp(cfg["pulse_hz"] + 1.0, 1.0, 20000.0))
                        save_cfg(cfg)
                    elif ev == "down":
                        cfg["pulse_hz"] = float(clamp(cfg["pulse_hz"] - 1.0, 1.0, 20000.0))
                        save_cfg(cfg)
                    elif ev == "select":
                        stop_playback()
                        state = STATE_MAIN
                        idx = 0

                elif state == STATE_SWEEP:
                    if ev == "up":
                        cfg["sweep_step_hz"] = float(clamp(cfg["sweep_step_hz"] + 1.0, 0.1, 2000.0))
                        save_cfg(cfg)
                    elif ev == "down":
                        cfg["sweep_step_hz"] = float(clamp(cfg["sweep_step_hz"] - 1.0, 0.1, 2000.0))
                        save_cfg(cfg)
                    elif ev == "select":
                        stop_playback()
                        state = STATE_MAIN
                        idx = 0

                elif state == STATE_SEQUENCE:
                    if ev == "select":
                        stop_playback()
                        state = STATE_MAIN
                        idx = 0
                    elif ev == "up" or ev == "down":
                        # toggle rise/fall
                        if cfg.get("_seq_kind") == "tritone_rise":
                            start_sequence("tritone_fall")
                        else:
                            start_sequence("tritone_rise")

            # Audio generation tick (20ms blocks)
            vol = float(cfg["volume"])

            if mode == Mode.STEADY and state == STATE_STEADY:
                hz = float(cfg["steady_hz"])
                block = gen_sine_block(ap, hz, vol)
                if not write_frames(ap, block):
                    break

            elif mode == Mode.PULSE and state == STATE_PULSE:
                hz = float(cfg["pulse_hz"])
                on_ms = int(cfg["pulse_on_ms"])
                off_ms = int(cfg["pulse_off_ms"])
                # accumulate ~20ms at a time
                pulse_phase_ms += 20
                if pulse_is_on:
                    block = gen_sine_block(ap, hz, vol)
                    if pulse_phase_ms >= on_ms:
                        pulse_is_on = False
                        pulse_phase_ms = 0
                else:
                    block = gen_silence_block()
                    if pulse_phase_ms >= off_ms:
                        pulse_is_on = True
                        pulse_phase_ms = 0
                if not write_frames(ap, block):
                    break

            elif mode == Mode.SWEEP and state == STATE_SWEEP:
                step = float(cfg["sweep_step_hz"])
                start_hz = float(cfg["sweep_start_hz"])
                end_hz = float(cfg["sweep_end_hz"])
                step_ms = int(cfg["sweep_step_ms"])

                # generate audio at current sweep_hz
                block = gen_sine_block(ap, sweep_hz, vol)
                ok = write_frames(ap, block)
                if not ok:
                    break

                # update sweep periodically
                # (20ms tick, only change when enough ms pass)
                if not hasattr(main, "_sweep_acc_ms"):
                    main._sweep_acc_ms = 0  # type: ignore
                main._sweep_acc_ms += 20  # type: ignore
                if main._sweep_acc_ms >= step_ms:  # type: ignore
                    main._sweep_acc_ms = 0  # type: ignore
                    sweep_hz += step * sweep_dir
                    if sweep_hz >= end_hz:
                        sweep_hz = end_hz
                        sweep_dir = -1
                    elif sweep_hz <= start_hz:
                        sweep_hz = start_hz
                        sweep_dir = 1

            elif mode == Mode.SEQ and state == STATE_SEQUENCE:
                kind = cfg.get("_seq_kind", "tritone_rise")
                base = 110.0
                ratio = tritone_ratio()

                now = time.monotonic()
                if now >= seq_next_change:
                    seq_step += 1
                    # 8-step little phrase
                    n = seq_step % 8
                    if kind == "tritone_rise":
                        # base -> tritone -> base*2 -> tritone*2 ...
                        seq_hz = base * (2 ** (n // 2)) * (ratio if (n % 2 == 1) else 1.0)
                    else:
                        # reverse feel
                        seq_hz = base * (2 ** (3 - (n // 2))) * (ratio if (n % 2 == 1) else 1.0)
                    seq_next_change = now + 0.25  # 250ms per step

                block = gen_sine_block(ap, seq_hz, vol)
                if not write_frames(ap, block):
                    break

            else:
                # not playing; feed silence so pipe doesn't stall noisily
                block = gen_silence_block()
                if not write_frames(ap, block):
                    break
                time.sleep(0.02)

            # Draw UI
            draw()

    finally:
        stop_audio_stream(ap)
        # clear display on exit so parent redraw is obvious
        oled_render(lambda d: d.rectangle((0, 0, OLED_W - 1, OLED_H - 1), outline=0, fill=0))


if __name__ == "__main__":
    main()
