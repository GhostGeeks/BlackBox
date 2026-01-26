#!/usr/bin/env python3
import sys
import time
import json
import select
from pathlib import Path

from luma.core.interface.serial import i2c
from luma.oled.device import ssd1306

# Shared UI helpers (package import)
from oled.ui_common import render, draw_header, draw_row, draw_row_lr, draw_centered

# =====================
# FILES / SETTINGS
# =====================
HERE = Path(__file__).resolve().parent
SETTINGS_FILE = HERE / "settings.json"

DEFAULT_SETTINGS = {
    "band": "FM",
    "fm_min": 76.0,
    "fm_max": 108.0,
    "step_mhz": 0.1,
    "sweep_ms": 150,
    "scan_style": "LOOP",      # LOOP / BOUNCE / RANDOM
    "mute_behavior": "NONE"    # future
}

# =====================
# OLED INIT (correct luma API)
# =====================
serial = i2c(port=1, address=0x3C)
device = ssd1306(serial, width=128, height=64)

# =====================
# SETTINGS
# =====================
def load_settings():
    if SETTINGS_FILE.exists():
        try:
            s = json.loads(SETTINGS_FILE.read_text())
            out = DEFAULT_SETTINGS.copy()
            out.update(s)
            return out
        except Exception:
            pass
    SETTINGS_FILE.write_text(json.dumps(DEFAULT_SETTINGS, indent=2))
    return DEFAULT_SETTINGS.copy()

def save_settings(s):
    SETTINGS_FILE.write_text(json.dumps(s, indent=2))

# =====================
# BUTTON INPUT (from stdin)
# =====================
def read_event():
    """
    Reads one event token from stdin, non-blocking.
    Expected tokens: up, down, select, select_hold, back
    app.py will send these lines when buttons pressed.
    """
    r, _, _ = select.select([sys.stdin], [], [], 0)
    if not r:
        return None
    line = sys.stdin.readline()
    if not line:
        return None
    return line.strip()

# =====================
# TEA5767 (stub for now)
# =====================
def tune(freq_mhz: float):
    # TODO: implement TEA5767 tuning (next step)
    pass

# =====================
# UI SCREENS
# =====================
def screen_ready(settings, sel_idx):
    menu = ["START", "SETTINGS", "BACK"]

    def _draw(d):
        draw_header(d, "SPIRIT BOX")
        d.text((2, 20), f"FM {settings['fm_min']:.0f}-{settings['fm_max']:.0f} MHz", fill=255)
        d.text((2, 30), f"Rate: {int(settings['sweep_ms'])} ms", fill=255)
        y0 = 44
        for i, item in enumerate(menu):
            draw_row(d, y0 + i * 10, item, selected=(i == sel_idx))

    return menu, _draw

def screen_settings(settings, sel_idx):
    # No "BACK" selection item — BACK is a hardware button
    items = ["FM/RATE", "STEP", "SCAN"]

    def _draw(d):
        draw_header(d, "SETTINGS")

        # Side-by-side row (FM band + sweep rate)
        fm_left = f"FM: {settings['fm_min']:.0f}-{settings['fm_max']:.0f}"
        rate_right = f"{int(settings['sweep_ms'])}ms"
        draw_row_lr(d, 20, fm_left, rate_right, selected=(sel_idx == 0), right_x=84)

        draw_row(d, 30, f"Step: {settings['step_mhz']:.1f} MHz", selected=(sel_idx == 1))
        draw_row(d, 40, f"Scan: {settings['scan_style']}", selected=(sel_idx == 2))

        # One hint line only
        d.text((2, 60), "SEL choose   BACK return", fill=255)

    return items, _draw

def screen_running(freq, settings):
    def _draw(d):
        draw_header(d, "FM SWEEP")
        draw_centered(d, 26, f"{freq:.1f} MHz", invert=False)
        d.text((2, 50), f"Rate: {int(settings['sweep_ms'])}ms", fill=255)
        d.text((2, 60), "BACK stop  HOLD=Settings", fill=255)
    return _draw

# =====================
# EDIT MODES
# =====================
def edit_sweep_rate(settings):
    original = int(settings["sweep_ms"])
    val = original
    MIN_MS, MAX_MS, STEP_MS = 50, 350, 50

    blink = False
    last_blink = time.time()

    while True:
        now = time.time()
        if now - last_blink > 0.35:
            blink = not blink
            last_blink = now

        def _draw(d):
            draw_header(d, "SWEEP RATE")
            draw_centered(d, 28, f"{val} ms", invert=blink)
            d.text((2, 52), "UP/DN adjust", fill=255)
            d.text((2, 60), "SEL save  BACK cancel", fill=255)

        render(device, _draw)

        ev = read_event()
        if ev == "up":
            val = min(MAX_MS, val + STEP_MS)
        elif ev == "down":
            val = max(MIN_MS, val - STEP_MS)
        elif ev == "select":
            settings["sweep_ms"] = val
            return settings, True
        elif ev == "back":
            settings["sweep_ms"] = original
            return settings, False

        time.sleep(0.03)

def edit_fm_band(settings):
    presets = [
        (76.0, 108.0, "76–108"),
        (87.5, 108.0, "87.5–108"),
    ]
    cur = (settings["fm_min"], settings["fm_max"])
    idx = 0
    for i, p in enumerate(presets):
        if (p[0], p[1]) == cur:
            idx = i

    while True:
        def _draw(d):
            draw_header(d, "FM BAND")
            d.text((2, 24), "Choose range:", fill=255)
            for i, p in enumerate(presets):
                draw_row(d, 34 + i * 10, p[2], selected=(i == idx))
            d.text((2, 60), "SEL save  BACK cancel", fill=255)

        render(device, _draw)

        ev = read_event()
        if ev == "up":
            idx = (idx - 1) % len(presets)
        elif ev == "down":
            idx = (idx + 1) % len(presets)
        elif ev == "select":
            settings["fm_min"], settings["fm_max"] = presets[idx][0], presets[idx][1]
            return settings, True
        elif ev == "back":
            return settings, False

        time.sleep(0.03)

def edit_scan_style(settings):
    styles = ["LOOP", "BOUNCE", "RANDOM"]
    cur = settings.get("scan_style", "LOOP")
    idx = styles.index(cur) if cur in styles else 0

    while True:
        def _draw(d):
            draw_header(d, "SCAN STYLE")
            for i, s in enumerate(styles):
                draw_row(d, 24 + i * 10, s, selected=(i == idx))
            d.text((2, 60), "SEL save  BACK cancel", fill=255)

        render(device, _draw)

        ev = read_event()
        if ev == "up":
            idx = (idx - 1) % len(styles)
        elif ev == "down":
            idx = (idx + 1) % len(styles)
        elif ev == "select":
            settings["scan_style"] = styles[idx]
            return settings, True
        elif ev == "back":
            return settings, False

        time.sleep(0.03)

def fm_rate_submenu(settings):
    opts = ["FM BAND", "SWEEP RATE", "BACK"]
    sel = 0

    while True:
        def _draw(d):
            draw_header(d, "FM / RATE")
            for i, o in enumerate(opts):
                draw_row(d, 24 + i * 10, o, selected=(i == sel))
            d.text((2, 60), "SEL choose  BACK return", fill=255)

        render(device, _draw)

        ev = read_event()
        if ev == "up":
            sel = (sel - 1) % len(opts)
        elif ev == "down":
            sel = (sel + 1) % len(opts)
        elif ev == "back":
            return settings
        elif ev == "select":
            if sel == 0:
                settings, _ = edit_fm_band(settings)
                save_settings(settings)
            elif sel == 1:
                settings, _ = edit_sweep_rate(settings)
                save_settings(settings)
            else:
                return settings

        time.sleep(0.05)

def settings_flow(settings):
    sel = 0
    while True:
        items, draw_fn = screen_settings(settings, sel)
        render(device, draw_fn)

        ev = read_event()
        if ev == "up":
            sel = (sel - 1) % len(items)
        elif ev == "down":
            sel = (sel + 1) % len(items)
        elif ev == "back":
            return settings
               elif ev == "select":
            if sel == 0:
                settings = fm_rate_submenu(settings)
            elif sel == 1:
                # Step size locked for now (0.1); future edit hook here
                pass
            elif sel == 2:
                settings, _ = edit_scan_style(settings)
                save_settings(settings)


        time.sleep(0.05)

# =====================
# RUN LOOP
# =====================
def run_sweep(settings):
    freq = float(settings["fm_min"])
    step = float(settings["step_mhz"])
    direction = 1

    while True:
        tune(freq)
        render(device, screen_running(freq, settings))

        ev = read_event()
        if ev == "back":
            return settings
        if ev == "select_hold":
            settings = settings_flow(settings)
            save_settings(settings)

        style = settings.get("scan_style", "LOOP")
        if style == "LOOP":
            freq += step
            if freq > settings["fm_max"]:
                freq = settings["fm_min"]
        elif style == "BOUNCE":
            freq += step * direction
            if freq >= settings["fm_max"]:
                freq = settings["fm_max"]
                direction = -1
            elif freq <= settings["fm_min"]:
                freq = settings["fm_min"]
                direction = 1
        else:  # RANDOM lightweight
            span = settings["fm_max"] - settings["fm_min"]
            freq = settings["fm_min"] + ((freq * 13.7) % span)

        delay = max(0.05, int(settings["sweep_ms"]) / 1000.0)
        time.sleep(delay)

# =====================
# MAIN
# =====================
def main():
    settings = load_settings()
    sel = 0

    while True:
        menu, draw_fn = screen_ready(settings, sel)
        render(device, draw_fn)

        ev = read_event()
        if ev == "up":
            sel = (sel - 1) % len(menu)
        elif ev == "down":
            sel = (sel + 1) % len(menu)
        elif ev == "back":
            return
        elif ev == "select":
            if menu[sel] == "START":
                settings = run_sweep(settings)
                settings = load_settings()
            elif menu[sel] == "SETTINGS":
                settings = settings_flow(settings)
                save_settings(settings)
            else:
                return

        time.sleep(0.05)

if __name__ == "__main__":
    main()
