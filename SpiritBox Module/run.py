#!/usr/bin/env python3
import time, json
from pathlib import Path
from gpiozero import Button
from luma.oled.device import ssd1306
from luma.core.interface.serial import i2c

from ui_common import render, draw_header, draw_row, draw_row_lr, draw_centered, LINE_H

# =====================
# CONFIG
# =====================
BTN_UP = 17
BTN_DOWN = 27
BTN_SELECT = 22
BTN_BACK = 23

HERE = Path(__file__).resolve().parent
SETTINGS_FILE = HERE / "settings.json"

DEFAULT_SETTINGS = {
    "band": "FM",
    "fm_min": 76.0,
    "fm_max": 108.0,
    "step_mhz": 0.1,
    "sweep_ms": 150,
    "scan_style": "LOOP",      # LOOP / BOUNCE / RANDOM (future)
    "mute_behavior": "NONE"    # NONE (future expansion)
}

# =====================
# OLED
# =====================
device = ssd1306(i2c(1, 0x3C), width=128, height=64)

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
# BUTTON EVENTS (with SELECT tap vs hold)
# =====================
def init_buttons(hold_ms=550):
    events = []
    select_pressed_at = {"t": None}

    def push(name):
        events.append(name)

    btn_up = Button(BTN_UP, pull_up=True, bounce_time=0.05)
    btn_down = Button(BTN_DOWN, pull_up=True, bounce_time=0.05)
    btn_back = Button(BTN_BACK, pull_up=True, bounce_time=0.05)
    btn_sel = Button(BTN_SELECT, pull_up=True, bounce_time=0.03)

    btn_up.when_pressed = lambda: push("up")
    btn_down.when_pressed = lambda: push("down")
    btn_back.when_pressed = lambda: push("back")

    def sel_down():
        select_pressed_at["t"] = time.time()

    def sel_up():
        t0 = select_pressed_at["t"]
        select_pressed_at["t"] = None
        if t0 is None:
            return
        dt = (time.time() - t0) * 1000
        if dt >= hold_ms:
            push("select_hold")
        else:
            push("select")

    btn_sel.when_pressed = sel_down
    btn_sel.when_released = sel_up

    def consume():
        if events:
            return events.pop(0)
        return None

    return consume, (btn_up, btn_down, btn_sel, btn_back)

# =====================
# TEA5767 (stub for now)
# =====================
def tune(freq_mhz: float):
    # TODO: implement TEA5767 I2C tune
    # This module is scaffold-first; tuning comes next.
    pass

# =====================
# UI: READY
# =====================
def ready_screen(settings, sel_idx):
    menu = ["START", "SETTINGS", "BACK"]
    def _draw(d):
        draw_header(d, "SPIRIT BOX")
        d.text((2, 20), f"FM {settings['fm_min']:.0f}-{settings['fm_max']:.0f} MHz", fill=255)
        d.text((2, 30), f"Rate: {int(settings['sweep_ms'])} ms", fill=255)
        y0 = 44
        for i, item in enumerate(menu):
            draw_row(d, y0 + i*10, f"{item}", selected=(i == sel_idx))
    return menu, _draw

# =====================
# UI: SETTINGS LIST (FM + Rate side-by-side)
# =====================
def settings_overview(settings, sel_idx):
    items = ["FM/RATE", "STEP", "SCAN", "BACK"]

    def _draw(d):
        draw_header(d, "SETTINGS")

        # Row 0: FM band + sweep rate side-by-side
        fm_left = f"FM: {settings['fm_min']:.0f}-{settings['fm_max']:.0f}"
        rate_right = f"{int(settings['sweep_ms'])}ms"
        draw_row_lr(d, 20, fm_left, rate_right, selected=(sel_idx == 0), right_x=84)

        # Row 1: Step
        draw_row(d, 30, f"Step: {settings['step_mhz']:.1f} MHz", selected=(sel_idx == 1))

        # Row 2: Scan style
        draw_row(d, 40, f"Scan: {settings['scan_style']}", selected=(sel_idx == 2))

        # Row 3: Back
        draw_row(d, 50, "Back", selected=(sel_idx == 3))

    return items, _draw

# =====================
# UI: EDIT MODES
# =====================
def edit_sweep_rate(settings, consume):
    original = settings["sweep_ms"]
    val = int(original)

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
            d.text((2, 62-10), "SEL save  BACK cancel", fill=255)

        render(device, _draw)

        ev = consume()
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

def edit_fm_band(settings, consume):
    # Simple edit: choose among common bands (future: per-field editing)
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
                draw_row(d, 34 + i*10, p[2], selected=(i == idx))
            d.text((2, 62-10), "SEL save  BACK cancel", fill=255)

        render(device, _draw)

        ev = consume()
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

def edit_scan_style(settings, consume):
    styles = ["LOOP", "BOUNCE", "RANDOM"]
    idx = styles.index(settings.get("scan_style", "LOOP")) if settings.get("scan_style", "LOOP") in styles else 0

    while True:
        def _draw(d):
            draw_header(d, "SCAN STYLE")
            for i, s in enumerate(styles):
                draw_row(d, 24 + i*10, s, selected=(i == idx))
            d.text((2, 62-10), "SEL save  BACK cancel", fill=255)

        render(device, _draw)

        ev = consume()
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

# =====================
# RUNNING: Sweep screen + hold-select to Settings
# =====================
def run_sweep(settings, consume):
    freq = float(settings["fm_min"])
    step = float(settings["step_mhz"])
    delay = max(0.05, int(settings["sweep_ms"]) / 1000.0)

    direction = 1  # for BOUNCE

    while True:
        tune(freq)

        def _draw(d):
            draw_header(d, "FM SWEEP")
            # Big frequency presentation (centered)
            draw_centered(d, 26, f"{freq:.1f} MHz", invert=False)
            d.text((2, 50), f"Rate: {int(settings['sweep_ms'])}ms", fill=255)
            d.text((2, 60), "BACK stop  HOLD=Settings", fill=255)

        render(device, _draw)

        ev = consume()
        if ev == "back":
            return  # stop sweep -> back to Ready
        if ev == "select_hold":
            # open settings while running (pause sweep visually)
            settings = settings_flow(settings, consume)
            save_settings(settings)
            # update sweep timing if changed
            delay = max(0.05, int(settings["sweep_ms"]) / 1000.0)
            # continue sweeping from current freq

        # Advance frequency based on scan style
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
        else:  # RANDOM (simple)
            # lightweight pseudo-random without imports
            freq = settings["fm_min"] + ( (freq * 13.7) % (settings["fm_max"] - settings["fm_min"]) )

        time.sleep(delay)

# =====================
# SETTINGS FLOW
# =====================
def settings_flow(settings, consume):
    sel = 0
    while True:
        items, draw_fn = settings_overview(settings, sel)
        render(device, draw_fn)

        ev = consume()
        if ev == "up":
            sel = (sel - 1) % len(items)
        elif ev == "down":
            sel = (sel + 1) % len(items)
        elif ev == "back":
            return settings
        elif ev == "select":
            if sel == 0:
                # FM/RATE row: open a mini submenu
                settings = fm_rate_submenu(settings, consume)
                save_settings(settings)
            elif sel == 1:
                # step size is locked to 0.1 for now; keep placeholder for future
                pass
            elif sel == 2:
                settings, _ = edit_scan_style(settings, consume)
                save_settings(settings)
            elif sel == 3:
                return settings

        time.sleep(0.05)

def fm_rate_submenu(settings, consume):
    # two options: edit band, edit rate
    opts = ["FM BAND", "SWEEP RATE", "BACK"]
    sel = 0
    while True:
        def _draw(d):
            draw_header(d, "FM / RATE")
            for i, o in enumerate(opts):
                draw_row(d, 24 + i*10, o, selected=(i == sel))
            d.text((2, 62-10), "SEL choose  BACK return", fill=255)

        render(device, _draw)

        ev = consume()
        if ev == "up":
            sel = (sel - 1) % len(opts)
        elif ev == "down":
            sel = (sel + 1) % len(opts)
        elif ev == "back":
            return settings
        elif ev == "select":
            if sel == 0:
                settings, _ = edit_fm_band(settings, consume)
            elif sel == 1:
                settings, _ = edit_sweep_rate(settings, consume)
            else:
                return settings
            save_settings(settings)

        time.sleep(0.05)

# =====================
# MAIN
# =====================
def main():
    settings = load_settings()
    consume, _btn_refs = init_buttons()

    # READY screen selection
    sel = 0
    while True:
        menu, draw_fn = ready_screen(settings, sel)
        render(device, draw_fn)

        ev = consume()
        if ev == "up":
            sel = (sel - 1) % len(menu)
        elif ev == "down":
            sel = (sel + 1) % len(menu)
        elif ev == "back":
            return
        elif ev == "select":
            if menu[sel] == "START":
                run_sweep(settings, consume)
                settings = load_settings()  # refresh in case changed during run
            elif menu[sel] == "SETTINGS":
                settings = settings_flow(settings, consume)
                save_settings(settings)
            else:
                return

        time.sleep(0.05)

if __name__ == "__main__":
    main()
