#!/usr/bin/env python3
import sys
import time
import json
import select
from pathlib import Path

from luma.core.interface.serial import i2c
from luma.oled.device import ssd1306
from luma.core.render import canvas

# ============================================================
# OLED CONFIG
# ============================================================
I2C_PORT = 1
I2C_ADDR = 0x3C
OLED_W, OLED_H = 128, 64

serial = i2c(port=I2C_PORT, address=I2C_ADDR)
device = ssd1306(serial, width=OLED_W, height=OLED_H)

# ============================================================
# FILES
# ============================================================
HERE = Path(__file__).resolve().parent
SETTINGS_FILE = HERE / "settings.json"

DEFAULT_SETTINGS = {
    "step_mhz": 0.1,          # fixed by request; still stored for future
    "sweep_ms": 150,          # 50..350 step 50
    "scan_style": "LOOP",     # LOOP / BOUNCE / RANDOM
    "volume": 80,             # 0..100 (global)
}

# ============================================================
# UI LAYOUT (tuned to avoid footer clipping)
# ============================================================
HEADER_Y = 0
DIVIDER_Y = 12

LIST_Y0 = 14        # content starts just under divider
ROW_H = 10          # tight spacing to fit more items

FOOTER_LINE_Y = 52  # lifted up
FOOTER_Y = 54       # lifted up so text is fully visible


# ============================================================
# INPUT (stdin from parent app.py)
# ============================================================
def read_event():
    """
    Reads one event token from stdin, non-blocking.
    Expected tokens: up, down, select, select_hold, back
    """
    try:
        r, _, _ = select.select([sys.stdin], [], [], 0)
    except Exception:
        return None
    if not r:
        return None
    line = sys.stdin.readline()
    if not line:
        return None
    return line.strip()


# ============================================================
# SETTINGS I/O
# ============================================================
def load_settings():
    if SETTINGS_FILE.exists():
        try:
            data = json.loads(SETTINGS_FILE.read_text())
            out = DEFAULT_SETTINGS.copy()
            out.update(data)
            return out
        except Exception:
            pass
    save_settings(DEFAULT_SETTINGS.copy())
    return DEFAULT_SETTINGS.copy()


def save_settings(s):
    try:
        SETTINGS_FILE.write_text(json.dumps(s, indent=2))
    except Exception as e:
        # Fail silently in production so module keeps running
        # Optional: print(e) for debugging
        pass

# ============================================================
# UI DRAW HELPERS
# ============================================================
def draw_header(d, title: str):
    d.text((2, HEADER_Y), title[:21], fill=255)
    d.line((0, DIVIDER_Y, 127, DIVIDER_Y), fill=255)


def draw_footer(d, text: str):
    d.line((0, FOOTER_LINE_Y, 127, FOOTER_LINE_Y), fill=255)
    d.text((2, FOOTER_Y), text[:21], fill=255)


def draw_row(d, y: int, text: str, selected: bool = False):
    # selection marker on left
    marker = ">" if selected else " "
    d.text((0, y), marker, fill=255)
    d.text((10, y), text[:19], fill=255)


def draw_centered_text(d, y: int, text: str):
    # crude centering
    x = max(0, (OLED_W - (len(text) * 6)) // 2)
    d.text((x, y), text, fill=255)


def render(draw_fn):
    with canvas(device) as d:
        draw_fn(d)


# ============================================================
# RADIO CONTROL (STUB for now)
# ============================================================
def radio_tune(_freq_mhz: float):
    """
    TODO: Implement TEA5767 tune over I2C.
    Keeping this stub so UI works now.
    """
    return


# ============================================================
# SCREEN: MAIN MENU
# ============================================================
def menu_screen(settings, sel_idx: int):
    items = ["START", "SETTINGS"]

    def _draw(d):
        draw_header(d, "SPIRIT BOX")

        # clean content block
        draw_row(d, LIST_Y0 + 0 * ROW_H, items[0], selected=(sel_idx == 0))
        draw_row(d, LIST_Y0 + 1 * ROW_H, items[1], selected=(sel_idx == 1))

        # footer
        draw_footer(d, "SEL choose  BACK exit")

    return items, _draw


# ============================================================
# SCREEN: SETTINGS
# ============================================================
def settings_screen(settings, sel_idx: int):
    """
    Per request:
      - each editable item on its own line
      - no FM band shown
      - no 'Back' as a menu item
    """
    rate = int(settings.get("sweep_ms", 150))
    style = settings.get("scan_style", "LOOP")
    vol = int(settings.get("volume", 80))

    lines = [
        f"Sweep: {rate}ms",
        f"Style: {style}",
        f"Vol:   {vol}%",
    ]

    def _draw(d):
        draw_header(d, "SETTINGS")

        # visible rows = 3 (fits perfectly with footer lifted)
        for i, line in enumerate(lines):
            draw_row(d, LIST_Y0 + i * ROW_H, line, selected=(sel_idx == i))

        draw_footer(d, "SEL edit  BACK menu")

    return lines, _draw


# ============================================================
# EDITORS
# ============================================================
def edit_sweep_rate(settings):
    val = int(settings.get("sweep_ms", 150))
    MIN_MS, MAX_MS, STEP_MS = 50, 350, 50

    while True:
        def _draw(d):
            draw_header(d, "SWEEP RATE")
            draw_centered_text(d, 26, f"{val} ms")
            d.text((2, 40), "UP/DN adjust", fill=255)
            draw_footer(d, "SEL save BACK cancel")

        render(_draw)

        ev = read_event()
        if ev == "up":
            val = min(MAX_MS, val + STEP_MS)
        elif ev == "down":
            val = max(MIN_MS, val - STEP_MS)
        elif ev == "select":
            settings["sweep_ms"] = val
            save_settings(settings)
            return
        elif ev == "back":
            return

        time.sleep(0.03)


def edit_scan_style(settings):
    options = ["LOOP", "BOUNCE", "RANDOM"]
    cur = settings.get("scan_style", "LOOP")
    idx = options.index(cur) if cur in options else 0

    while True:
        def _draw(d):
            draw_header(d, "SCAN STYLE")
            for i, opt in enumerate(options):
                draw_row(d, LIST_Y0 + i * ROW_H, opt, selected=(i == idx))
            draw_footer(d, "SEL save BACK cancel")

        render(_draw)

        ev = read_event()
        if ev == "up":
            idx = (idx - 1) % len(options)
        elif ev == "down":
            idx = (idx + 1) % len(options)
        elif ev == "select":
            settings["scan_style"] = options[idx]
            save_settings(settings)
            return
        elif ev == "back":
            return

        time.sleep(0.03)


def edit_volume(settings):
    val = int(settings.get("volume", 80))
    val = max(0, min(100, val))

    while True:
        def _draw(d):
            draw_header(d, "VOLUME")
            draw_centered_text(d, 26, f"{val}%")
            d.text((2, 40), "UP/DN adjust", fill=255)
            draw_footer(d, "SEL save BACK cancel")

        render(_draw)

        ev = read_event()
        if ev == "up":
            val = min(100, val + 5)
        elif ev == "down":
            val = max(0, val - 5)
        elif ev == "select":
            settings["volume"] = val
            save_settings(settings)
            return
        elif ev == "back":
            return

        time.sleep(0.03)


# ============================================================
# SCREEN: FM SWEEP (freq only)
# ============================================================
def sweep_screen(freq_mhz: float):
    def _draw(d):
        draw_header(d, "FM SWEEP")

        # freq only, big and centered
        draw_centered_text(d, 26, f"{freq_mhz:.1f} MHz")

        # no sweep rate / style displayed (per request)
        draw_footer(d, "BACK stop  HOLD settings")

    return _draw


# ============================================================
# FLOW: SETTINGS
# ============================================================
def settings_flow(settings):
    sel = 0
    while True:
        lines, draw_fn = settings_screen(settings, sel)
        render(draw_fn)

        ev = read_event()
        if ev == "up":
            sel = (sel - 1) % len(lines)
        elif ev == "down":
            sel = (sel + 1) % len(lines)
        elif ev == "back":
            # BACK returns to Spirit Box menu (not exiting module)
            return "MENU"
        elif ev == "select":
            if sel == 0:
                edit_sweep_rate(settings)
            elif sel == 1:
                edit_scan_style(settings)
            elif sel == 2:
                edit_volume(settings)

        time.sleep(0.05)


# ============================================================
# FLOW: SWEEP
# ============================================================
def sweep_flow(settings):
    # fixed full range as requested
    FMIN, FMAX = 76.0, 108.0
    step = float(settings.get("step_mhz", 0.1))

    freq = FMIN
    direction = 1

    while True:
        radio_tune(freq)
        render(sweep_screen(freq))

        ev = read_event()
        if ev == "back":
            # BACK returns to Spirit Box menu (not exiting module)
            return "MENU"
        if ev == "select_hold":
            # quick access to settings; return back to sweep after
            next_state = settings_flow(settings)
            if next_state == "MENU":
                # user backed out of settings -> go back to sweep
                pass

        style = settings.get("scan_style", "LOOP")

        # advance
        if style == "LOOP":
            freq += step
            if freq > FMAX:
                freq = FMIN

        elif style == "BOUNCE":
            freq += step * direction
            if freq >= FMAX:
                freq = FMAX
                direction = -1
            elif freq <= FMIN:
                freq = FMIN
                direction = 1

        else:  # RANDOM
            span = max(0.1, (FMAX - FMIN))
            # deterministic pseudo-random hop (no imports)
            freq = FMIN + ((freq * 13.7 + 1.3) % span)

        delay = max(0.05, int(settings.get("sweep_ms", 150)) / 1000.0)
        time.sleep(delay)


# ============================================================
# MAIN
# ============================================================
def main():
    settings = load_settings()

    state = "MENU"
    menu_sel = 0

    while True:
        if state == "MENU":
            items, draw_fn = menu_screen(settings, menu_sel)
            render(draw_fn)

            ev = read_event()
            if ev == "up":
                menu_sel = (menu_sel - 1) % len(items)
            elif ev == "down":
                menu_sel = (menu_sel + 1) % len(items)
            elif ev == "select":
                state = "SWEEP" if menu_sel == 0 else "SETTINGS"
            elif ev == "back":
                # BACK from initial Spirit Box menu exits module
                def _draw(d):
                    draw_header(d, "SPIRIT BOX")
                    draw_centered_text(d, 28, "Returning...")
                    draw_footer(d, " ")
                render(_draw)
                time.sleep(0.2)
                return


            time.sleep(0.05)

        elif state == "SETTINGS":
            state = settings_flow(settings)

        elif state == "SWEEP":
            state = sweep_flow(settings)

        else:
            state = "MENU"


if __name__ == "__main__":
    main()
