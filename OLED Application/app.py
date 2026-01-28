#!/usr/bin/env python3
"""
Ghost Geeks OLED UI - app.py (clean replacement)

Fixes:
- Always refresh OLED after a module exits (prevents “blank menu” until next button press)
- Run modules as child processes and forward button events via stdin
- Capture each module's stdout/stderr to /home/ghostgeeks01/oled/logs/<module>_YYYYMMDD_HHMMSS.log
- More robust process cleanup (terminate -> kill fallback)
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

from gpiozero import Button
from luma.core.interface.serial import i2c
from luma.core.render import canvas
from luma.oled.device import ssd1306

# =====================================================
# CONFIG
# =====================================================

OLED_ADDR = 0x3C
OLED_W, OLED_H = 128, 64

MODULE_DIR = Path("/home/ghostgeeks01/oled/modules")
LOG_DIR = Path("/home/ghostgeeks01/oled/logs")

# Buttons (BCM)
BTN_UP = 17
BTN_DOWN = 27
BTN_SELECT = 22
BTN_BACK = 23
SELECT_HOLD_SECONDS = 0.8

# UI metrics
HEADER_H = 12
FOOTER_Y = 54  # lift the footer a bit so it isn't clipped
LINE_H = 12

# =====================================================
# OLED SETUP
# =====================================================

serial = i2c(port=1, address=OLED_ADDR)
device = ssd1306(serial, width=OLED_W, height=OLED_H)


def _hr_line(d, y: int):
    d.line((0, y, OLED_W - 1, y), fill=255)


def oled_message(title: str, lines: List[str], footer: str = ""):
    """Simple 3-line screen with header + footer."""
    with canvas(device) as d:
        d.text((0, 0), title[:21], fill=255)
        _hr_line(d, HEADER_H - 1)

        y = HEADER_H
        for s in lines[:3]:
            d.text((0, y), (s or "")[:21], fill=255)
            y += LINE_H

        if footer:
            _hr_line(d, FOOTER_Y - 2)
            d.text((0, FOOTER_Y), footer[:21], fill=255)


def oled_clear():
    device.clear()
    device.show()


def force_menu_redraw(mods: List["Module"], sel: int):
    """
    Force a hard refresh of the menu (prevents blank OLED after module exit).
    Some child modules leave the OLED cleared; this guarantees we redraw.
    """
    try:
        device.clear()
        device.show()
    except Exception:
        pass
    time.sleep(0.05)
    draw_menu(mods, sel)


# =====================================================
# MODULE DISCOVERY
# =====================================================

@dataclass
class Module:
    id: str
    name: str
    subtitle: str
    entry_path: str
    order: int = 999


def discover_modules(module_dir: Path) -> List[Module]:
    mods: List[Module] = []
    if not module_dir.exists():
        return mods

    for d in sorted(module_dir.iterdir()):
        if not d.is_dir():
            continue

        entry = d / "run.py"
        meta_path = d / "module.json"
        if not entry.exists():
            continue

        meta = {}
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
            except Exception:
                meta = {}

        mods.append(
            Module(
                id=str(meta.get("id", d.name)),
                name=str(meta.get("name", d.name)),
                subtitle=str(meta.get("subtitle", "")),
                entry_path=str(entry),
                order=int(meta.get("order", 999)),
            )
        )

    mods.sort(key=lambda m: (m.order, m.name.lower()))
    return mods


# =====================================================
# BUTTONS
# =====================================================

def init_buttons():
    events = {"up": False, "down": False, "select": False, "select_hold": False, "back": False}

    btn_up = Button(BTN_UP, pull_up=True, bounce_time=0.06)
    btn_down = Button(BTN_DOWN, pull_up=True, bounce_time=0.06)

    btn_select = Button(
        BTN_SELECT, pull_up=True, bounce_time=0.06, hold_time=SELECT_HOLD_SECONDS
    )
    btn_back = Button(BTN_BACK, pull_up=True, bounce_time=0.06)

    btn_up.when_pressed = lambda: events.__setitem__("up", True)
    btn_down.when_pressed = lambda: events.__setitem__("down", True)

    btn_select.when_pressed = lambda: events.__setitem__("select", True)
    btn_select.when_held = lambda: events.__setitem__("select_hold", True)

    btn_back.when_pressed = lambda: events.__setitem__("back", True)

    def consume(k: str) -> bool:
        if events[k]:
            events[k] = False
            return True
        return False

    def clear():
        for k in events:
            events[k] = False

    # return button objects so they stay alive
    return consume, clear, (btn_up, btn_down, btn_select, btn_back)


# =====================================================
# MENU UI
# =====================================================

def draw_menu(mods: List[Module], sel: int):
    with canvas(device) as d:
        d.text((0, 0), "MODULES", fill=255)
        _hr_line(d, HEADER_H - 1)

        # Display 4 rows max
        rows = 4
        start = max(0, min(sel - 1, max(0, len(mods) - rows)))
        y = HEADER_H

        for i in range(start, min(start + rows, len(mods))):
            m = mods[i]
            prefix = ">" if i == sel else " "
            label = f"{prefix} {m.name}"
            d.text((0, y), label[:21], fill=255)
            y += LINE_H

        _hr_line(d, FOOTER_Y - 2)
        d.text((0, FOOTER_Y), "UP/DN sel  SEL run  BK quit"[:21], fill=255)


# =====================================================
# MODULE RUNNER
# =====================================================

def _open_logfile(module_id: str):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = LOG_DIR / f"{module_id}_{ts}.log"
    return path, open(path, "w", buffering=1)


def run_module(mod: Module, consume, clear, mods: List[Module], sel_index: int):
    """
    Runs a module as a child process and forwards button events to it via stdin.

    Expected stdin commands in the module:
      up, down, select, select_hold, back
    """
    oled_message("RUNNING", [mod.name, mod.subtitle], "BACK = exit")

    cmd = ["/home/ghostgeeks01/oledenv/bin/python", mod.entry_path]

    log_path, logf = _open_logfile(mod.id)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    # Allow modules to import shared helpers from /home/ghostgeeks01/oled
    env.setdefault("PYTHONPATH", "/home/ghostgeeks01/oled")
    # Keep gpio backend consistent if a module uses gpiozero (most should NOT)
    env.setdefault("GPIOZERO_PIN_FACTORY", "lgpio")

    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=logf,
            stderr=logf,
            text=True,
            bufsize=1,
            env=env,
        )
    except Exception as e:
        logf.write(f"[launcher] FAILED: {e}\n")
        logf.close()
        oled_message("LAUNCH FAIL", [mod.name, str(e)[:21], str(log_path.name)], "BACK = menu")
        time.sleep(1.5)
        clear()
        force_menu_redraw(mods, sel_index)
        return

    def send(cmd_text: str):
        try:
            if proc.poll() is None and proc.stdin:
                proc.stdin.write(cmd_text + "\n")
                proc.stdin.flush()
        except Exception:
            pass

    # Main loop while module runs
    while proc.poll() is None:
        if consume("up"):
            send("up")
        if consume("down"):
            send("down")
        if consume("select"):
            send("select")
        if consume("select_hold"):
            send("select_hold")

        if consume("back"):
            # ask module to exit
            send("back")
            # wait briefly for clean shutdown
            for _ in range(30):
                if proc.poll() is not None:
                    break
                time.sleep(0.05)
            if proc.poll() is None:
                try:
                    proc.terminate()
                except Exception:
                    pass
            break

        time.sleep(0.02)

    # Cleanup
    try:
        if proc.stdin:
            proc.stdin.close()
    except Exception:
        pass

    try:
        proc.wait(timeout=1.0)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass

    try:
        logf.write(f"[launcher] exit_code={proc.returncode}\n")
        logf.flush()
        logf.close()
    except Exception:
        pass

    clear()

    # Critical: hard redraw menu after module exit (fixes blank OLED)
    force_menu_redraw(mods, sel_index)


# =====================================================
# MAIN
# =====================================================

def main():
    mods = discover_modules(MODULE_DIR)
    if not mods:
        oled_message("NO MODULES", ["No run.py found", str(MODULE_DIR)], "BACK = quit")
        consume, clear, _btns = init_buttons()
        while True:
            if consume("back"):
                break
            time.sleep(0.05)
        oled_clear()
        return

    consume, clear, _btns = init_buttons()
    sel = 0

    force_menu_redraw(mods, sel)

    last_refresh = time.time()

    while True:
        # Periodic refresh safety-net (prevents “stale/blank” if anything weird happens)
        if time.time() - last_refresh > 1.0:
            draw_menu(mods, sel)
            last_refresh = time.time()

        moved = False

        if consume("up"):
            sel = (sel - 1) % len(mods)
            moved = True

        if consume("down"):
            sel = (sel + 1) % len(mods)
            moved = True

        if consume("select"):
            clear()
            run_module(mods[sel], consume, clear, mods, sel)
            # run_module already forced redraw
            last_refresh = time.time()
            continue

        if moved:
            draw_menu(mods, sel)
            last_refresh = time.time()

        if consume("back"):
            oled_message("GOODBYE", ["Ghost Geeks"], "",)
            time.sleep(0.4)
            oled_clear()
            break

        time.sleep(0.02)


if __name__ == "__main__":
    main()
