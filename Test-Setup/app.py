#!/usr/bin/env python3
import os
os.environ["GPIOZERO_PIN_FACTORY"] = "lgpio"

import time
import math
import subprocess
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from gpiozero import Button
from luma.core.interface.serial import i2c
from luma.oled.device import ssd1306
from luma.core.render import canvas

# =====================================================
# CONFIG
# =====================================================
I2C_PORT = 1
I2C_ADDR = 0x3C
OLED_W, OLED_H = 128, 64

# Buttons (BCM)
BTN_UP = 17
BTN_DOWN = 27
BTN_SELECT = 22
BTN_BACK = 23

# Hold timings
SELECT_HOLD_SECONDS = 1.0
BACK_REBOOT_HOLD = 2.0
BACK_POWEROFF_HOLD = 5.0

HOME = Path("/home/ghostgeeks01")
APP_DIR = HOME / "oled"
MODULE_DIR = APP_DIR / "modules"
SD_TEST_FILE = APP_DIR / ".sd_write_test"

SPLASH_MIN_SECONDS = 5.0
SPLASH_FRAME_SLEEP = 0.08

# =====================================================
# OLED
# =====================================================
serial = i2c(port=I2C_PORT, address=I2C_ADDR)
device = ssd1306(serial, width=OLED_W, height=OLED_H)

# =====================================================
# UTILITIES
# =====================================================
def sd_write_check() -> Optional[str]:
    """
    Basic check to confirm we can write to the filesystem.
    If this fails, something is seriously wrong (read-only fs, bad mount, etc.)
    """
    try:
        APP_DIR.mkdir(parents=True, exist_ok=True)
        SD_TEST_FILE.write_text("ok\n")
        # Python <3.8 doesn't support missing_ok; you are on newer so it's fine.
        SD_TEST_FILE.unlink(missing_ok=True)
        return None
    except Exception as e:
        return str(e)

def get_ip() -> str:
    try:
        ips = subprocess.check_output(["hostname", "-I"], text=True).split()
        for ip in ips:
            if ip.count(".") == 3 and not ip.startswith("169.254"):
                return ip
        return ""
    except Exception:
        return ""

def hostname() -> str:
    return subprocess.getoutput("hostname").strip()

def uptime_short() -> str:
    return subprocess.getoutput("uptime -p").replace("up ", "").strip()

def oled_message(title: str, lines: List[str], footer: str = ""):
    with canvas(device) as draw:
        draw.text((0, 0), title[:21], fill=255)
        draw.line((0, 12, 127, 12), fill=255)
        y = 16
        for ln in lines[:3]:
            draw.text((0, y), ln[:21], fill=255)
            y += 12
        if footer:
            draw.text((0, 54), footer[:21], fill=255)

def draw_waveform(draw, phase):
    mid = 40
    amp = 10
    pts = []
    for x in range(120):
        t = (x / 120) * (2 * math.pi)
        y = math.sin(t * 2 + phase)
        pts.append((4 + x, int(mid - y * amp)))
    draw.rectangle((2, 26, 125, 53), outline=255)
    draw.line(pts, fill=255)

# =====================================================
# BUTTONS
# =====================================================
def init_buttons():
    events = {
        "up": False,
        "down": False,
        "select": False,
        "select_hold": False,
        "back": False,
    }

    btn_up = Button(BTN_UP, pull_up=True, bounce_time=0.06)
    btn_down = Button(BTN_DOWN, pull_up=True, bounce_time=0.06)

    btn_select = Button(BTN_SELECT, pull_up=True, bounce_time=0.06, hold_time=SELECT_HOLD_SECONDS)
    btn_back = Button(BTN_BACK, pull_up=True, bounce_time=0.06)

    btn_up.when_pressed = lambda: events.__setitem__("up", True)
    btn_down.when_pressed = lambda: events.__setitem__("down", True)

    btn_select.when_pressed = lambda: events.__setitem__("select", True)
    btn_select.when_held = lambda: events.__setitem__("select_hold", True)

    btn_back.when_pressed = lambda: events.__setitem__("back", True)

    def consume(k):
        if events[k]:
            events[k] = False
            return True
        return False

    def clear():
        for k in events:
            events[k] = False

    # IMPORTANT: return button objects so they stay alive
    return consume, clear, (btn_up, btn_down, btn_select, btn_back)

# =====================================================
# MODULES
# =====================================================
@dataclass
class Module:
    id: str
    name: str
    subtitle: str
    entry_path: str
    order: int = 999

def ensure_modules_dir():
    """
    New world: modules live under ~/oled/modules/<module_id> with module.json + run.py
    We only ensure the directory exists; we do NOT auto-generate legacy scripts.
    """
    MODULE_DIR.mkdir(parents=True, exist_ok=True)

def discover_modules(modules_root: Path) -> List[Module]:
    mods: List[Module] = []
    if not modules_root.exists():
        return mods

    for d in sorted(modules_root.iterdir()):
        if not d.is_dir():
            continue

        meta_path = d / "module.json"
        if not meta_path.exists():
            continue

        try:
            meta = json.loads(meta_path.read_text())

            if not meta.get("enabled", True):
                continue

            entry = meta.get("entry", "run.py")
            entry_path = d / entry
            if not entry_path.exists():
                continue

            mods.append(Module(
                id=str(meta.get("id", d.name)),
                name=str(meta.get("name", d.name)),
                subtitle=str(meta.get("subtitle", "")),
                entry_path=str(entry_path),
                order=int(meta.get("order", 999)),
            ))
        except Exception:
            # Ignore bad module entries so one broken module doesn't kill the UI
            continue

    mods.sort(key=lambda m: (m.order, m.name.lower()))
    return mods

# =====================================================
# UI SCREENS
# =====================================================
def splash():
    start = time.time()
    phase = 0.0
    while True:
        with canvas(device) as draw:
            draw.text((0, 2), "GHOST GEEKS", fill=255)
            draw.text((0, 14), "REAL GHOST GEAR", fill=255)
            draw.text((0, 54), "BOOTING UP THE LAB...", fill=255)
            draw_waveform(draw, phase)
        phase += 0.15
        if time.time() - start >= SPLASH_MIN_SECONDS and get_ip():
            return
        time.sleep(SPLASH_FRAME_SLEEP)

def draw_menu(mods: List[Module], idx: int):
    with canvas(device) as draw:
        draw.text((0, 0), "GHOST GEEKS MENU", fill=255)
        draw.line((0, 12, 127, 12), fill=255)

        # Fit as many items as we can
        max_rows = 4  # 16, 28, 40, 52 (but last row conflicts with footer; we use 3 + footer)
        # We'll show up to 3 items plus footer to keep it clean
        visible_rows = 3

        start_i = 0
        if len(mods) > visible_rows:
            # scroll window so the selection stays visible
            start_i = max(0, min(idx - 1, len(mods) - visible_rows))

        for row in range(visible_rows):
            i = start_i + row
            if i >= len(mods):
                break
            prefix = ">" if i == idx else " "
            draw.text((0, 16 + row * 12), f"{prefix} {mods[i].name}"[:21], fill=255)

        draw.text((0, 54), "SEL run  HOLD=cfg", fill=255)

# =====================================================
# POWER CONFIRM
# =====================================================
def confirm_action(label, consume, clear):
    clear()
    end = time.time() + 3
    while True:
        remaining = int(end - time.time()) + 1
        if remaining <= 0:
            return True
        oled_message(label, [f"Confirm in {remaining}s", "Tap BACK to cancel", ""])
        if consume("back"):
            oled_message("CANCELLED", ["Returning...", "", ""])
            time.sleep(0.5)
            clear()
            return False
        time.sleep(0.05)

def reboot():
    oled_message("REBOOT", ["Rebooting...", "", ""])
    subprocess.Popen(["sudo", "-n", "systemctl", "reboot"])

def poweroff():
    oled_message("POWEROFF", ["Shutting down...", "", ""])
    subprocess.Popen(["sudo", "-n", "systemctl", "poweroff"])

# =====================================================
# MODULE RUNNER (stdin-controlled)
# =====================================================
def run_module(mod, consume, clear):
    oled_message("RUNNING", [mod.name, mod.subtitle], "BACK = exit")

    proc = subprocess.Popen(
        mod.cmd,
        stdin=subprocess.PIPE,
        text=True
    )

    while proc.poll() is None:
        if consume("back"):
            # Ask the module to exit cleanly
            try:
                proc.stdin.write("back\n")
                proc.stdin.flush()
            except Exception:
                pass

            # Give it a moment to shut down nicely
            for _ in range(20):  # ~1s total
                if proc.poll() is not None:
                    break
                time.sleep(0.05)

            # If still running, terminate
            if proc.poll() is None:
                proc.terminate()
            break

        time.sleep(0.05)

    # Make sure it's not left behind
    try:
        proc.wait(timeout=1.0)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass

    clear()

    # IMPORTANT: force redraw after returning so OLED doesn't go "dark"
    oled_message("MODULES", ["Select a module", "", ""], "UP/DN + SELECT")
    time.sleep(0.2)


    while proc.poll() is None:
        if consume("up"):
            send("up")
        if consume("down"):
            send("down")
        if consume("select"):
            send("select")
        if consume("select_hold"):
            send("select_hold")

        # BACK should request module exit, then fall back to terminate if needed
        if consume("back"):
            send("back")
            # give it a moment to exit cleanly
            for _ in range(30):  # ~1.5s
                if proc.poll() is not None:
                    break
                time.sleep(0.05)
            if proc.poll() is None:
                proc.terminate()
            break

        time.sleep(0.02)

    clear()


# =====================================================
# SETTINGS
# =====================================================
def settings(consume, clear):
    clear()
    while True:
        oled_message(
            "SETTINGS",
            [hostname(), f"IP {get_ip()}", f"UPTIME {uptime_short()}"],
            "BACK = menu",
        )
        if consume("back"):
            clear()
            return
        time.sleep(0.1)

# =====================================================
# MAIN
# =====================================================
def main():
    if err := sd_write_check():
        oled_message("SD ERROR", [err[:21]], "")
        while True:
            time.sleep(1)

    ensure_modules_dir()
    consume, clear, buttons = init_buttons()
    btn_up, btn_down, btn_select, btn_back = buttons

    splash()

    modules = discover_modules(MODULE_DIR)
    if not modules:
        oled_message("NO MODULES", ["Add modules under", "~/oled/modules", ""], "HOLD=cfg")
        # Still allow settings and power holds
        modules = [Module(id="none", name="(none)", subtitle="", entry_path="/bin/false", order=0)]

    idx = 0
    draw_menu(modules, idx)

    back_pressed_at = None

    while True:
        if consume("up"):
            idx = (idx - 1) % len(modules)
            draw_menu(modules, idx)

        if consume("down"):
            idx = (idx + 1) % len(modules)
            draw_menu(modules, idx)

        if consume("select"):
            # ignore the dummy item
            if modules[idx].id != "none":
                run_module(modules[idx], consume, clear)
            draw_menu(modules, idx)

        if consume("select_hold"):
            settings(consume, clear)
            draw_menu(modules, idx)

        # BACK holds
        if btn_back.is_pressed:
            if back_pressed_at is None:
                back_pressed_at = time.time()
            held = time.time() - back_pressed_at

            if held >= BACK_POWEROFF_HOLD:
                if confirm_action("POWEROFF?", consume, clear):
                    poweroff()
                    return
                back_pressed_at = None

            elif held >= BACK_REBOOT_HOLD:
                if confirm_action("REBOOT?", consume, clear):
                    reboot()
                    return
                back_pressed_at = None
        else:
            back_pressed_at = None

        time.sleep(0.02)

if __name__ == "__main__":
    main()
