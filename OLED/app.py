#!/usr/bin/env python3
import os
os.environ["GPIOZERO_PIN_FACTORY"] = "lgpio"

import sys
import time
import math
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
from datetime import datetime

from gpiozero import Button
from luma.core.interface.serial import i2c
from luma.oled.device import ssd1306
from luma.core.render import canvas


# =====================================================
# PATHS (no hardcoded /home/*)
# =====================================================
def _resolve_root() -> Path:
    """
    Resolve the project root.
    Priority:
      1) BLACKBOX_ROOT env var (explicit override)
      2) folder above this file (…/OLED/app.py -> root is …/)
    """
    env = os.environ.get("BLACKBOX_ROOT", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return Path(__file__).resolve().parents[1]


ROOT_DIR = _resolve_root()

# Detect modules dir with flexible casing
MODULE_DIR_CANDIDATES = [
    ROOT_DIR / "OLED" / "Modules",
    ROOT_DIR / "OLED" / "modules",
    ROOT_DIR / "Modules",
    ROOT_DIR / "modules",
]

MODULE_DIR = next((p for p in MODULE_DIR_CANDIDATES if p.exists()), ROOT_DIR / "OLED" / "modules")

# Logs/data kept inside install tree by default (works under /opt/blackbox with correct perms)
DATA_DIR = Path(os.environ.get("BLACKBOX_DATA", str(ROOT_DIR / "data"))).expanduser().resolve()
LOG_DIR = DATA_DIR / "logs"
SD_TEST_FILE = DATA_DIR / ".sd_write_test"


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

SPLASH_MIN_SECONDS = 5.0
SPLASH_FRAME_SLEEP = 0.08

# Menu refresh watchdog (helps recover from rare "blank menu" states)
MENU_REFRESH_SECONDS = 2.0

# Device Branding Reference
BRAND_NAME = "BLACKBOX"


# =====================================================
# OLED (re-init safe)
# =====================================================
_serial = None
device = None


def oled_init() -> None:
    """(Re)initialize the OLED device object."""
    global _serial, device
    _serial = i2c(port=I2C_PORT, address=I2C_ADDR)
    device = ssd1306(_serial, width=OLED_W, height=OLED_H)


def oled_hard_wake() -> None:
    """
    Some child processes may leave the SSD1306 in DISPLAYOFF (or I2C in a weird state).
    Re-init fixes it reliably.
    """
    global device
    try:
        oled_init()
    except Exception:
        time.sleep(0.05)
        oled_init()


def oled_guard() -> None:
    global device
    if device is None:
        oled_hard_wake()


# Initialize once
oled_init()


# =====================================================
# UTILITIES
# =====================================================
def sd_write_check() -> Optional[str]:
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)

        # Errno 21 guard: if a directory exists at the test-file path, remove it.
        if SD_TEST_FILE.exists() and SD_TEST_FILE.is_dir():
            return f"{SD_TEST_FILE} is a directory (Errno 21). Remove it."

        SD_TEST_FILE.write_text("ok\n")
        try:
            SD_TEST_FILE.unlink()
        except Exception:
            pass
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
    try:
        return subprocess.getoutput("hostname").strip()
    except Exception:
        return ""


def uptime_short() -> str:
    try:
        return subprocess.getoutput("uptime -p").replace("up ", "").strip()
    except Exception:
        return ""


def ensure_dirs() -> None:
    MODULE_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def log_path_for(module_id: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe = "".join([c if (c.isalnum() or c in "_-") else "_" for c in module_id])[:40]
    return LOG_DIR / f"{safe}_{ts}.log"


# =====================================================
# OLED DRAW HELPERS
# =====================================================
def oled_message(title: str, lines: List[str], footer: str = "") -> None:
    oled_guard()
    with canvas(device) as draw:
        draw.text((0, 0), title[:21], fill=255)
        draw.line((0, 12, 127, 12), fill=255)
        y = 16
        for ln in lines[:3]:
            draw.text((0, y), ln[:21], fill=255)
            y += 12
        if footer:
            draw.text((0, 52), footer[:21], fill=255)


def draw_waveform(draw, phase: float) -> None:
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

    btn_select = Button(
        BTN_SELECT,
        pull_up=True,
        bounce_time=0.06,
        hold_time=SELECT_HOLD_SECONDS,
    )
    btn_back = Button(BTN_BACK, pull_up=True, bounce_time=0.06)

    btn_up.when_pressed = lambda: events.__setitem__("up", True)
    btn_down.when_pressed = lambda: events.__setitem__("down", True)

    btn_select.when_pressed = lambda: events.__setitem__("select", True)
    btn_select.when_held = lambda: events.__setitem__("select_hold", True)

    btn_back.when_pressed = lambda: events.__setitem__("back", True)

    def consume(k: str) -> bool:
        if events.get(k):
            events[k] = False
            return True
        return False

    def clear() -> None:
        for k in events:
            events[k] = False

    return consume, clear, (btn_up, btn_down, btn_select, btn_back)


def drain_events(consume, seconds: float = 0.25) -> None:
    """Drain queued button events to prevent ghost presses."""
    end = time.time() + seconds
    while time.time() < end:
        consume("up")
        consume("down")
        consume("select")
        consume("select_hold")
        consume("back")
        time.sleep(0.01)


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

            mods.append(
                Module(
                    id=str(meta.get("id", d.name)),
                    name=str(meta.get("name", d.name)),
                    subtitle=str(meta.get("subtitle", "")),
                    entry_path=str(entry_path),
                    order=int(meta.get("order", 999)),
                )
            )
        except Exception:
            continue

    mods.sort(key=lambda m: (m.order, m.name.lower()))
    return mods


# =====================================================
# RADIO CONNECTIONS (Bluetooth autoconnect)
# =====================================================
CONNECTIONS_FILE = DATA_DIR / "connections.json"


def load_connections() -> dict:
    """
    Load /opt/blackbox/data/connections.json (or BLACKBOX_DATA override).
    Returns {} if missing or invalid.
    """
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        if not CONNECTIONS_FILE.exists():
            return {}
        return json.loads(CONNECTIONS_FILE.read_text())
    except Exception:
        return {}


def bluetooth_is_connected(mac: str) -> bool:
    try:
        r = subprocess.run(["bluetoothctl", "info", mac], capture_output=True, text=True, timeout=3)
        return "Connected: yes" in (r.stdout or "")
    except Exception:
        return False


def bluetooth_connect(mac: str, timeout: float = 8.0) -> bool:
    """
    Attempt to connect to a known/trusted device.
    Returns True if connected, False otherwise.
    Never blocks longer than timeout.
    """
    # Safe: ensures BT isn't soft-blocked
    try:
        subprocess.run(["rfkill", "unblock", "bluetooth"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=2)
    except Exception:
        pass

    # If already connected, we're done
    if bluetooth_is_connected(mac):
        return True

    start = time.time()
    while (time.time() - start) < timeout:
        try:
            subprocess.run(["bluetoothctl", "connect", mac], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=6)
        except Exception:
            pass

        if bluetooth_is_connected(mac):
            return True

        time.sleep(1.5)

    return False


def bluetooth_autoconnect_ui() -> bool:
    """
    Reads connections.json and attempts BT connect if configured.
    Shows a short status message on OLED.
    Returns True if connected, else False (or no config).
    """
    cfg = load_connections()
    bt = (cfg or {}).get("bluetooth", {}) if isinstance(cfg, dict) else {}
    mac = (bt.get("mac") or "").strip()
    autoconnect = bool(bt.get("autoconnect", False))
    global _status_bt_mac
    _status_bt_mac = mac


    if not mac or not autoconnect:
        return False

    oled_message("CONNECTING BT", [mac, "Please wait...", ""], "BACK = skip")

    # Try a short connect window
    ok = bluetooth_connect(mac, timeout=8.0)

    if ok:
        oled_message("BT CONNECTED", [mac, "", ""], "")
        time.sleep(0.6)
        return True

    oled_message("BT NOT CONNECTED", [mac, "Continuing...", ""], "")
    time.sleep(0.6)
    return False

# =====================================================
# STATUS INDICATORS (Wi-Fi bars + Bluetooth icon)
# =====================================================

_last_status_check = 0.0
_wifi_bars = 0           # 0..3
_bt_ok = False
_status_bt_mac = ""


def wifi_rssi_dbm(interface: str = "wlan0") -> Optional[int]:
    """
    Returns RSSI in dBm (e.g., -52) or None if not connected/unknown.
    Uses: iw dev wlan0 link
    """
    try:
        r = subprocess.run(
            ["iw", "dev", interface, "link"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        out = (r.stdout or "").splitlines()
        # If not connected, output often contains "Not connected."
        if any("Not connected" in ln for ln in out):
            return None

        for ln in out:
            ln = ln.strip().lower()
            # Example: "signal: -52 dBm"
            if ln.startswith("signal:"):
                parts = ln.replace("dbm", "").split()
                # parts like ["signal:", "-52"]
                for p in parts:
                    if p.lstrip("-").isdigit():
                        return int(p)
        return None
    except Exception:
        return None


def wifi_bars_from_rssi(rssi: Optional[int]) -> int:
    """
    Map RSSI (dBm) to 0..3 bars.
    Typical guidance:
      >= -55: excellent (3)
      >= -67: good      (2)
      >= -80: fair      (1)
      else:   weak/off  (0)
    """
    if rssi is None:
        return 0
    if rssi >= -55:
        return 3
    if rssi >= -67:
        return 2
    if rssi >= -80:
        return 1
    return 0


def draw_wifi_bars(draw, x_right: int, y_top: int, bars: int) -> int:
    """
    Draw 0..3 Wi-Fi bars ending at x_right.
    Returns the left-most x used (so caller can place other icons to the left).
    """
    # Bar geometry
    w = 2
    gap = 1
    heights = [3, 6, 9]  # 3 bars
    total_w = 3 * w + 2 * gap
    x0 = x_right - total_w

    # Draw outline bars (unfilled)
    for i, h in enumerate(heights):
        x = x0 + i * (w + gap)
        y0 = y_top + (9 - h)
        draw.rectangle((x, y0, x + w - 1, y_top + 9), outline=255, fill=0)

    # Fill bars up to 'bars'
    for i in range(min(3, max(0, bars))):
        h = heights[i]
        x = x0 + i * (w + gap)
        y0 = y_top + (9 - h)
        draw.rectangle((x, y0, x + w - 1, y_top + 9), outline=255, fill=255)

    return x0


def draw_bt_icon(draw, x_right: int, y_top: int, connected: bool) -> int:
    """
    Draw a tiny Bluetooth indicator at the top-right area.
    We'll do 'B' plus a dot that is filled when connected.
    Returns the left-most x used.
    """
    # Small "B"
    text = "B"
    # Put text so its right edge is at x_right
    x_text = max(0, x_right - 6)  # ~6px per char
    draw.text((x_text, y_top), text, fill=255)

    # Dot to the left of the B
    dot_x = x_text - 5
    dot_y = y_top + 3
    if connected:
        draw.ellipse((dot_x, dot_y, dot_x + 3, dot_y + 3), outline=255, fill=255)
    else:
        draw.ellipse((dot_x, dot_y, dot_x + 3, dot_y + 3), outline=255, fill=0)

    return dot_x


def status_refresh(force: bool = False) -> None:
    """
    Refresh Wi-Fi + BT status at most every ~2 seconds (unless forced).
    """
    global _last_status_check, _wifi_bars, _bt_ok

    now = time.time()
    if (not force) and (now - _last_status_check) < 2.0:
        return
    _last_status_check = now

    # Wi-Fi bars: use RSSI if possible; fall back to "has IP" => 1 bar
    rssi = wifi_rssi_dbm("wlan0")
    if rssi is None:
        _wifi_bars = 1 if get_ip() else 0
    else:
        _wifi_bars = wifi_bars_from_rssi(rssi)

    # BT connected?
    if _status_bt_mac:
        _bt_ok = bluetooth_is_connected(_status_bt_mac)
    else:
        _bt_ok = False

# =====================================================
# UI SCREENS
# =====================================================
def splash() -> None:
    start = time.time()
    phase = 0.0
    while True:
        oled_guard()
        with canvas(device) as draw:
            draw.text((0, 2), "GHOST GEEKS", fill=255)
            draw.text((0, 14), "REAL GHOST GEAR", fill=255)
            draw.text((0, 52), "BOOTING UP...", fill=255)
            draw_waveform(draw, phase)

        phase += 0.15
        if (time.time() - start) >= SPLASH_MIN_SECONDS and get_ip():
            return
        time.sleep(SPLASH_FRAME_SLEEP)


def draw_menu(mods: List[Module], idx: int) -> None:
    status_refresh(force=False)

    oled_guard()
    with canvas(device) as draw:
        # Header left
        draw.text((0, 0), f"{BRAND_NAME} MENU", fill=255)

        # Header right: Bluetooth + Wi-Fi bars
        # Start from far-right and draw BT, then Wi-Fi to its left
        x = OLED_W - 1
        x = draw_bt_icon(draw, x_right=x, y_top=0, connected=_bt_ok) - 3
        _ = draw_wifi_bars(draw, x_right=x, y_top=1, bars=_wifi_bars)

        draw.line((0, 12, 127, 12), fill=255)

        visible_rows = 3
        start_i = 0
        if len(mods) > visible_rows:
            start_i = max(0, min(idx - 1, len(mods) - visible_rows))

        for row in range(visible_rows):
            i = start_i + row
            if i >= len(mods):
                break
            prefix = ">" if i == idx else " "
            draw.text((0, 16 + row * 12), f"{prefix} {mods[i].name}"[:21], fill=255)

        draw.text((0, 52), "SEL=run  HOLD=cfg", fill=255)


# =====================================================
# POWER CONFIRM
# =====================================================
def confirm_action(label: str, consume, clear) -> bool:
    clear()
    end = time.time() + 3
    while True:
        remaining = int(end - time.time()) + 1
        if remaining <= 0:
            return True
        oled_message(label, [f"Confirm in {remaining}s", "Tap BACK to cancel", ""], "")
        if consume("back"):
            oled_message("CANCELLED", ["Returning...", "", ""], "")
            time.sleep(0.5)
            clear()
            return False
        time.sleep(0.05)


def reboot() -> None:
    oled_message("REBOOT", ["Rebooting...", "", ""], "")
    subprocess.Popen(["sudo", "-n", "systemctl", "reboot"])


def poweroff() -> None:
    oled_message("POWEROFF", ["Shutting down...", "", ""], "")
    subprocess.Popen(["sudo", "-n", "systemctl", "poweroff"])


# =====================================================
# SETTINGS
# =====================================================
def settings(consume, clear) -> None:
    clear()
    while True:
        oled_message(
            "SETTINGS",
            [hostname(), f"IP {get_ip()}", f"UP {uptime_short()}"],
            "BACK = menu",
        )
        if consume("back"):
            clear()
            return
        time.sleep(0.1)


# =====================================================
# MODULE RUNNER
# =====================================================
def run_module(mod: Module, consume, clear) -> None:
    """
    Runs a module as a child process and forwards button events to it via stdin.
    Expected stdin commands in module:
      up, down, select, select_hold, back
    """
    ensure_dirs()

    # CRITICAL: clear/drain BEFORE launch so stale BACK never gets forwarded.
    clear()
    drain_events(consume, seconds=0.20)

    oled_message("RUNNING", [mod.name, mod.subtitle], "BACK = exit")

    # Use the current interpreter (systemd already launches app.py inside the venv)
    cmd = [sys.executable, mod.entry_path]

    log_path = log_path_for(mod.id)
    try:
        logf = open(log_path, "w", buffering=1)
    except Exception:
        logf = None

    def log(line: str) -> None:
        try:
            if logf:
                logf.write(line.rstrip() + "\n")
        except Exception:
            pass

    log(f"[launcher] cmd={cmd!r}")

    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=logf if logf else subprocess.DEVNULL,
            stderr=logf if logf else subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
    except Exception as e:
        if logf:
            log(f"[launcher] failed_to_start: {e!r}")
            try:
                logf.close()
            except Exception:
                pass
        oled_message("LAUNCH FAIL", [mod.name, str(e)[:21], ""], "BACK = menu")
        time.sleep(1.2)
        clear()
        oled_hard_wake()
        return

    def send(cmd_text: str) -> None:
        try:
            if proc.poll() is None and proc.stdin:
                proc.stdin.write(cmd_text + "\n")
                proc.stdin.flush()
        except Exception:
            pass

    # Run loop while module is alive
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
            send("back")
            # give it a moment to exit cleanly
            for _ in range(40):
                if proc.poll() is not None:
                    break
                time.sleep(0.02)
            if proc.poll() is None:
                try:
                    proc.terminate()
                except Exception:
                    pass
            break

        time.sleep(0.02)

    # cleanup stdin
    try:
        if proc.stdin:
            proc.stdin.close()
    except Exception:
        pass

    # wait a bit; force kill if needed
    try:
        proc.wait(timeout=1.0)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass

    exit_code = proc.returncode
    log(f"[launcher] exit_code={exit_code}")

    if logf:
        try:
            logf.close()
        except Exception:
            pass

    # Clear and drain AGAIN after return
    clear()
    drain_events(consume, seconds=0.30)

    # Child may have left OLED off; recover here
    oled_hard_wake()


# =====================================================
# MAIN
# =====================================================
def main() -> None:
    if (err := sd_write_check()) is not None:
        oled_message("SD ERROR", [err[:21], "", ""], "")
        while True:
            time.sleep(1)

    ensure_dirs()
    consume, clear, buttons = init_buttons()
    btn_up, btn_down, btn_select, btn_back = buttons

    splash()

    status_refresh(force=True)

    # Attempt BT autoconnect AFTER splash, BEFORE menu
    # (quick, timeout-limited, continues either way)
    clear()
    drain_events(consume, seconds=0.10)
    bluetooth_autoconnect_ui()
    clear()
    drain_events(consume, seconds=0.10)

    status_refresh(force=True)

    modules = discover_modules(MODULE_DIR)
    if not modules:
        oled_message("NO MODULES", [str(MODULE_DIR)[:21], "Add module folders", ""], "HOLD=cfg")
        modules = [Module(id="none", name="(none)", subtitle="", entry_path="/bin/false", order=0)]

    idx = 0
    draw_menu(modules, idx)
    last_menu_draw = time.time()

    back_pressed_at = None

    while True:
        # watchdog refresh for "blank menu" recovery
        if time.time() - last_menu_draw >= MENU_REFRESH_SECONDS:
            draw_menu(modules, idx)
            last_menu_draw = time.time()

        if consume("up"):
            idx = (idx - 1) % len(modules)
            draw_menu(modules, idx)
            last_menu_draw = time.time()

        if consume("down"):
            idx = (idx + 1) % len(modules)
            draw_menu(modules, idx)
            last_menu_draw = time.time()

        if consume("select"):
            # prevent the SELECT press from being forwarded accidentally in weird timings
            clear()
            drain_events(consume, seconds=0.10)

            if modules[idx].id != "none":
                run_module(modules[idx], consume, clear)

            # Always redraw menu immediately after returning
            draw_menu(modules, idx)
            last_menu_draw = time.time()

        if consume("select_hold"):
            settings(consume, clear)
            draw_menu(modules, idx)
            last_menu_draw = time.time()

        # BACK holds for reboot/poweroff
        if btn_back.is_pressed:
            if back_pressed_at is None:
                back_pressed_at = time.time()
            held = time.time() - back_pressed_at

            if held >= BACK_POWEROFF_HOLD:
                if confirm_action("POWEROFF?", consume, clear):
                    poweroff()
                    return
                back_pressed_at = None
                draw_menu(modules, idx)
                last_menu_draw = time.time()

            elif held >= BACK_REBOOT_HOLD:
                if confirm_action("REBOOT?", consume, clear):
                    reboot()
                    return
                back_pressed_at = None
                draw_menu(modules, idx)
                last_menu_draw = time.time()
        else:
            back_pressed_at = None

        # CRITICAL FIX:
        # Always consume/clear any BACK press event in the menu loop.
        consume("back")

        time.sleep(0.02)


if __name__ == "__main__":
    main()
