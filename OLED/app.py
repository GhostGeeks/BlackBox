#!/usr/bin/env python3
import os
os.environ["GPIOZERO_PIN_FACTORY"] = "lgpio"

import sys
import time
import math
import json
import selectors
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
BRAND_NAME = "CHOST GEEKS"

# =====================================================
# BRANDING
# =====================================================
PRODUCT_NAME = "BLACKBOX"
PRODUCT_SUBTITLE = "PARANORMAL AUDIO"
TAGLINE = "FIELD UNIT"
VERSION = "v0.9"  # bump as you like

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

    oled_message(f"{PRODUCT_NAME} AUDIO", [mac, "Connecting...", ""], "BACK = skip")

    # Try a short connect window
    ok = bluetooth_connect(mac, timeout=8.0)

    if ok:
        oled_message("AUDIO READY", [mac, "", ""], "")
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
    """
    Instrument-style boot screen:
      - shows product name + subtitle
      - shows version + minimal status line
      - waits SPLASH_MIN_SECONDS and for IP (current behavior)
    """
    start = time.time()
    phase = 0.0

    while True:
        ip = get_ip()
        net = "NET OK" if ip else "NET..."
        bt = "BT OK" if (_status_bt_mac and bluetooth_is_connected(_status_bt_mac)) else "BT..."

        oled_guard()
        with canvas(device) as draw:
            # Top: product name
            draw.text((0, 0), PRODUCT_NAME[:21], fill=255)

            # Right: version
            draw.text((OLED_W - (len(VERSION) * 6), 0), VERSION, fill=255)

            draw.line((0, 12, 127, 12), fill=255)

            # Middle: subtitle + tagline
            draw.text((0, 16), PRODUCT_SUBTITLE[:21], fill=255)
            draw.text((0, 28), TAGLINE[:21], fill=255)

            # Bottom: status line
            draw.line((0, 44, 127, 44), fill=255)
            draw.text((0, 48), f"{net}  {bt}"[:21], fill=255)

            # Tiny waveform “alive” indicator (subtle)
            draw_waveform(draw, phase)

        phase += 0.15

        # Keep your existing behavior: minimum splash time + wait for IP
        if (time.time() - start) >= SPLASH_MIN_SECONDS and ip:
            return

        time.sleep(SPLASH_FRAME_SLEEP)

def startup_sequence(consume, clear) -> None:
    """
    Instrument-style boot sequence with real checks.
    Keeps it short and purposeful.
    """
    clear()
    drain_events(consume, seconds=0.10)

    # Step 1: BOOT
    t0 = time.time()
    while time.time() - t0 < 0.9:
        oled_guard()
        with canvas(device) as draw:
            draw.text((0, 0), PRODUCT_NAME[:21], fill=255)
            draw.text((OLED_W - (len(VERSION) * 6), 0), VERSION, fill=255)
            draw.line((0, 12, 127, 12), fill=255)
            draw.text((0, 18), "INITIALIZING...", fill=255)
            draw_progress(draw, (time.time() - t0) / 0.9)
        time.sleep(0.05)

    # Step 2: SELF TEST (do real checks)
    sd_err = sd_write_check()
    i2c_ok = True
    try:
        oled_hard_wake()
    except Exception:
        i2c_ok = False

    oled_message(
        "SELF TEST",
        [
            f"OLED {'OK' if i2c_ok else 'FAIL'}",
            f"SD  {'OK' if sd_err is None else 'FAIL'}",
            "BTN READY",
        ],
        "",
    )
    time.sleep(1.2)

    # Step 3: RADIOS (wait briefly for Wi-Fi + attempt BT)
    # Wi-Fi: give it a short window; we no longer need the old "splash wait for IP" behavior
    t1 = time.time()
    ip = ""
    while time.time() - t1 < 2.5:
        ip = get_ip()
        oled_guard()
        with canvas(device) as draw:
            draw.text((0, 0), "RADIOS", fill=255)
            draw.line((0, 12, 127, 12), fill=255)
            draw.text((0, 18), f"WIFI {'OK' if ip else '...'}", fill=255)
            draw.text((0, 30), "BT   ...", fill=255)
            draw_progress(draw, min(1.0, (time.time() - t1) / 2.5))
        if ip:
            break
        time.sleep(0.1)

    # BT autoconnect with UI (already shows its own screen briefly)
    bt_ok = bluetooth_autoconnect_ui()

    # Step 4: READY
    oled_message(
        "READY",
        [
            f"WIFI {'OK' if get_ip() else 'OFF'}",
            f"AUDIO {'BT' if bt_ok else 'LOCAL'}",
            "",
        ],
        "",
    )
    time.sleep(0.7)

    clear()
    drain_events(consume, seconds=0.10)


def draw_menu(mods: List[Module], idx: int) -> None:
    status_refresh(force=False)

    oled_guard()
    with canvas(device) as draw:
        # Header left
        draw.text((0, 0), f"{PRODUCT_NAME} MENU"[:21], fill=255)

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


def draw_progress(draw, pct: float) -> None:
    # pct 0.0..1.0
    x0, y0, x1, y1 = 8, 54, 120, 62
    draw.rectangle((x0, y0, x1, y1), outline=255, fill=0)
    w = int((x1 - x0 - 2) * max(0.0, min(1.0, pct)))
    draw.rectangle((x0 + 1, y0 + 1, x0 + 1 + w, y1 - 1), outline=255, fill=255)



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
        status_refresh(force=True)
        ip = get_ip() or "none"
        up = uptime_short() or "unknown"
        host = hostname() or "blackbox"

        bt_state = "OK" if _bt_ok else "none"
        wf = f"{_wifi_bars}/3"

        oled_message(
            f"{PRODUCT_NAME} STATUS",
            [f"HOST {host}"[:21], f"IP {ip}"[:21], f"WIFI {wf}  BT {bt_state}"[:21]],
            "BACK = menu",
        )

        if consume("back"):
            clear()
            return
        time.sleep(0.15)

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

    clear()
    drain_events(consume, seconds=0.20)

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

    is_uap = (mod.id == "uap_caller")

    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE if is_uap else (logf if logf else subprocess.DEVNULL),
            stderr=logf if logf else subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
    except Exception as e:
        log(f"[launcher] failed_to_start: {e!r}")
        if logf:
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

    # -----------------------------
    # UAP Caller: app owns OLED; module emits JSON status on stdout
    # -----------------------------
    if is_uap:
        out_sel = selectors.DefaultSelector()
        if proc.stdout:
            out_sel.register(proc.stdout, selectors.EVENT_READ)

        state = {
            "page": "build",
            "build_pct": 0.0,
            "build_step": "",
            "playing": False,
            "elapsed_s": 0,
        }

        def draw_build() -> None:
            oled_message(
                "UAP Call Sig",
                [state["build_step"][:21], f"{int(state['build_pct']*100):3d}%"],
                "Loading...",
            )

        def draw_playback() -> None:
            mm, ss = divmod(int(state["elapsed_s"]), 60)
            st = "PLAYING" if state["playing"] else "READY"
            oled_message("UAP Caller", [st, f"Time {mm:02d}:{ss:02d}"], "SEL=Play BACK")

        def pump_stdout() -> None:
            # Drain EVERYTHING available; never stop on non-JSON lines.
            for key, _ in out_sel.select(timeout=0):
                while True:
                    try:
                        line = key.fileobj.readline()
                    except Exception:
                        return
                    if not line:
                        return

                    log(f"[child] {line.rstrip()}")

                    try:
                        msg = json.loads(line)
                    except Exception:
                        # Not JSON? ignore and keep draining.
                        continue

                    t = msg.get("type")
                    if t == "page":
                        state["page"] = msg.get("name", state["page"])
                    elif t == "build":
                        state["build_pct"] = float(msg.get("pct", state["build_pct"]))
                        state["build_step"] = str(msg.get("step", state["build_step"]))
                        state["elapsed_s"] = int(msg.get("elapsed_s", state["elapsed_s"]))
                    elif t == "state":
                        state["playing"] = bool(msg.get("playing", state["playing"]))
                        state["elapsed_s"] = int(msg.get("elapsed_s", state["elapsed_s"]))
                    elif t == "fatal":
                        state["page"] = "fatal"
                        state["build_step"] = str(msg.get("message", "fatal"))[:21]
                    elif t == "exit":
                        pass

        while proc.poll() is None:
            pump_stdout()

            if state.get("page") == "build":
                draw_build()
            elif state.get("page") == "fatal":
                oled_message("UAP Caller", ["ERROR", state.get("build_step", "")], "BACK")
            else:
                draw_playback()

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

        try:
            if proc.stdout:
                try:
                    out_sel.unregister(proc.stdout)
                except Exception:
                    pass
                proc.stdout.close()
        except Exception:
            pass

    # -----------------------------
    # Default modules: classic runner
    # -----------------------------
    else:
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

    log(f"[launcher] exit_code={proc.returncode}")

    if logf:
        try:
            logf.close()
        except Exception:
            pass

    clear()
    drain_events(consume, seconds=0.30)
    oled_hard_wake()

# =====================================================
# MAIN
# =====================================================
def main() -> None:
    # ---- Early SD guard (hard stop) ----
    err = sd_write_check()
    if err is not None:
        oled_message("SD ERROR", [err[:21], "", ""], "")
        while True:
            time.sleep(1)

    ensure_dirs()
    consume, clear, buttons = init_buttons()
    _, _, _, btn_back = buttons

    startup_sequence(consume, clear)
    status_refresh(force=True)

    modules = discover_modules(MODULE_DIR)
    if not modules:
        oled_message("NO MODULES", [str(MODULE_DIR)[:21], "Add module folders", ""], "HOLD=cfg")
        modules = [Module(id="none", name="(none)", subtitle="", entry_path="/bin/false", order=0)]

    idx = 0
    last_menu_draw = 0.0  # force immediate draw
    back_pressed_at = None

    def redraw_menu() -> None:
        nonlocal last_menu_draw
        draw_menu(modules, idx)
        last_menu_draw = time.time()

    redraw_menu()

    while True:
        now = time.time()

        # ---- Watchdog refresh (blank menu recovery) ----
        if now - last_menu_draw >= MENU_REFRESH_SECONDS:
            redraw_menu()

        # ---- Navigation ----
        if consume("up"):
            idx = (idx - 1) % len(modules)
            redraw_menu()

        if consume("down"):
            idx = (idx + 1) % len(modules)
            redraw_menu()

        # ---- Select: run module ----
        if consume("select"):
            # Prevent SELECT press from forwarding into the module at launch timing
            clear()
            drain_events(consume, seconds=0.10)

            if modules[idx].id != "none":
                run_module(modules[idx], consume, clear)

            # Always redraw menu immediately after returning from a module
            redraw_menu()

        # ---- Hold Select: settings ----
        if consume("select_hold"):
            settings(consume, clear)
            redraw_menu()

        # ---- BACK hold: reboot/poweroff (uses raw button state) ----
        if btn_back.is_pressed:
            if back_pressed_at is None:
                back_pressed_at = now

            held = now - back_pressed_at

            # Prioritize the longer hold first (poweroff > reboot)
            if held >= BACK_POWEROFF_HOLD:
                if confirm_action("POWEROFF?", consume, clear):
                    poweroff()
                    return
                back_pressed_at = None
                redraw_menu()

            elif held >= BACK_REBOOT_HOLD:
                if confirm_action("REBOOT?", consume, clear):
                    reboot()
                    return
                back_pressed_at = None
                redraw_menu()

        else:
            back_pressed_at = None

        # ---- CRITICAL FIX ----
        # Always eat any queued BACK events while in the menu loop.
        # Otherwise a stale BACK can leak into a module launch/return path.
        consume("back")

        time.sleep(0.02)


if __name__ == "__main__":
    main()
