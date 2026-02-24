#!/usr/bin/env python3
import os
os.environ["GPIOZERO_PIN_FACTORY"] = "lgpio"

import sys
import time
import math
import json
import selectors
import subprocess
import signal
import errno
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Any
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

# Branding
PRODUCT_NAME = "BLACKBOX"
PRODUCT_SUBTITLE = "PARANORMAL AUDIO"
TAGLINE = "FIELD UNIT"
VERSION = "v0.9"


# =====================================================
# OLED (re-init safe)
# =====================================================
_serial = None
device = None


def oled_init() -> None:
    global _serial, device
    _serial = i2c(port=I2C_PORT, address=I2C_ADDR)
    device = ssd1306(_serial, width=OLED_W, height=OLED_H)


def oled_hard_wake() -> None:
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


oled_init()


# =====================================================
# UTILITIES
# =====================================================
def sd_write_check() -> Optional[str]:
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
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


def draw_progress(draw, pct: float) -> None:
    x0, y0, x1, y1 = 8, 54, 120, 62
    draw.rectangle((x0, y0, x1, y1), outline=255, fill=0)
    w = int((x1 - x0 - 2) * max(0.0, min(1.0, pct)))
    draw.rectangle((x0 + 1, y0 + 1, x0 + 1 + w, y1 - 1), outline=255, fill=255)


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
# RADIO CONNECTIONS (Bluetooth autoconnect) - UNCHANGED
# =====================================================
CONNECTIONS_FILE = DATA_DIR / "connections.json"


def load_connections() -> dict:
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
    try:
        subprocess.run(["rfkill", "unblock", "bluetooth"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=2)
    except Exception:
        pass

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


_status_bt_mac = ""
_bt_ok = False
_last_status_check = 0.0
_wifi_bars = 0


def bluetooth_autoconnect_ui() -> bool:
    cfg = load_connections()
    bt = (cfg or {}).get("bluetooth", {}) if isinstance(cfg, dict) else {}
    mac = (bt.get("mac") or "").strip()
    autoconnect = bool(bt.get("autoconnect", False))
    global _status_bt_mac
    _status_bt_mac = mac

    if not mac or not autoconnect:
        return False

    oled_message(f"{PRODUCT_NAME} AUDIO", [mac, "Connecting...", ""], "BACK = skip")

    ok = bluetooth_connect(mac, timeout=8.0)

    if ok:
        oled_message("AUDIO READY", [mac, "", ""], "")
        time.sleep(0.6)
        return True

    oled_message("BT NOT CONNECTED", [mac, "Continuing...", ""], "")
    time.sleep(0.6)
    return False


def wifi_rssi_dbm(interface: str = "wlan0") -> Optional[int]:
    try:
        r = subprocess.run(["iw", "dev", interface, "link"], capture_output=True, text=True, timeout=2)
        out = (r.stdout or "").splitlines()
        if any("Not connected" in ln for ln in out):
            return None
        for ln in out:
            ln = ln.strip().lower()
            if ln.startswith("signal:"):
                parts = ln.replace("dbm", "").split()
                for p in parts:
                    if p.lstrip("-").isdigit():
                        return int(p)
        return None
    except Exception:
        return None


def wifi_bars_from_rssi(rssi: Optional[int]) -> int:
    if rssi is None:
        return 0
    if rssi >= -55:
        return 3
    if rssi >= -67:
        return 2
    if rssi >= -80:
        return 1
    return 0


def status_refresh(force: bool = False) -> None:
    global _last_status_check, _wifi_bars, _bt_ok

    now = time.time()
    if (not force) and (now - _last_status_check) < 2.0:
        return
    _last_status_check = now

    rssi = wifi_rssi_dbm("wlan0")
    if rssi is None:
        _wifi_bars = 1 if get_ip() else 0
    else:
        _wifi_bars = wifi_bars_from_rssi(rssi)

    if _status_bt_mac:
        _bt_ok = bluetooth_is_connected(_status_bt_mac)
    else:
        _bt_ok = False


def draw_wifi_bars(draw, x_right: int, y_top: int, bars: int) -> int:
    w = 2
    gap = 1
    heights = [3, 6, 9]
    total_w = 3 * w + 2 * gap
    x0 = x_right - total_w

    for i, h in enumerate(heights):
        x = x0 + i * (w + gap)
        y0 = y_top + (9 - h)
        draw.rectangle((x, y0, x + w - 1, y_top + 9), outline=255, fill=0)

    for i in range(min(3, max(0, bars))):
        h = heights[i]
        x = x0 + i * (w + gap)
        y0 = y_top + (9 - h)
        draw.rectangle((x, y0, x + w - 1, y_top + 9), outline=255, fill=255)

    return x0


def draw_bt_icon(draw, x_right: int, y_top: int, connected: bool) -> int:
    text = "B"
    x_text = max(0, x_right - 6)
    draw.text((x_text, y_top), text, fill=255)

    dot_x = x_text - 5
    dot_y = y_top + 3
    if connected:
        draw.ellipse((dot_x, dot_y, dot_x + 3, dot_y + 3), outline=255, fill=255)
    else:
        draw.ellipse((dot_x, dot_y, dot_x + 3, dot_y + 3), outline=255, fill=0)

    return dot_x


# =====================================================
# UI SCREENS
# =====================================================
def splash() -> None:
    start = time.time()
    phase = 0.0

    while True:
        ip = get_ip()
        net = "NET OK" if ip else "NET..."
        bt = "BT OK" if (_status_bt_mac and bluetooth_is_connected(_status_bt_mac)) else "BT..."

        oled_guard()
        with canvas(device) as draw:
            draw.text((0, 0), PRODUCT_NAME[:21], fill=255)
            draw.text((OLED_W - (len(VERSION) * 6), 0), VERSION, fill=255)
            draw.line((0, 12, 127, 12), fill=255)
            draw.text((0, 16), PRODUCT_SUBTITLE[:21], fill=255)
            draw.text((0, 28), TAGLINE[:21], fill=255)
            draw.line((0, 44, 127, 44), fill=255)
            draw.text((0, 48), f"{net}  {bt}"[:21], fill=255)
            draw_waveform(draw, phase)

        phase += 0.15

        if (time.time() - start) >= SPLASH_MIN_SECONDS and ip:
            return

        time.sleep(SPLASH_FRAME_SLEEP)


def startup_sequence(consume, clear) -> None:
    clear()
    drain_events(consume, seconds=0.10)

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

    bt_ok = bluetooth_autoconnect_ui()

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
        draw.text((0, 0), f"{PRODUCT_NAME} MENU"[:21], fill=255)

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
# UAP Caller: hardened non-blocking stdout pump
# =====================================================
class StdoutJSONPump:
    """
    Non-blocking pump that:
      - never calls readline()
      - drains as much as is available per tick
      - splits by newline into JSON lines
      - tolerates partial lines
    """
    def __init__(self, fileobj, log_fn):
        self.fileobj = fileobj
        self.log = log_fn
        self.buf = bytearray()
        self.fd = fileobj.fileno()
        os.set_blocking(self.fd, False)
        self.sel = selectors.DefaultSelector()
        self.sel.register(self.fd, selectors.EVENT_READ)

    def close(self):
        try:
            self.sel.unregister(self.fd)
        except Exception:
            pass
        try:
            self.fileobj.close()
        except Exception:
            pass
        try:
            self.sel.close()
        except Exception:
            pass

    def pump(self, max_bytes: int = 65536, max_lines: int = 50) -> List[Dict[str, Any]]:
        """
        Drain available stdout and return parsed JSON messages (up to max_lines).
        Never blocks.
        """
        msgs: List[Dict[str, Any]] = []
        if not self.sel.select(timeout=0):
            return msgs

        drained = 0
        while drained < max_bytes:
            try:
                chunk = os.read(self.fd, min(4096, max_bytes - drained))
            except BlockingIOError:
                break
            except OSError as e:
                if e.errno in (errno.EAGAIN, errno.EWOULDBLOCK):
                    break
                break

            if not chunk:
                break

            drained += len(chunk)
            self.buf.extend(chunk)

            # Process complete lines
            while b"\n" in self.buf and len(msgs) < max_lines:
                line, _, rest = self.buf.partition(b"\n")
                self.buf = bytearray(rest)

                if not line:
                    continue
                try:
                    s = line.decode("utf-8", errors="strict").strip()
                except Exception:
                    continue
                if not s:
                    continue

                # Child must never emit non-JSON; but be defensive.
                try:
                    obj = json.loads(s)
                except Exception:
                    self.log(f"[child-nonjson] {s}")
                    continue

                self.log(f"[child] {s}")
                msgs.append(obj)

            if len(msgs) >= max_lines:
                break

        return msgs


# =====================================================
# MODULE RUNNER
# =====================================================
def run_module(mod: Module, consume, clear) -> None:
    """
    Runs a module as a child process and forwards button events to it via stdin.

    Special JSON stdout UI paths:
      - uap_caller
      - noise_generator
      - tone_generator
      - spirit_box

    All other modules remain legacy (stdin forwarding only; stdout->log/devnull).
    """
    ensure_dirs()

    # Clear/drain BEFORE launch so stale BACK never gets forwarded.
    clear()
    drain_events(consume, seconds=0.20)

    oled_message("RUNNING", [mod.name, mod.subtitle], "BACK = exit")

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
    is_noise = (mod.id == "noise_generator")
    is_tone = (mod.id == "tone_generator")
    is_spirit = (mod.id == "spirit_box")
    is_json_ui = is_uap or is_noise or is_tone or is_spirit

    proc = None
    pump = None

    # Start child
    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE if is_json_ui else (logf if logf else subprocess.DEVNULL),
            stderr=logf if logf else subprocess.DEVNULL,
            text=False if is_json_ui else True,
            bufsize=0 if is_json_ui else 1,
            close_fds=True,
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
            if proc and proc.poll() is None and proc.stdin:
                proc.stdin.write((cmd_text + "\n").encode("utf-8"))
                proc.stdin.flush()
        except Exception:
            pass

    def graceful_exit() -> None:
        """Ask child to exit and then terminate if it doesn't."""
        try:
            send("back")
        except Exception:
            pass
        # give it a moment to exit cleanly
        for _ in range(50):
            if proc.poll() is not None:
                return
            time.sleep(0.02)
        try:
            proc.terminate()
        except Exception:
            pass

    def hold_first_buttons() -> bool:
        """
        Forward buttons with HOLD FIRST ordering.
        Returns True if BACK was pressed (caller can break).
        """
        if consume("up"):
            send("up")
        if consume("down"):
            send("down")

        if consume("select_hold"):
            send("select_hold")
            consume("select")  # discard any queued short-press from same physical press
        elif consume("select"):
            send("select")

        if consume("back"):
            graceful_exit()
            return True

        return False

    # Initialize JSON pump if needed
    if is_json_ui:
        try:
            if proc.stdout is None:
                raise RuntimeError("JSON UI modules require stdout=PIPE")
            pump = StdoutJSONPump(proc.stdout, log)
        except Exception as e:
            log(f"[launcher] pump_init_failed: {e!r}")
            oled_message(mod.name[:21], ["Pump init failed", str(e)[:21], ""], "BACK")
            try:
                proc.terminate()
            except Exception:
                pass
            time.sleep(1.0)
            oled_hard_wake()
            if logf:
                try:
                    logf.close()
                except Exception:
                    pass
            return

    try:
        # =====================================================
        # UAP Caller JSON UI path
        # =====================================================
        if is_uap:
            state: Dict[str, Any] = {
                "page": "build",          # build|playback|fatal
                "build_pct": 0.0,
                "build_step": "",
                "playing": False,
                "elapsed_s": 0,
                "fatal": "",
            }

            last_msg_time = time.time()
            last_draw_time = 0.0

            def draw_build() -> None:
                pct = float(state.get("build_pct") or 0.0)
                step = str(state.get("build_step") or "")[:21]
                oled_message("UAP Call Sig", [step, f"{int(pct*100):3d}%"], "Loading…")

            def draw_playback() -> None:
                mm, ss = divmod(int(state.get("elapsed_s") or 0), 60)
                playing = bool(state.get("playing"))
                stt = "PLAYING" if playing else "READY"
                oled_message("UAP Caller", [stt, f"Time {mm:02d}:{ss:02d}"], "SEL=Play  BACK")

            def apply_msg(msg: Dict[str, Any]) -> None:
                t = msg.get("type")
                if t == "page":
                    state["page"] = str(msg.get("name") or state["page"])
                elif t == "build":
                    state["page"] = "build"
                    state["build_pct"] = float(msg.get("pct", state["build_pct"]) or 0.0)
                    state["build_step"] = str(msg.get("step", state["build_step"]) or "")
                    if "elapsed_s" in msg:
                        state["elapsed_s"] = int(msg.get("elapsed_s") or 0)
                elif t == "state":
                    state["page"] = "playback"
                    state["playing"] = bool(msg.get("playing", state["playing"]))
                    state["elapsed_s"] = int(msg.get("elapsed_s", state["elapsed_s"]) or 0)
                elif t == "fatal":
                    state["page"] = "fatal"
                    state["fatal"] = str(msg.get("message") or "fatal")[:21]
                elif t == "exit":
                    pass

            while proc.poll() is None:
                msgs = pump.pump(max_bytes=65536, max_lines=80)
                if msgs:
                    last_msg_time = time.time()
                    for m in msgs:
                        apply_msg(m)

                now = time.time()
                if now - last_draw_time >= 0.10:
                    if state.get("page") == "build":
                        draw_build()
                    elif state.get("page") == "fatal":
                        oled_message("UAP Caller", ["ERROR", state.get("fatal", "")], "BACK")
                    else:
                        draw_playback()
                    last_draw_time = now

                if hold_first_buttons():
                    break

                if (time.time() - last_msg_time) > 20.0:
                    log("[launcher] watchdog: uap_caller silent >20s; terminating")
                    try:
                        proc.terminate()
                    except Exception:
                        pass
                    break

                time.sleep(0.02)

        # =====================================================
        # Noise Generator JSON UI path
        # =====================================================
        elif is_noise:
            state: Dict[str, Any] = {
                "page": "main",
                "ready": False,
                "noise_type": "white",
                "pulse_ms": 200,
                "playing": False,
                "cursor": "noise",        # main: noise|rate|play
                "menu_noise_idx": 0,      # scroll menu highlight
                "fatal": "",
            }

            last_msg_time = time.time()
            last_draw_time = 0.0

            def _noise_disp() -> str:
                raw = str(state.get("noise_type") or "white").strip().lower()
                return {"white": "White", "pink": "Pink", "brown": "Brown"}.get(
                    raw, (raw[:1].upper() + raw[1:]) if raw else "White"
                )

            def draw_main() -> None:
                noise_disp = _noise_disp()
                pulse_ms = int(state.get("pulse_ms") or 200)
                playing = bool(state.get("playing"))
                cursor = str(state.get("cursor") or "noise")

                items = [
                    ("noise", f"Noise Type: {noise_disp}"),
                    ("rate",  f"Sweep Rate: {pulse_ms}ms"),
                    ("play",  f"Play:       {'STOP' if playing else 'PLAY'}"),
                ]

                lines = []
                for k, text in items:
                    prefix = ">" if k == cursor else " "
                    lines.append((prefix + text)[:21])

                oled_message("Noise Generator", lines, "SEL=Open HOLD=Menu BACK")

            def draw_noise_menu_cycle() -> None:
                noise_disp = _noise_disp()
                lines = [
                    ("> " + noise_disp)[:21],
                    "  SEL=Next/Apply"[:21],
                    "  HOLD=Scroll"[:21],
                ]
                oled_message("Noise Types", lines, "BACK=Done")

            def draw_noise_menu_scroll() -> None:
                types = ["White", "Pink", "Brown"]
                idx = int(state.get("menu_noise_idx") or 0) % len(types)
                lines = []
                for i, name in enumerate(types):
                    prefix = ">" if i == idx else " "
                    lines.append((prefix + name)[:21])
                oled_message("Noise Types", lines, "SEL=Choose BACK=Cancel")

            def draw_fatal() -> None:
                msg = (str(state.get("fatal") or "Unknown error"))[:21]
                oled_message("Noise Gen", ["ERROR", msg, ""], "BACK")

            while proc.poll() is None:
                msgs = pump.pump(max_bytes=65536, max_lines=200)
                exit_requested = False

                if msgs:
                    last_msg_time = time.time()

                for msg in msgs:
                    t = msg.get("type")
                    if t == "page":
                        state["page"] = str(msg.get("name") or state.get("page") or "main")
                    elif t == "state":
                        for k in ("ready", "page", "noise_type", "pulse_ms", "playing", "cursor", "menu_noise_idx"):
                            if k in msg:
                                state[k] = msg.get(k)
                    elif t == "fatal":
                        state["page"] = "fatal"
                        state["fatal"] = str(msg.get("message", "fatal"))
                    elif t == "exit":
                        exit_requested = True

                now = time.time()
                if (now - last_draw_time) >= 0.08:
                    pg = str(state.get("page") or "main")
                    if pg == "fatal":
                        draw_fatal()
                    elif pg == "noise_menu_cycle":
                        draw_noise_menu_cycle()
                    elif pg == "noise_menu_scroll":
                        draw_noise_menu_scroll()
                    else:
                        draw_main()
                    last_draw_time = now

                if exit_requested:
                    break

                if hold_first_buttons():
                    break

                if (now - last_msg_time) > 15.0:
                    log("[launcher] watchdog: noise_generator silent >15s; terminating")
                    try:
                        proc.terminate()
                    except Exception:
                        pass
                    break

                time.sleep(0.02)

        # =====================================================
        # Tone Generator JSON UI path
        # =====================================================
        elif is_tone:
            # (unchanged from your file)
            state: Dict[str, Any] = {
                "page": "main",
                "ready": False,
                "playing": False,
                "freq_hz": 440,
                "volume": 70,
                "selection_label": "440Hz",
                "cursor_main": "frequency",
                "cursor_freq_menu": "manual",
                "idx_special_freq": 0,
                "idx_special_tone": 0,
                "toast": "",
                "toast_until": 0.0,
                "fatal": "",
            }

            SPECIAL_FREQS_UI = [
                (174, "Foundation Freq"),
                (285, "Healing Freq"),
                (396, "Liberating Freq"),
                (417, "Resonating Freq"),
                (528, "Love Freq"),
                (639, "Connecting Freq"),
                (741, "Awakening Freq"),
                (852, "Intuition Freq"),
                (936, "The Universe"),
            ]
            SPECIAL_TONES_UI = [
                ("sweep_asc", "Frequency Sweep Asc"),
                ("sweep_des", "Frequency Sweep Des"),
                ("sweep_bell", "Frequency Sweep Bell"),
                ("shepard_asc", "Shepard Tone Asc"),
                ("shepard_des", "Shepard Tone Des"),
                ("contact_call", "Contact Call (Original)"),
                ("contact_resp", "Contact Response (Original)"),
            ]

            last_msg_time = time.time()
            last_draw_time = 0.0

            def _text_w_px(s: str) -> int:
                return len(s) * 6

            def _draw_header(draw, title: str, status: str = ""):
                draw.text((2, 0), title[:21], fill=255)
                if status:
                    s = status[:6]
                    x = max(0, OLED_W - _text_w_px(s) - 2)
                    draw.text((x, 0), s, fill=255)
                draw.line((0, 12, 127, 12), fill=255)

            def _draw_footer(draw, text: str):
                draw.line((0, 52, 127, 52), fill=255)
                draw.text((2, 54), text[:21], fill=255)

            def _draw_row(draw, y: int, text: str, selected: bool):
                marker = ">" if selected else " "
                draw.text((0, y), marker, fill=255)
                draw.text((10, y), text[:19], fill=255)

            def _toast_active() -> str:
                now = time.time()
                if state.get("toast") and now < float(state.get("toast_until") or 0.0):
                    return str(state.get("toast") or "")[:21]
                return ""

            def _draw_toast(draw, toast_text: str):
                if not toast_text:
                    return
                draw.rectangle((0, 38, 127, 51), outline=255, fill=0)
                draw.text((2, 40), toast_text, fill=255)

            def _status() -> str:
                ready = bool(state.get("ready"))
                playing = bool(state.get("playing"))
                return "PLAY" if playing else ("RDY" if ready else "ERR")

            def draw_main():
                label = str(state.get("selection_label") or f"{int(state.get('freq_hz') or 440)}Hz")
                vol = int(state.get("volume") or 70)
                playing = bool(state.get("playing"))
                cursor = str(state.get("cursor_main") or "frequency")

                rows = [
                    ("frequency", f"Frequency: {label}"[:19]),
                    ("volume",    f"Volume: {vol}%"[:19]),
                    ("play",      f"Play: {'STOP' if playing else 'PLAY'}"[:19]),
                ]
                toast_text = _toast_active()

                oled_guard()
                with canvas(device) as draw:
                    _draw_header(draw, "Tone Generator", status=_status())
                    y0 = 14
                    row_h = 12
                    for i, (k, txt) in enumerate(rows):
                        _draw_row(draw, y0 + i * row_h, txt, selected=(k == cursor))
                    _draw_footer(draw, "SEL=enter/chg BACK")
                    _draw_toast(draw, toast_text)

            def draw_freq_menu():
                cursor = str(state.get("cursor_freq_menu") or "manual")
                rows = [
                    ("manual",       "Manual Frequency"),
                    ("special_freq", "Special Frequency"),
                    ("special_tone", "Special Tones"),
                ]
                toast_text = _toast_active()

                oled_guard()
                with canvas(device) as draw:
                    _draw_header(draw, "Frequency", status=_status())
                    y0 = 14
                    row_h = 12
                    for i, (k, txt) in enumerate(rows):
                        _draw_row(draw, y0 + i * row_h, txt, selected=(k == cursor))
                    _draw_footer(draw, "SEL=enter  BACK")
                    _draw_toast(draw, toast_text)

            def draw_freq_edit():
                freq = int(state.get("freq_hz") or 440)
                toast_text = _toast_active()

                oled_guard()
                with canvas(device) as draw:
                    _draw_header(draw, "Manual Freq", status=_status())
                    draw.text((2, 18), f"{freq} Hz"[:21], fill=255)
                    draw.text((2, 32), "UP/DN change"[:21], fill=255)
                    draw.text((2, 44), "SEL done  BACK"[:21], fill=255)
                    _draw_toast(draw, toast_text)

            def draw_list_page(title: str, items: List[str], idx: int):
                n = len(items)
                idx = 0 if n == 0 else max(0, min(n - 1, idx))
                start = max(0, min(idx - 1, n - 3))
                window = items[start:start + 3]
                toast_text = _toast_active()

                oled_guard()
                with canvas(device) as draw:
                    _draw_header(draw, title, status=_status())
                    y0 = 14
                    row_h = 12
                    for i, label2 in enumerate(window):
                        selected = (start + i) == idx
                        _draw_row(draw, y0 + i * row_h, label2, selected)
                    _draw_footer(draw, "SEL pick  BACK")
                    _draw_toast(draw, toast_text)

            def draw_special_freqs():
                idx = int(state.get("idx_special_freq") or 0)
                items = [f"{hz}Hz {name}"[:19] for (hz, name) in SPECIAL_FREQS_UI]
                draw_list_page("Special Freqs", items, idx)

            def draw_special_tones():
                idx = int(state.get("idx_special_tone") or 0)
                items = [lbl[:19] for (_tid, lbl) in SPECIAL_TONES_UI]
                draw_list_page("Special Tones", items, idx)

            def draw_fatal():
                msg = (str(state.get("fatal") or "Unknown error"))[:21]
                oled_message("Tone Gen", ["ERROR", msg, ""], "BACK")

            while proc.poll() is None:
                msgs = pump.pump(max_bytes=65536, max_lines=200)
                exit_requested = False

                if msgs:
                    last_msg_time = time.time()

                for msg in msgs:
                    t = msg.get("type")
                    if t == "page":
                        state["page"] = msg.get("name", state.get("page", "main"))
                    elif t == "state":
                        for k in (
                            "page", "ready", "playing", "freq_hz", "volume",
                            "selection_label", "cursor_main", "cursor_freq_menu",
                            "idx_special_freq", "idx_special_tone"
                        ):
                            if k in msg:
                                state[k] = msg.get(k)
                    elif t == "toast":
                        txt = str(msg.get("message") or "")[:21]
                        if txt:
                            state["toast"] = txt
                            state["toast_until"] = time.time() + 1.2
                    elif t == "fatal":
                        state["page"] = "fatal"
                        state["fatal"] = str(msg.get("message", "fatal"))
                        state["toast"] = state["fatal"][:21]
                        state["toast_until"] = time.time() + 2.0
                    elif t == "exit":
                        exit_requested = True

                now = time.time()
                if (now - last_draw_time) >= 0.08:
                    pg = str(state.get("page") or "main")
                    if pg == "fatal":
                        draw_fatal()
                    elif pg == "freq_menu":
                        draw_freq_menu()
                    elif pg == "freq_edit":
                        draw_freq_edit()
                    elif pg == "special_freqs":
                        draw_special_freqs()
                    elif pg == "special_tones":
                        draw_special_tones()
                    else:
                        draw_main()
                    last_draw_time = now

                if exit_requested:
                    break

                if hold_first_buttons():
                    break

                if (now - last_msg_time) > 15.0:
                    log("[launcher] watchdog: tone_generator silent >15s; terminating")
                    try:
                        proc.terminate()
                    except Exception:
                        pass
                    break

                time.sleep(0.02)

        # =====================================================
        # Spirit Box JSON UI path
        # =====================================================
        elif is_spirit:
            # Spirit Box UI v4:
            # - MAIN: display Rate/Mode/Frequency + selectable: Status and Settings
            # - SETTINGS: up/down selects Rate/Mode; select toggles edit; up/down edits when in edit
            #             select_hold saves & returns to MAIN; back returns to MAIN
            state: Dict[str, Any] = {
                "page": "main",                 # main|settings|fatal
                "ready": False,
                "sweep_ms": 200,
                "mode": "fwd",                  # fwd|bwd|rdm
                "freq_mhz": 99.5,
                "playing": False,

                "main_cursor": "play",          # play|settings
                "settings_cursor": "rate",      # rate|mode
                "settings_edit": False,

                # legacy (older spirit_box modules)
                "cursor": "rate",
                "direction": "fwd",
                "mode_legacy": "scan",
                "fatal": "",
            }

            last_msg_time = time.time()
            last_draw_time = 0.0
            saw_new_schema = False
            first_state_seen = False
            legacy_fallback_allowed_after = time.time() + 1.0  # avoid brief legacy flash

            def _mode_disp_new() -> str:
                m = str(state.get("mode") or "fwd").lower().strip()
                return {"fwd": "Fwd", "bwd": "Bwd", "rdm": "Rdm"}.get(m, (m[:3].upper() if m else "Fwd"))

            def _freq_disp() -> str:
                f = state.get("freq_mhz", None)
                try:
                    return f"{float(f):0.1f} MHz"
                except Exception:
                    return "—"

            def draw_loading() -> None:
                oled_message("Spirit Box", ["Loading…", "", ""], "BACK")

            def draw_main_clean() -> None:
                sweep_ms = int(state.get("sweep_ms") or 200)
                playing = bool(state.get("playing"))
                status_txt = "PLAY" if playing else "STOP"
                mode_txt = _mode_disp_new()
                freq_txt = _freq_disp()
                cur = str(state.get("main_cursor") or "play").lower().strip()

                l0 = f"Rate:{sweep_ms} Mode:{mode_txt}"[:21]
                l1 = f"Freq: {freq_txt}"[:21]
                l2 = (('>' if cur == 'play' else ' ') + f"Status: {status_txt}")[:21]
                l3 = (('>' if cur == 'settings' else ' ') + "Settings")[:21]

                oled_guard()
                with canvas(device) as draw:
                    draw.text((0, 0), l0, fill=255)
                    draw.line((0, 12, 127, 12), fill=255)
                    draw.text((0, 16), l1, fill=255)
                    draw.text((0, 32), l2, fill=255)
                    draw.text((0, 48), l3, fill=255)

            def draw_settings_clean() -> None:
                sweep_ms = int(state.get("sweep_ms") or 200)
                mode_txt = _mode_disp_new()
                cur = str(state.get("settings_cursor") or "rate").lower().strip()
                edit = bool(state.get("settings_edit"))

                # selection marker
                r_mark = ">" if cur == "rate" else " "
                m_mark = ">" if cur == "mode" else " "

                # edit marker (when in edit, make the active value obvious)
                r_edit = "*" if (edit and cur == "rate") else " "
                m_edit = "*" if (edit and cur == "mode") else " "

                l1 = f"{r_mark}Rate: {sweep_ms}ms {r_edit}"[:21]
                l2 = f"{m_mark}Mode: {mode_txt} {m_edit}"[:21]

                footer = "SEL=Edit UP/DN nav" if not edit else "UP/DN change SEL=OK"
                # Back/hold info rotates in the title area via launcher; keep footer short
                footer2 = "HOLD=Save  BACK=Main"

                oled_guard()
                with canvas(device) as draw:
                    draw.text((0, 0), "Spirit Box Settings"[:21], fill=255)
                    draw.line((0, 12, 127, 12), fill=255)
                    draw.text((0, 16), l1, fill=255)
                    draw.text((0, 32), l2, fill=255)
                    draw.line((0, 44, 127, 44), fill=255)
                    # alternate footers every ~1s so user sees both
                    if int(time.time()) % 2 == 0:
                        draw.text((0, 48), footer[:21], fill=255)
                    else:
                        draw.text((0, 48), footer2[:21], fill=255)

            # --- legacy UI fallback ---
            def _dir_disp_legacy() -> str:
                d = str(state.get("direction") or "fwd").lower()
                return "REV" if d.startswith("r") else "FWD"

            def _mode_disp_legacy() -> str:
                m = str(state.get("mode_legacy") or "scan").lower()
                return "Burst" if m == "burst" else "Scan"

            def draw_main_legacy() -> None:
                sweep_ms = int(state.get("sweep_ms") or 200)
                playing = bool(state.get("playing"))
                cursor = str(state.get("cursor") or "rate")

                items = [
                    ("rate",      f"Sweep Rate: {sweep_ms} ms"),
                    ("direction", f"Direction:  {_dir_disp_legacy()}"),
                    ("mode",      f"Mode:       {_mode_disp_legacy()}"),
                    ("play",      f"Play:       {'STOP' if playing else 'PLAY'}"),
                ]

                keys = [k for (k, _) in items]
                try:
                    idx = keys.index(cursor)
                except Exception:
                    idx = 0
                start = max(0, min(idx - 1, len(items) - 3))
                window = items[start:start + 3]

                lines = []
                for k, txt in window:
                    prefix = ">" if k == cursor else " "
                    lines.append((prefix + txt)[:21])

                oled_message("Spirit Box", lines, "SEL/HOLD chg  BACK")

            def draw_fatal() -> None:
                msg = (str(state.get("fatal") or "Unknown error"))[:21]
                oled_message("Spirit Box", ["ERROR", msg, ""], "BACK")

            # show loading immediately to avoid legacy flash
            draw_loading()
            last_draw_time = time.time()

            while proc.poll() is None:
                msgs = pump.pump(max_bytes=65536, max_lines=200)
                exit_requested = False

                if msgs:
                    last_msg_time = time.time()

                for msg in msgs:
                    t = msg.get("type")
                    if t == "page":
                        state["page"] = str(msg.get("name") or state.get("page") or "main")
                    elif t == "state":
                        first_state_seen = True

                        # detect new schema
                        if "main_cursor" in msg or "settings_cursor" in msg or "settings_edit" in msg or "freq_mhz" in msg:
                            saw_new_schema = True
                        if "mode" in msg and str(msg.get("mode") or "").lower() in ("fwd", "bwd", "rdm"):
                            saw_new_schema = True
                        if "page" in msg and str(msg.get("page") or "") in ("main", "settings"):
                            saw_new_schema = True

                        # new keys
                        for k in ("ready", "page", "sweep_ms", "playing", "freq_mhz", "mode",
                                  "main_cursor", "settings_cursor", "settings_edit"):
                            if k in msg:
                                state[k] = msg.get(k)

                        # legacy keys (if old module)
                        for k in ("direction", "cursor"):
                            if k in msg:
                                state[k] = msg.get(k)
                        if "mode_legacy" in msg:
                            state["mode_legacy"] = msg.get("mode_legacy")
                        if "mode" in msg and str(msg.get("mode") or "").lower() in ("scan", "burst"):
                            state["mode_legacy"] = msg.get("mode")

                    elif t == "fatal":
                        state["page"] = "fatal"
                        state["fatal"] = str(msg.get("message", "fatal"))
                    elif t == "exit":
                        exit_requested = True

                now = time.time()
                if (now - last_draw_time) >= 0.08:
                    pg = str(state.get("page") or "main")
                    if pg == "fatal":
                        draw_fatal()
                    else:
                        if saw_new_schema:
                            if pg == "settings":
                                draw_settings_clean()
                            else:
                                draw_main_clean()
                        else:
                            if first_state_seen and now >= legacy_fallback_allowed_after:
                                draw_main_legacy()
                            else:
                                draw_loading()
                    last_draw_time = now

                if exit_requested:
                    break

                if hold_first_buttons():
                    break

                if (now - last_msg_time) > 15.0:
                    log("[launcher] watchdog: spirit_box silent >15s; terminating")
                    try:
                        proc.terminate()
                    except Exception:
                        pass
                    break

                time.sleep(0.02)

        else:
            while proc.poll() is None:
                if consume("up"):
                    send("up")
                if consume("down"):
                    send("down")
                if consume("select_hold"):
                    send("select_hold")
                    consume("select")
                elif consume("select"):
                    send("select")

                if consume("back"):
                    graceful_exit()
                    break

                time.sleep(0.02)

    finally:
        # Cleanup pump/stdout
        try:
            if pump:
                pump.close()
        except Exception:
            pass

        # cleanup stdin
        try:
            if proc and proc.stdin:
                proc.stdin.close()
        except Exception:
            pass

        # wait a bit; force kill if needed
        try:
            if proc:
                proc.wait(timeout=1.0)
        except Exception:
            try:
                if proc:
                    proc.kill()
            except Exception:
                pass

        try:
            if proc:
                log(f"[launcher] exit_code={proc.returncode}")
        except Exception:
            pass

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
    last_menu_draw = 0.0
    back_pressed_at = None

    def redraw_menu() -> None:
        nonlocal last_menu_draw
        draw_menu(modules, idx)
        last_menu_draw = time.time()

    redraw_menu()

    while True:
        now = time.time()

        if now - last_menu_draw >= MENU_REFRESH_SECONDS:
            redraw_menu()

        if consume("up"):
            idx = (idx - 1) % len(modules)
            redraw_menu()

        if consume("down"):
            idx = (idx + 1) % len(modules)
            redraw_menu()

        if consume("select"):
            clear()
            drain_events(consume, seconds=0.10)

            if modules[idx].id != "none":
                run_module(modules[idx], consume, clear)

            redraw_menu()

        if consume("select_hold"):
            settings(consume, clear)
            redraw_menu()

        if btn_back.is_pressed:
            if back_pressed_at is None:
                back_pressed_at = now

            held = now - back_pressed_at

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

        # Eat queued BACK events while in menu loop.
        consume("back")

        time.sleep(0.02)


if __name__ == "__main__":
    main()
