#!/usr/bin/env python3
import sys
import time
import random
import struct
import subprocess
import threading
import select
from dataclasses import dataclass

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

HEADER_Y = 0
DIVIDER_Y = 12
LIST_Y0 = 14
ROW_H = 12
FOOTER_LINE_Y = 52
FOOTER_Y = 54


def render(draw_fn):
    with canvas(device) as d:
        draw_fn(d)


def draw_header(d, title: str):
    d.text((2, HEADER_Y), title[:21], fill=255)
    d.line((0, DIVIDER_Y, 127, DIVIDER_Y), fill=255)


def draw_footer(d, text: str):
    d.line((0, FOOTER_LINE_Y, 127, FOOTER_LINE_Y), fill=255)
    d.text((2, FOOTER_Y), text[:21], fill=255)


def draw_row(d, y: int, text: str, selected: bool):
    marker = ">" if selected else " "
    d.text((0, y), marker, fill=255)
    d.text((10, y), text[:19], fill=255)


# ============================================================
# INPUT (stdin from parent app.py) — SpiritBox style
# ============================================================
def read_event():
    try:
        r, _, _ = select.select([sys.stdin], [], [], 0)
    except Exception:
        return None
    if not r:
        return None
    line = sys.stdin.readline()
    if not line:
        return None
    return line.strip().lower()


# ============================================================
# AUDIO ENGINE — continuous raw PCM stream into aplay
# ============================================================
RATE = 48000
CH = 1
FMT = "S16_LE"
FRAMES_PER_CHUNK = 1024

NOISE_TYPES = ["White", "Pink", "Brown"]


@dataclass
class AudioState:
    noise_idx: int = 0
    playing: bool = False
    volume: int = 80            # 0..100
    pulse_on: bool = False
    pulse_ms: int = 250         # 50..2000
    duty: float = 0.50          # 0.05..0.95


class NoiseEngine:
    def __init__(self, st: AudioState):
        self.st = st
        self._stop = threading.Event()
        self._thread = None
        self._proc = None

        self._brown = 0.0
        self._pink_b = [0.0] * 7

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def shutdown(self):
        self._stop.set()
        self._close_proc()

    def _ensure_proc(self):
        if self._proc and self._proc.poll() is None:
            return
        self._proc = subprocess.Popen(
            ["aplay", "-q", "-f", FMT, "-r", str(RATE), "-c", str(CH), "-t", "raw", "-"],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def _close_proc(self):
        try:
            if self._proc and self._proc.stdin:
                self._proc.stdin.close()
        except Exception:
            pass
        try:
            if self._proc and self._proc.poll() is None:
                self._proc.terminate()
        except Exception:
            pass
        self._proc = None

    def _white(self, n):
        return [(random.random() * 2.0 - 1.0) for _ in range(n)]

    def _pink(self, w):
        b = self._pink_b
        out = []
        for x in w:
            b[0] = 0.99886*b[0] + x*0.0555179
            b[1] = 0.99332*b[1] + x*0.0750759
            b[2] = 0.96900*b[2] + x*0.1538520
            b[3] = 0.86650*b[3] + x*0.3104856
            b[4] = 0.55000*b[4] + x*0.5329522
            b[5] = -0.7616*b[5] - x*0.0168980
            y = b[0]+b[1]+b[2]+b[3]+b[4]+b[5]+b[6] + x*0.5362
            b[6] = x*0.115926
            out.append(y * 0.11)
        return out

    def _brown_noise(self, w):
        out = []
        y = self._brown
        for x in w:
            y = 0.98*y + 0.02*x
            out.append(y * 3.5)
        self._brown = y
        return out

    def _amp(self, t0):
        amp = max(0.0, min(1.0, self.st.volume / 100.0))
        if self.st.pulse_on:
            period = max(50, min(2000, int(self.st.pulse_ms))) / 1000.0
            duty = max(0.05, min(0.95, float(self.st.duty)))
            ph = (time.time() - t0) % period
            gate = 1.0 if ph < (period * duty) else 0.0
            amp *= gate
        return amp

    def _float_to_pcm(self, xs):
        buf = bytearray()
        for x in xs:
            x = max(-1.0, min(1.0, x))
            buf += struct.pack("<h", int(x * 32767))
        return bytes(buf)

    def _run(self):
        t0 = time.time()
        while not self._stop.is_set():
            if not self.st.playing:
                self._close_proc()
                time.sleep(0.03)
                continue

            self._ensure_proc()

            w = self._white(FRAMES_PER_CHUNK)
            if self.st.noise_idx == 0:
                xs = w
            elif self.st.noise_idx == 1:
                xs = self._pink(w)
            else:
                xs = self._brown_noise(w)

            a = self._amp(t0)
            xs = [x * a for x in xs]
            pcm = self._float_to_pcm(xs)

            try:
                if self._proc and self._proc.stdin:
                    self._proc.stdin.write(pcm)
                    self._proc.stdin.flush()
            except Exception:
                pass

            time.sleep(FRAMES_PER_CHUNK / RATE)


# ============================================================
# UI + STATE MACHINE (SpiritBox model)
# ============================================================
def menu_screen(sel_idx):
    def _draw(d):
        draw_header(d, "NOISE GEN")
        for i, name in enumerate(NOISE_TYPES):
            draw_row(d, LIST_Y0 + i * ROW_H, name, selected=(i == sel_idx))
        draw_footer(d, "SEL enter  BACK exit")
    return _draw


def play_screen(st: AudioState):
    def _draw(d):
        draw_header(d, "NOISE")
        d.text((2, LIST_Y0), NOISE_TYPES[st.noise_idx][:21], fill=255)
        status = "PLAYING" if st.playing else "STOPPED"
        d.text((2, LIST_Y0 + 12), status, fill=255)
        mode = "PULSE" if st.pulse_on else "STEADY"
        d.text((2, LIST_Y0 + 24), f"{mode}  VOL {st.volume}%"[:21], fill=255)
        draw_footer(d, "SEL toggle  HOLD cfg  BK")
    return _draw


def settings_screen(st: AudioState, sel_idx: int):
    items = [
        ("Pulse", "On" if st.pulse_on else "Off"),
        ("Pulse ms", str(st.pulse_ms)),
        ("Duty", f"{int(st.duty*100)}%"),
        ("Volume", str(st.volume)),
    ]

    def _draw(d):
        draw_header(d, "SETTINGS")
        for i, (k, v) in enumerate(items):
            draw_row(d, LIST_Y0 + i * ROW_H, f"{k}: {v}", selected=(i == sel_idx))
        draw_footer(d, "SEL toggle  ^v adj  BK")
    return _draw


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def main():
    st = AudioState()
    engine = NoiseEngine(st)
    engine.start()  # thread is idle unless st.playing=True

    STATE_MENU = "MENU"
    STATE_PLAY = "PLAY"
    STATE_CFG = "CFG"

    state = STATE_MENU
    menu_sel = 0
    cfg_sel = 0

    while True:
        if state == STATE_MENU:
            render(menu_screen(menu_sel))
        elif state == STATE_PLAY:
            render(play_screen(st))
        elif state == STATE_CFG:
            render(settings_screen(st, cfg_sel))

        ev = read_event()

        if ev is None:
            time.sleep(0.02)
            continue

        # ---------- MENU ----------
        if state == STATE_MENU:
            if ev == "up":
                menu_sel = (menu_sel - 1) % len(NOISE_TYPES)
            elif ev == "down":
                menu_sel = (menu_sel + 1) % len(NOISE_TYPES)
            elif ev == "select":
                st.noise_idx = menu_sel
                st.playing = False
                state = STATE_PLAY
            elif ev == "back":
                # EXIT MODULE only from top menu
                break

        # ---------- PLAY ----------
        elif state == STATE_PLAY:
            if ev == "select":
                st.playing = not st.playing
            elif ev == "select_hold":
                cfg_sel = 0
                state = STATE_CFG
            elif ev == "back":
                # BACK returns to menu, does NOT exit module
                st.playing = False
                state = STATE_MENU
            elif ev == "up":
                st.volume = clamp(st.volume + 5, 0, 100)
            elif ev == "down":
                st.volume = clamp(st.volume - 5, 0, 100)

        # ---------- CFG ----------
        elif state == STATE_CFG:
            if ev == "up":
                cfg_sel = (cfg_sel - 1) % 4
            elif ev == "down":
                cfg_sel = (cfg_sel + 1) % 4
            elif ev == "select":
                if cfg_sel == 0:
                    st.pulse_on = not st.pulse_on
            elif ev == "back":
                state = STATE_PLAY
            elif ev == "select_hold":
                state = STATE_PLAY

            # Adjust numeric values with UP/DOWN while highlighted? (SpiritBox style = edit screen)
            # We'll do lightweight direct edits here:
            if ev in ("up", "down"):
                step = +1 if ev == "up" else -1
                if cfg_sel == 1:  # pulse_ms
                    st.pulse_ms = clamp(st.pulse_ms + step * 50, 50, 2000)
                elif cfg_sel == 2:  # duty
                    st.duty = clamp(st.duty + step * 0.05, 0.05, 0.95)
                elif cfg_sel == 3:  # volume
                    st.volume = clamp(st.volume + step * 5, 0, 100)

        time.sleep(0.02)

    st.playing = False
    engine.shutdown()
    render(lambda d: d.rectangle((0, 0, OLED_W - 1, OLED_H - 1), outline=0, fill=0))
    time.sleep(0.15)


if __name__ == "__main__":
    main()
