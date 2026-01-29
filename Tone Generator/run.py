#!/usr/bin/env python3
import sys
import time
import math
import struct
import subprocess
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


def _text_w_px(s: str) -> int:
    return len(s) * 6  # default font width approximation


def draw_header(d, title: str, status: str = ""):
    d.text((2, HEADER_Y), title[:21], fill=255)
    if status:
        s = status[:6]
        x = max(0, OLED_W - _text_w_px(s) - 2)
        d.text((x, HEADER_Y), s, fill=255)
    d.line((0, DIVIDER_Y, 127, DIVIDER_Y), fill=255)


def draw_footer(d, text: str):
    d.line((0, FOOTER_LINE_Y, 127, FOOTER_LINE_Y), fill=255)
    d.text((2, FOOTER_Y), text[:21], fill=255)


def draw_row(d, y: int, text: str, selected: bool):
    marker = ">" if selected else " "
    d.text((0, y), marker, fill=255)
    d.text((10, y), text[:19], fill=255)


def read_event():
    """Non-blocking stdin event read (SpiritBox style)."""
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
# AUDIO STREAMING (same proven path as NoiseGenerator)
# ============================================================
RATE = 48000
CH = 1
FMT = "S16_LE"
FRAMES_PER_CHUNK = 1024
TWOPI = 2.0 * math.pi


def start_aplay():
    return subprocess.Popen(
        ["aplay", "-q", "-f", FMT, "-r", str(RATE), "-c", str(CH), "-t", "raw", "-"],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def stop_proc(p):
    if not p:
        return
    try:
        if p.stdin:
            p.stdin.close()
    except Exception:
        pass
    try:
        if p.poll() is None:
            p.terminate()
    except Exception:
        pass


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


@dataclass
class ToneState:
    mode: str = "STEADY"   # STEADY/PULSE/SWEEP
    playing: bool = False
    hz: float = 432.0
    volume: int = 70       # 0..100

    # pulse
    pulse_on_ms: int = 250
    pulse_off_ms: int = 250
    _pulse_acc_ms: int = 0
    _pulse_on: bool = True

    # sweep
    sweep_start: float = 100.0
    sweep_end: float = 1200.0
    sweep_step: float = 5.0
    sweep_step_ms: int = 100
    _sweep_hz: float = 100.0
    _sweep_dir: int = 1
    _sweep_acc_ms: int = 0


def gen_sine_block(phase: float, hz: float, vol: float):
    """Return (bytes, new_phase)."""
    inc = TWOPI * hz / RATE
    out = bytearray()
    for _ in range(FRAMES_PER_CHUNK):
        v = math.sin(phase) * vol
        s = int(max(-1.0, min(1.0, v)) * 32767)
        out += struct.pack("<h", s)
        phase += inc
        if phase > TWOPI:
            phase -= TWOPI
    return bytes(out), phase


def gen_silence_block():
    return b"\x00\x00" * FRAMES_PER_CHUNK


# ============================================================
# UI
# ============================================================
MAIN_ITEMS = ["Steady", "Pulse", "Sweep"]


def menu_screen(sel):
    def _draw(d):
        draw_header(d, "TONE GEN")
        for i, it in enumerate(MAIN_ITEMS):
            draw_row(d, LIST_Y0 + i * ROW_H, it, selected=(i == sel))
        draw_footer(d, "SEL enter  BACK exit")
    return _draw


def play_screen(st: ToneState):
    def _draw(d):
        status = "PLAY" if st.playing else "STOP"
        draw_header(d, "TONE", status=status)
        d.text((2, LIST_Y0), f"Mode: {st.mode}"[:21], fill=255)
        d.text((2, LIST_Y0 + 12), f"{st.hz:.1f} Hz"[:21], fill=255)
        d.text((2, LIST_Y0 + 24), f"VOL {st.volume}%"[:21], fill=255)
        draw_footer(d, "SEL toggle  ^v vol  BK")
    return _draw


# ============================================================
# MAIN (SpiritBox-style state machine)
# ============================================================
def main():
    st = ToneState()
    st._sweep_hz = st.sweep_start

    STATE_MENU = "MENU"
    STATE_PLAY = "PLAY"

    state = STATE_MENU
    menu_sel = 0

    # Strong back lockout after returning to menu so a held/back-bounce
    # can't immediately cause MENU->EXIT.
    back_block_until = 0.0

    # audio
    proc = None
    phase = 0.0

    render(menu_screen(menu_sel))

    last_audio_tick = time.monotonic()

    while True:
        ev = read_event()
        now = time.monotonic()

        # -----------------------
        # Handle UI events
        # -----------------------
        if state == STATE_MENU:
            if ev == "up":
                menu_sel = (menu_sel - 1) % len(MAIN_ITEMS)
                render(menu_screen(menu_sel))

            elif ev == "down":
                menu_sel = (menu_sel + 1) % len(MAIN_ITEMS)
                render(menu_screen(menu_sel))

            elif ev == "select":
                st.mode = MAIN_ITEMS[menu_sel].upper()
                st.playing = False
                # reset mode-specific internals
                st._pulse_acc_ms = 0
                st._pulse_on = True
                st._sweep_hz = st.sweep_start
                st._sweep_dir = 1
                st._sweep_acc_ms = 0

                state = STATE_PLAY
                render(play_screen(st))

            elif ev == "back":
                if now >= back_block_until:
                    break  # exit module ONLY from top menu

        elif state == STATE_PLAY:
            if ev == "up":
                st.volume = clamp(st.volume + 5, 0, 100)
                render(play_screen(st))

            elif ev == "down":
                st.volume = clamp(st.volume - 5, 0, 100)
                render(play_screen(st))

            elif ev == "select":
                st.playing = not st.playing
                render(play_screen(st))

            elif ev == "back":
                # stop playback + return to menu (do NOT exit)
                st.playing = False
                stop_proc(proc)
                proc = None
                state = STATE_MENU
                render(menu_screen(menu_sel))
                back_block_until = time.monotonic() + 1.25  # strong lockout

        # -----------------------
        # Audio tick (continuous)
        # -----------------------
        # Stream at roughly real-time pace.
        # This is what avoids the "2s chunk loop" and ensures continuous tone.
        dt = now - last_audio_tick
        if dt < (FRAMES_PER_CHUNK / RATE):
            time.sleep(0.001)
            continue
        last_audio_tick = now

        if st.playing:
            if proc is None or proc.poll() is not None:
                proc = start_aplay()

            vol = clamp(st.volume / 100.0, 0.0, 1.0)

            if st.mode == "STEADY":
                buf, phase = gen_sine_block(phase, st.hz, vol)

            elif st.mode == "PULSE":
                tick_ms = int(1000 * (FRAMES_PER_CHUNK / RATE))
                st._pulse_acc_ms += tick_ms
                if st._pulse_on:
                    buf, phase = gen_sine_block(phase, st.hz, vol)
                    if st._pulse_acc_ms >= st.pulse_on_ms:
                        st._pulse_acc_ms = 0
                        st._pulse_on = False
                else:
                    buf = gen_silence_block()
                    if st._pulse_acc_ms >= st.pulse_off_ms:
                        st._pulse_acc_ms = 0
                        st._pulse_on = True

            else:  # SWEEP
                tick_ms = int(1000 * (FRAMES_PER_CHUNK / RATE))
                buf, phase = gen_sine_block(phase, st._sweep_hz, vol)

                st._sweep_acc_ms += tick_ms
                if st._sweep_acc_ms >= st.sweep_step_ms:
                    st._sweep_acc_ms = 0
                    st._sweep_hz += st.sweep_step * st._sweep_dir
                    if st._sweep_hz >= st.sweep_end:
                        st._sweep_hz = st.sweep_end
                        st._sweep_dir = -1
                    elif st._sweep_hz <= st.sweep_start:
                        st._sweep_hz = st.sweep_start
                        st._sweep_dir = 1

            try:
                if proc.stdin:
                    proc.stdin.write(buf)
                    proc.stdin.flush()
            except Exception:
                stop_proc(proc)
                proc = None

        else:
            # not playing -> ensure proc is stopped
            if proc is not None:
                stop_proc(proc)
                proc = None

        time.sleep(0.001)

    # cleanup
    st.playing = False
    stop_proc(proc)
    render(lambda d: d.rectangle((0, 0, OLED_W - 1, OLED_H - 1), outline=0, fill=0))
    time.sleep(0.15)


if __name__ == "__main__":
    main()
