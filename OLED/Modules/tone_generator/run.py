#!/usr/bin/env python3
"""
BlackBox Tone Generator (headless JSON stdout protocol) - v8 (prebaked Shepard assets, no build UI)

- NO OLED access
- JSON-only stdout
- Non-blocking stdin
- Continuous output via paplay loop in its own process group
- STOP is immediate (killpg TERM then KILL)
- Shepard Asc/Des use prebaked 5-minute WAV assets on disk (no generation / no build page)

Pages:
- main
- freq_menu
- freq_edit
- special_freqs
- special_tones
"""

import os
import sys
import json
import time
import math
import wave
import signal
import shutil
import selectors
import subprocess
from dataclasses import dataclass
from typing import Optional, List, Tuple


MODULE_NAME = "tone_generator"
MODULE_VERSION = "tg_v8_prebaked_shepard"

AUDIO_ERR = "/tmp/blackbox_tone_audio.err"
MODULE_ERR = "/tmp/blackbox_tone_module.err"

# Standard tone generation (manual/special frequency)
RATE_STD = 48000
CHANNELS = 1
SAMPWIDTH = 2
DUR_STD = 6.0

HEARTBEAT_S = 0.25
TOAST_MIN_INTERVAL_S = 0.10

MAIN_ROWS = ["frequency", "volume", "play"]
FREQ_MENU_ROWS = ["manual", "special_freq", "special_tone"]

# File locations
MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSET_DIR = os.path.join(MODULE_DIR, "assets")
STD_WAV = "/tmp/blackbox_tone_standard.wav"

# Prebaked Shepard assets (5 minutes each). You will generate these once and commit them.
SHEP_ASC_WAV = os.path.join(ASSET_DIR, "shepard_asc_5m.wav")
SHEP_DES_WAV = os.path.join(ASSET_DIR, "shepard_des_5m.wav")

SPECIAL_FREQS: List[Tuple[int, str]] = [
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

SPECIAL_TONES: List[Tuple[str, str]] = [
    ("sweep_asc", "Frequency Sweep Asc"),
    ("sweep_des", "Frequency Sweep Des"),
    ("sweep_bell", "Frequency Sweep Bell"),
    ("shepard_asc", "Shepard Tone Asc"),
    ("shepard_des", "Shepard Tone Des"),
    # copyright-safe placeholders (not the film’s exact tones)
    ("contact_call", "Contact Call (Original)"),
    ("contact_resp", "Contact Response (Original)"),
]


# ---------------- logging (file only) ----------------
def _log_err(msg: str) -> None:
    try:
        with open(MODULE_ERR, "a", encoding="utf-8") as f:
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")
    except Exception:
        pass


# ---------------- strict JSON stdout ----------------
def _emit(obj: dict) -> None:
    try:
        sys.stdout.write(json.dumps(obj, separators=(",", ":")) + "\n")
        sys.stdout.flush()
    except Exception:
        pass


def _toast(msg: str) -> None:
    _emit({"type": "toast", "message": msg})


def _fatal(msg: str) -> None:
    _emit({"type": "fatal", "message": msg})


def _clamp(v: int, lo: int, hi: int) -> int:
    return lo if v < lo else hi if v > hi else v


def _freq_step(freq: int) -> int:
    if freq < 200:
        return 5
    if freq <= 2000:
        return 10
    return 50


def _freq_big_step(freq: int) -> int:
    if freq < 200:
        return 25
    if freq <= 2000:
        return 100
    return 500


def _move_cursor(lst: List[str], cur: str, direction: int) -> str:
    try:
        i = lst.index(cur)
    except Exception:
        i = 0
    return lst[(i + direction) % len(lst)]


def _pulse_volume_16bit(volume_pct: int) -> int:
    # PulseAudio pa_volume_t: PA_VOLUME_NORM = 0x10000 = 65536
    v = _clamp(int(volume_pct), 0, 100)
    return int(65536 * (v / 100.0))


# ---------------- stdin non-blocking ----------------
class StdinReader:
    def __init__(self):
        self.sel = selectors.DefaultSelector()
        self.fd = sys.stdin.fileno()
        try:
            os.set_blocking(self.fd, False)
        except Exception:
            pass
        self.sel.register(self.fd, selectors.EVENT_READ)
        self.buf = b""

    def poll_lines(self, timeout: float = 0.0) -> List[str]:
        out: List[str] = []
        try:
            events = self.sel.select(timeout)
        except Exception:
            return out

        for key, _ in events:
            if key.fd != self.fd:
                continue
            try:
                chunk = os.read(self.fd, 4096)
            except BlockingIOError:
                continue
            except Exception:
                continue
            if not chunk:
                continue
            self.buf += chunk

        while b"\n" in self.buf:
            line, self.buf = self.buf.split(b"\n", 1)
            s = line.decode("utf-8", errors="ignore").strip().lower()
            if s:
                out.append(s)
        return out


# ---------------- WAV writer (standard sine only) ----------------
def _write_wav_from_samples(path: str, samples_iter, seconds: float, rate: int) -> None:
    total_frames = int(rate * seconds)
    tmp_path = path + ".tmp"
    with wave.open(tmp_path, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(SAMPWIDTH)
        wf.setframerate(rate)

        block = bytearray()
        block_target_frames = 2048
        i = 0

        for s in samples_iter:
            v = int(max(-1.0, min(1.0, s)) * 32767)
            block += int(v).to_bytes(2, byteorder="little", signed=True)
            i += 1

            if len(block) >= block_target_frames * 2:
                wf.writeframes(block)
                block.clear()

            if i >= total_frames:
                break

        if block:
            wf.writeframes(block)

    os.replace(tmp_path, path)


def _gen_standard_sine(path: str, freq_hz: int) -> None:
    freq_hz = _clamp(int(freq_hz), 20, 20000)
    amp = 0.90
    phase = 0.0
    inc = float(freq_hz) / float(RATE_STD)

    def it():
        nonlocal phase
        while True:
            phase += inc
            yield math.sin(2.0 * math.pi * (phase % 1.0)) * amp

    _write_wav_from_samples(path, it(), DUR_STD, RATE_STD)


# ---------------- player loop (immediate stop) ----------------
def _which_player() -> Optional[str]:
    # Prefer paplay for per-stream volume
    return shutil.which("paplay") or shutil.which("pw-play")


def _start_audio_loop(player_path: str, wav_path: str, volume_pct: int) -> subprocess.Popen:
    try:
        open(AUDIO_ERR, "a").close()
    except Exception:
        pass

    if os.path.basename(player_path) == "paplay":
        vol16 = _pulse_volume_16bit(volume_pct)
        play_cmd = f'"{player_path}" --volume={vol16} "{wav_path}"'
    else:
        play_cmd = f'"{player_path}" "{wav_path}"'

    # Loop forever inside one PGID so stop is immediate via killpg
    cmd = f'exec 2>>"{AUDIO_ERR}"; while true; do {play_cmd}; done'
    return subprocess.Popen(
        ["/bin/sh", "-lc", cmd],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
        close_fds=True,
        env=os.environ.copy(),
    )


def _stop_proc(p: Optional[subprocess.Popen]) -> None:
    if not p:
        return
    try:
        if p.poll() is None:
            # TERM whole process group (shell + paplay)
            try:
                os.killpg(p.pid, signal.SIGTERM)
            except Exception:
                try:
                    p.terminate()
                except Exception:
                    pass
            # short wait, then KILL if needed
            t0 = time.time()
            while time.time() - t0 < 0.20:
                if p.poll() is not None:
                    break
                time.sleep(0.01)
        if p.poll() is None:
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except Exception:
                try:
                    p.kill()
                except Exception:
                    pass
    except Exception:
        pass


def _is_valid_wav(path: str) -> bool:
    try:
        if not os.path.exists(path) or os.path.getsize(path) < 2048:
            return False
        with wave.open(path, "rb") as wf:
            _ = wf.getnchannels()
            _ = wf.getframerate()
            _ = wf.getnframes()
        return True
    except Exception:
        return False


# ---------------- state ----------------
@dataclass
class UIState:
    page: str = "main"  # main|freq_menu|freq_edit|special_freqs|special_tones
    cursor_main: str = "frequency"
    cursor_freq_menu: str = "manual"
    idx_special_freq: int = 0
    idx_special_tone: int = 0

    freq_hz: int = 440
    volume: int = 70

    selection: str = "manual"          # manual|special_freq|special_tone
    selection_label: str = "440Hz"
    special_tone_id: str = ""

    playing: bool = False
    ready: bool = False

    _last_toast_t: float = 0.0


def _emit_page(st: UIState) -> None:
    _emit({"type": "page", "name": st.page})


def _emit_state(st: UIState) -> None:
    _emit({
        "type": "state",
        "page": st.page,
        "ready": bool(st.ready),
        "playing": bool(st.playing),
        "freq_hz": int(st.freq_hz),
        "volume": int(st.volume),
        "selection": st.selection,
        "selection_label": st.selection_label,
        "special_tone": st.special_tone_id,
        "cursor_main": st.cursor_main,
        "cursor_freq_menu": st.cursor_freq_menu,
        "idx_special_freq": int(st.idx_special_freq),
        "idx_special_tone": int(st.idx_special_tone),
    })


def _toast_throttle(st: UIState, msg: str) -> None:
    now = time.monotonic()
    if now - st._last_toast_t >= TOAST_MIN_INTERVAL_S:
        st._last_toast_t = now
        _toast(msg)


def _selected_wav_path(st: UIState) -> str:
    if st.selection == "special_tone":
        if st.special_tone_id == "shepard_asc":
            return SHEP_ASC_WAV
        if st.special_tone_id == "shepard_des":
            return SHEP_DES_WAV
        # other special tones still generated dynamically if you keep them (optional)
        # For now they play as standard sine at current freq unless you later add baked assets for them too.
        return STD_WAV
    return STD_WAV


def main() -> int:
    exiting = {"flag": False}

    def _sig_handler(_signo, _frame):
        exiting["flag"] = True

    try:
        signal.signal(signal.SIGTERM, _sig_handler)
        signal.signal(signal.SIGINT, _sig_handler)
    except Exception:
        pass

    # Ensure assets folder exists (for prebaked WAVs)
    try:
        os.makedirs(ASSET_DIR, exist_ok=True)
    except Exception:
        pass

    reader = StdinReader()
    st = UIState()
    st.selection_label = f"{st.freq_hz}Hz"

    _emit({"type": "hello", "module": MODULE_NAME, "version": MODULE_VERSION})
    _emit_page(st)

    player_path = _which_player()
    if not player_path:
        st.ready = False
        _emit_state(st)
        _fatal("Audio player not available (need paplay or pw-play)")
    else:
        st.ready = True
        _emit_state(st)

    # Validate prebaked Shepard assets. If missing, show fatal but keep running (so UI doesn’t bounce).
    if not _is_valid_wav(SHEP_ASC_WAV) or not _is_valid_wav(SHEP_DES_WAV):
        _log_err("Missing/invalid Shepard assets. Generate: assets/shepard_asc_5m.wav and shepard_des_5m.wav")
        _toast("Missing Shepard WAVs")
        # Keep module alive; Shepard options will toast error on attempt.

    audio_proc: Optional[subprocess.Popen] = None
    last_hb = 0.0

    def stop_audio():
        nonlocal audio_proc
        _stop_proc(audio_proc)
        audio_proc = None

    def start_audio():
        nonlocal audio_proc
        stop_audio()

        wav = _selected_wav_path(st)

        # If user selected Shepard but assets missing, refuse to play and toast.
        if st.selection == "special_tone" and st.special_tone_id in ("shepard_asc", "shepard_des"):
            if not _is_valid_wav(wav):
                st.playing = False
                _emit_state(st)
                _toast("Shepard WAV missing")
                return

        # For standard tone we (re)generate quickly
        if st.selection != "special_tone":
            try:
                _gen_standard_sine(STD_WAV, st.freq_hz)
            except Exception as e:
                _log_err(f"Std tone build failed: {e!r}")
                st.playing = False
                _emit_state(st)
                _fatal("Failed to generate tone")
                return
            wav = STD_WAV
        else:
            # Non-shepard specials currently fall back to std wav (you can add their own assets later)
            if wav == STD_WAV:
                try:
                    _gen_standard_sine(STD_WAV, st.freq_hz)
                except Exception as e:
                    _log_err(f"Fallback tone build failed: {e!r}")
                    st.playing = False
                    _emit_state(st)
                    _fatal("Failed to generate tone")
                    return

        audio_proc = _start_audio_loop(player_path, wav, st.volume)

    def go(page: str):
        st.page = page
        _emit_page(st)
        _emit_state(st)

    def back_to_main():
        go("main")

    # ---------- input handlers ----------
    def handle_main(cmd: str):
        if cmd == "up":
            st.cursor_main = _move_cursor(MAIN_ROWS, st.cursor_main, -1)
            _emit_state(st); return
        if cmd == "down":
            st.cursor_main = _move_cursor(MAIN_ROWS, st.cursor_main, +1)
            _emit_state(st); return

        if cmd == "select":
            if st.cursor_main == "frequency":
                go("freq_menu"); return
            if st.cursor_main == "volume":
                st.volume = _clamp(st.volume + 5, 0, 100)
                _toast_throttle(st, f"Vol: {st.volume}%")
                _emit_state(st)
                if st.playing:
                    start_audio()
                return
            if st.cursor_main == "play":
                st.playing = not st.playing
                _toast_throttle(st, "PLAY" if st.playing else "STOP")
                _emit_state(st)
                if st.playing:
                    start_audio()
                else:
                    stop_audio()
                return

        if cmd == "select_hold":
            if st.cursor_main == "volume":
                st.volume = _clamp(st.volume - 5, 0, 100)
                _toast_throttle(st, f"Vol: {st.volume}%")
                _emit_state(st)
                if st.playing:
                    start_audio()
                return
            if st.cursor_main == "play":
                if st.playing:
                    st.playing = False
                    _toast_throttle(st, "STOP")
                    _emit_state(st)
                    stop_audio()
                return

        if cmd == "back":
            st.playing = False
            _emit_state(st)
            stop_audio()
            exiting["flag"] = True
            return

    def handle_freq_menu(cmd: str):
        if cmd == "up":
            st.cursor_freq_menu = _move_cursor(FREQ_MENU_ROWS, st.cursor_freq_menu, -1)
            _emit_state(st); return
        if cmd == "down":
            st.cursor_freq_menu = _move_cursor(FREQ_MENU_ROWS, st.cursor_freq_menu, +1)
            _emit_state(st); return
        if cmd == "select":
            if st.cursor_freq_menu == "manual":
                go("freq_edit"); return
            if st.cursor_freq_menu == "special_freq":
                go("special_freqs"); return
            if st.cursor_freq_menu == "special_tone":
                go("special_tones"); return
        if cmd == "back":
            back_to_main(); return

    def handle_freq_edit(cmd: str):
        if cmd == "up":
            st.freq_hz = _clamp(st.freq_hz + _freq_step(st.freq_hz), 20, 20000)
            st.selection = "manual"
            st.special_tone_id = ""
            st.selection_label = f"{st.freq_hz}Hz"
            _toast_throttle(st, f"Freq: {st.freq_hz}Hz")
            _emit_state(st)
            if st.playing:
                start_audio()
            return
        if cmd == "down":
            st.freq_hz = _clamp(st.freq_hz - _freq_step(st.freq_hz), 20, 20000)
            st.selection = "manual"
            st.special_tone_id = ""
            st.selection_label = f"{st.freq_hz}Hz"
            _toast_throttle(st, f"Freq: {st.freq_hz}Hz")
            _emit_state(st)
            if st.playing:
                start_audio()
            return
        if cmd == "select_hold":
            st.freq_hz = _clamp(st.freq_hz + _freq_big_step(st.freq_hz), 20, 20000)
            st.selection = "manual"
            st.special_tone_id = ""
            st.selection_label = f"{st.freq_hz}Hz"
            _toast_throttle(st, f"Freq: {st.freq_hz}Hz")
            _emit_state(st)
            if st.playing:
                start_audio()
            return
        if cmd == "select" or cmd == "back":
            back_to_main(); return

    def handle_special_freqs(cmd: str):
        if cmd == "up":
            st.idx_special_freq = (st.idx_special_freq - 1) % len(SPECIAL_FREQS)
            _emit_state(st); return
        if cmd == "down":
            st.idx_special_freq = (st.idx_special_freq + 1) % len(SPECIAL_FREQS)
            _emit_state(st); return
        if cmd == "select":
            hz, name = SPECIAL_FREQS[st.idx_special_freq]
            st.freq_hz = int(hz)
            st.selection = "special_freq"
            st.special_tone_id = ""
            st.selection_label = f"{hz}Hz {name}"
            _toast_throttle(st, st.selection_label[:21])
            _emit_state(st)
            back_to_main()
            if st.playing:
                start_audio()
            return
        if cmd == "back":
            back_to_main(); return

    def handle_special_tones(cmd: str):
        if cmd == "up":
            st.idx_special_tone = (st.idx_special_tone - 1) % len(SPECIAL_TONES)
            _emit_state(st); return
        if cmd == "down":
            st.idx_special_tone = (st.idx_special_tone + 1) % len(SPECIAL_TONES)
            _emit_state(st); return
        if cmd == "select":
            tid, label = SPECIAL_TONES[st.idx_special_tone]
            st.selection = "special_tone"
            st.special_tone_id = tid
            st.selection_label = label
            _toast_throttle(st, label[:21])
            _emit_state(st)
            back_to_main()
            if st.playing:
                start_audio()
            return
        if cmd == "back":
            back_to_main(); return

    # ---------- loop ----------
    while True:
        now = time.monotonic()

        for cmd in reader.poll_lines(timeout=0.0):
            if st.page == "main":
                handle_main(cmd)
            elif st.page == "freq_menu":
                handle_freq_menu(cmd)
            elif st.page == "freq_edit":
                handle_freq_edit(cmd)
            elif st.page == "special_freqs":
                handle_special_freqs(cmd)
            elif st.page == "special_tones":
                handle_special_tones(cmd)
            else:
                back_to_main()

        # If audio loop died unexpectedly, stop playing
        if st.playing and audio_proc and (audio_proc.poll() is not None):
            _log_err("Audio loop exited unexpectedly")
            st.playing = False
            _emit_state(st)
            _toast("Audio stopped")
            stop_audio()

        # Heartbeat
        if now - last_hb >= HEARTBEAT_S:
            last_hb = now
            _emit_state(st)

        if exiting["flag"]:
            stop_audio()
            _emit({"type": "exit"})
            return 0

        time.sleep(0.01)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as e:
        _log_err(f"FATAL unhandled: {e!r}")
        _fatal("Unhandled error in tone generator")
        try:
            _emit({"type": "exit"})
        except Exception:
            pass
        raise SystemExit(1)
