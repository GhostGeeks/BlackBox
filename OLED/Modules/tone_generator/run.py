#!/usr/bin/env python3
"""
BlackBox Tone Generator (headless JSON stdout protocol) - v5 (Main 3 rows + Frequency Settings)

Main page:
- Frequency: <Value or Tone>
- Volume: <Value>
- Play: PLAY/STOP

Frequency settings page:
- Manual Frequency
- Special Frequency
- Special Tones

Manual Frequency edit page:
- UP/DOWN adjusts frequency
- HOLD = big step up
- SELECT = done -> main
- BACK = main

STRICT:
- No OLED access
- JSON-only stdout
- Non-blocking stdin
- Audio loops in one background process; STOP is immediate via killpg
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
MODULE_VERSION = "tg_v5_ui_ready"

PATTERN_WAV = "/tmp/blackbox_tone_pattern.wav"
AUDIO_ERR = "/tmp/blackbox_tone_audio.err"
MODULE_ERR = "/tmp/blackbox_tone_module.err"

RATE = 48000
CHANNELS = 1
SAMPWIDTH = 2

PATTERN_SECONDS_STD = 6.0
PATTERN_SECONDS_SWEEP = 8.0
PATTERN_SECONDS_SHEP = 10.0
PATTERN_SECONDS_MOTIF = 4.0

HEARTBEAT_S = 0.25
TOAST_MIN_INTERVAL_S = 0.10


# ---------------- Logging (file only) ----------------
def _log_err(msg: str) -> None:
    try:
        with open(MODULE_ERR, "a", encoding="utf-8") as f:
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")
    except Exception:
        pass


# ---------------- JSON stdout (STRICT) ----------------
def _emit(obj: dict) -> None:
    try:
        sys.stdout.write(json.dumps(obj, separators=(",", ":")) + "\n")
        sys.stdout.flush()
    except Exception:
        pass


def _hello() -> None:
    _emit({"type": "hello", "module": MODULE_NAME, "version": MODULE_VERSION})
    _emit({"type": "page", "name": "main"})


def _toast(msg: str) -> None:
    _emit({"type": "toast", "message": msg})


def _fatal(msg: str) -> None:
    _emit({"type": "fatal", "message": msg})


# ---------------- Helpers ----------------
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


# ---------------- Lists ----------------
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
    # Copyright-safe placeholders (not the film’s exact tones)
    ("contact_call", "Contact Call (Original)"),
    ("contact_resp", "Contact Response (Original)"),
]


# ---------------- WAV writer ----------------
def _write_wav_from_samples(samples_iter, seconds: float) -> None:
    total_frames = int(RATE * seconds)
    tmp_path = PATTERN_WAV + ".tmp"
    with wave.open(tmp_path, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(SAMPWIDTH)
        wf.setframerate(RATE)

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

    os.replace(tmp_path, PATTERN_WAV)


# ---------------- Sound generators ----------------
def _gen_standard_tone(freq_hz: int, volume: int) -> None:
    freq_hz = _clamp(int(freq_hz), 20, 20000)
    volume = _clamp(int(volume), 0, 100)
    amp = (volume / 100.0) * 0.95

    phase = 0.0
    phase_inc = float(freq_hz) / float(RATE)

    def it():
        nonlocal phase
        while True:
            phase += phase_inc
            yield math.sin(2.0 * math.pi * (phase % 1.0)) * amp

    _write_wav_from_samples(it(), PATTERN_SECONDS_STD)


def _gen_sweep(volume: int, direction: str) -> None:
    volume = _clamp(int(volume), 0, 100)
    amp = (volume / 100.0) * 0.85

    f0 = 20.0
    f1 = 20000.0
    dur = PATTERN_SECONDS_SWEEP
    total = int(RATE * dur)

    log_f0 = math.log(f0)
    log_f1 = math.log(f1)

    phase = 0.0

    def freq_at(u: float) -> float:
        return math.exp(log_f0 + (log_f1 - log_f0) * u)

    def it():
        nonlocal phase
        for n in range(total):
            t = n / float(total - 1 if total > 1 else 1)
            if direction == "asc":
                u = t
            elif direction == "des":
                u = 1.0 - t
            else:
                u = (t * 2.0) if t < 0.5 else ((1.0 - t) * 2.0)

            f = freq_at(u)
            phase += f / float(RATE)
            yield math.sin(2.0 * math.pi * (phase % 1.0)) * amp

        while True:
            yield 0.0

    _write_wav_from_samples(it(), dur)


def _gen_shepard(volume: int, direction: str) -> None:
    volume = _clamp(int(volume), 0, 100)
    amp = (volume / 100.0) * 0.75

    dur = PATTERN_SECONDS_SHEP
    total = int(RATE * dur)

    base = 55.0
    octaves = 8
    sigma = 1.2
    phases = [0.0 for _ in range(octaves)]

    def gauss(x: float) -> float:
        return math.exp(-0.5 * (x / sigma) ** 2)

    def it():
        for n in range(total):
            t = n / float(total - 1 if total > 1 else 1)
            pos = t if direction == "asc" else (1.0 - t)
            frac_oct = pos * 1.0

            ssum = 0.0
            wsum = 0.0
            for i in range(octaves):
                f = base * (2.0 ** (i + frac_oct))
                while f > 20000.0:
                    f *= 0.5
                while f < 20.0:
                    f *= 2.0

                x = (i + frac_oct) - (octaves / 2.0)
                w = gauss(x)

                phases[i] += f / float(RATE)
                ssum += math.sin(2.0 * math.pi * (phases[i] % 1.0)) * w
                wsum += w

            yield (ssum / wsum) * amp if wsum > 0 else 0.0

        while True:
            yield 0.0

    _write_wav_from_samples(it(), dur)


def _gen_contact_motif(volume: int, variant: str) -> None:
    # Original placeholder motif (NOT the movie)
    volume = _clamp(int(volume), 0, 100)
    amp = (volume / 100.0) * 0.85
    dur = PATTERN_SECONDS_MOTIF

    if variant == "call":
        notes = [392.0, 523.25, 659.25, 440.0, 587.33]
    else:
        notes = [587.33, 440.0, 659.25, 523.25, 392.0]

    tone_s = 0.35
    gap_s = 0.08

    seq = []
    for f in notes:
        seq.append(("tone", f, tone_s))
        seq.append(("gap", 0.0, gap_s))
    seq.append(("gap", 0.0, max(0.2, dur - sum(seg[2] for seg in seq))))

    phase = 0.0

    def it():
        nonlocal phase
        seg_idx = 0
        seg_left = int(RATE * seq[seg_idx][2])
        while True:
            typ, f, _secs = seq[seg_idx]
            if typ == "gap":
                yield 0.0
            else:
                phase += f / float(RATE)
                yield math.sin(2.0 * math.pi * (phase % 1.0)) * amp

            seg_left -= 1
            if seg_left <= 0:
                seg_idx = (seg_idx + 1) % len(seq)
                seg_left = int(RATE * seq[seg_idx][2])

    _write_wav_from_samples(it(), dur)


# ---------------- Player loop ----------------
def _which_player() -> Optional[str]:
    return shutil.which("paplay") or shutil.which("pw-play")


def _start_audio_loop(player_path: str) -> subprocess.Popen:
    try:
        open(AUDIO_ERR, "a").close()
    except Exception:
        pass

    cmd = 'exec 2>>"{err}"; while true; do "{player}" "{wav}"; done'.format(
        err=AUDIO_ERR, player=player_path, wav=PATTERN_WAV
    )

    return subprocess.Popen(
        ["/bin/sh", "-lc", cmd],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,  # new PGID
        close_fds=True,
        env=os.environ.copy(),
    )


def _stop_proc(p: Optional[subprocess.Popen]) -> None:
    if not p:
        return
    try:
        if p.poll() is None:
            try:
                os.killpg(p.pid, signal.SIGTERM)
            except Exception:
                p.terminate()
            try:
                p.wait(timeout=0.35)
            except Exception:
                pass
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


# ---------------- Non-blocking stdin ----------------
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


# ---------------- UI state ----------------
MAIN_ROWS = ["frequency", "volume", "play"]
FREQ_MENU_ROWS = ["manual", "special_freq", "special_tone"]


@dataclass
class UIState:
    page: str = "main"  # main|freq_menu|freq_edit|special_freqs|special_tones

    cursor_main: str = "frequency"
    cursor_freq_menu: str = "manual"
    idx_special_freq: int = 0
    idx_special_tone: int = 0

    freq_hz: int = 440
    volume: int = 70

    selection: str = "manual"         # manual|special_freq|special_tone
    selection_label: str = "440Hz"    # shown on main

    special_tone_id: str = ""         # only if selection==special_tone

    playing: bool = False
    ready: bool = False

    _need_audio_restart: bool = False
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


def _move_cursor(lst: List[str], cur: str, direction: int) -> str:
    try:
        i = lst.index(cur)
    except Exception:
        i = 0
    return lst[(i + direction) % len(lst)]


def _render_wav_for_selection(st: UIState) -> None:
    if st.selection == "special_tone":
        tid = st.special_tone_id
        if tid == "sweep_asc":
            _gen_sweep(st.volume, "asc")
        elif tid == "sweep_des":
            _gen_sweep(st.volume, "des")
        elif tid == "sweep_bell":
            _gen_sweep(st.volume, "bell")
        elif tid == "shepard_asc":
            _gen_shepard(st.volume, "asc")
        elif tid == "shepard_des":
            _gen_shepard(st.volume, "des")
        elif tid == "contact_call":
            _gen_contact_motif(st.volume, "call")
        elif tid == "contact_resp":
            _gen_contact_motif(st.volume, "resp")
        else:
            _gen_standard_tone(st.freq_hz, st.volume)
    else:
        _gen_standard_tone(st.freq_hz, st.volume)


# ---------------- Main loop ----------------
def main() -> int:
    exiting = {"flag": False}

    def _sig_handler(_signo, _frame):
        exiting["flag"] = True

    try:
        signal.signal(signal.SIGTERM, _sig_handler)
        signal.signal(signal.SIGINT, _sig_handler)
    except Exception:
        pass

    reader = StdinReader()
    st = UIState()
    st.selection_label = f"{st.freq_hz}Hz"

    _hello()

    player_path = _which_player()
    if not player_path:
        st.ready = False
        _emit_state(st)
        _fatal("Audio player not available (need paplay or pw-play)")
    else:
        st.ready = True
        _emit_state(st)

    audio_proc: Optional[subprocess.Popen] = None

    # initial wav (only to validate; doesn't auto-start playback)
    try:
        _render_wav_for_selection(st)
    except Exception as e:
        _log_err(f"WAV init failed: {e!r}")
        st.ready = False
        st.playing = False
        _emit_state(st)
        _fatal("Failed to initialize tone pattern")

    last_hb = 0.0

    def stop_audio():
        nonlocal audio_proc
        _stop_proc(audio_proc)
        audio_proc = None

    def restart_audio_if_needed():
        nonlocal audio_proc
        if not st._need_audio_restart:
            return
        st._need_audio_restart = False

        stop_audio()

        if st.playing and st.ready and player_path:
            try:
                _render_wav_for_selection(st)
            except Exception as e:
                _log_err(f"WAV regen failed: {e!r}")
                st.playing = False
                _emit_state(st)
                _fatal("Failed to generate tone pattern")
            else:
                audio_proc = _start_audio_loop(player_path)

    def go(page: str):
        st.page = page
        _emit_page(st)
        _emit_state(st)

    def back_to_main():
        go("main")

    # ------------- input handlers -------------
    def handle_main(cmd: str):
        if cmd == "up":
            st.cursor_main = _move_cursor(MAIN_ROWS, st.cursor_main, -1)
            _emit_state(st)
            return
        if cmd == "down":
            st.cursor_main = _move_cursor(MAIN_ROWS, st.cursor_main, +1)
            _emit_state(st)
            return
        if cmd == "select":
            if st.cursor_main == "frequency":
                go("freq_menu")
                return
            if st.cursor_main == "volume":
                st.volume = _clamp(st.volume + 5, 0, 100)
                st._need_audio_restart = True
                _toast_throttle(st, f"Vol: {st.volume}%")
                _emit_state(st)
                return
            if st.cursor_main == "play":
                st.playing = not st.playing
                st._need_audio_restart = True
                _toast_throttle(st, "PLAY" if st.playing else "STOP")
                _emit_state(st)
                return
        if cmd == "select_hold":
            if st.cursor_main == "volume":
                st.volume = _clamp(st.volume - 5, 0, 100)
                st._need_audio_restart = True
                _toast_throttle(st, f"Vol: {st.volume}%")
                _emit_state(st)
                return
            if st.cursor_main == "play":
                if st.playing:
                    st.playing = False
                    st._need_audio_restart = True
                    _toast_throttle(st, "STOP")
                    _emit_state(st)
                return
        if cmd == "back":
            st.playing = False
            st._need_audio_restart = True
            exiting["flag"] = True
            return

    def handle_freq_menu(cmd: str):
        if cmd == "up":
            st.cursor_freq_menu = _move_cursor(FREQ_MENU_ROWS, st.cursor_freq_menu, -1)
            _emit_state(st)
            return
        if cmd == "down":
            st.cursor_freq_menu = _move_cursor(FREQ_MENU_ROWS, st.cursor_freq_menu, +1)
            _emit_state(st)
            return
        if cmd == "select":
            if st.cursor_freq_menu == "manual":
                go("freq_edit")
                return
            if st.cursor_freq_menu == "special_freq":
                go("special_freqs")
                return
            if st.cursor_freq_menu == "special_tone":
                go("special_tones")
                return
        if cmd == "back":
            back_to_main()
            return

    def handle_freq_edit(cmd: str):
        if cmd == "up":
            st.freq_hz = _clamp(st.freq_hz + _freq_step(st.freq_hz), 20, 20000)
            st.selection = "manual"
            st.special_tone_id = ""
            st.selection_label = f"{st.freq_hz}Hz"
            st._need_audio_restart = True
            _toast_throttle(st, f"Freq: {st.freq_hz}Hz")
            _emit_state(st)
            return
        if cmd == "down":
            st.freq_hz = _clamp(st.freq_hz - _freq_step(st.freq_hz), 20, 20000)
            st.selection = "manual"
            st.special_tone_id = ""
            st.selection_label = f"{st.freq_hz}Hz"
            st._need_audio_restart = True
            _toast_throttle(st, f"Freq: {st.freq_hz}Hz")
            _emit_state(st)
            return
        if cmd == "select_hold":
            st.freq_hz = _clamp(st.freq_hz + _freq_big_step(st.freq_hz), 20, 20000)
            st.selection = "manual"
            st.special_tone_id = ""
            st.selection_label = f"{st.freq_hz}Hz"
            st._need_audio_restart = True
            _toast_throttle(st, f"Freq: {st.freq_hz}Hz")
            _emit_state(st)
            return
        if cmd == "select" or cmd == "back":
            back_to_main()
            return

    def handle_special_freqs(cmd: str):
        if cmd == "up":
            st.idx_special_freq = (st.idx_special_freq - 1) % len(SPECIAL_FREQS)
            _emit_state(st)
            return
        if cmd == "down":
            st.idx_special_freq = (st.idx_special_freq + 1) % len(SPECIAL_FREQS)
            _emit_state(st)
            return
        if cmd == "select":
            hz, name = SPECIAL_FREQS[st.idx_special_freq]
            st.freq_hz = int(hz)
            st.selection = "special_freq"
            st.special_tone_id = ""
            st.selection_label = f"{hz}Hz {name}"
            st._need_audio_restart = True
            _toast_throttle(st, st.selection_label[:21])
            back_to_main()
            return
        if cmd == "back":
            back_to_main()
            return

    def handle_special_tones(cmd: str):
        if cmd == "up":
            st.idx_special_tone = (st.idx_special_tone - 1) % len(SPECIAL_TONES)
            _emit_state(st)
            return
        if cmd == "down":
            st.idx_special_tone = (st.idx_special_tone + 1) % len(SPECIAL_TONES)
            _emit_state(st)
            return
        if cmd == "select":
            tid, label = SPECIAL_TONES[st.idx_special_tone]
            st.selection = "special_tone"
            st.special_tone_id = tid
            st.selection_label = label
            st._need_audio_restart = True
            _toast_throttle(st, label[:21])
            back_to_main()
            return
        if cmd == "back":
            back_to_main()
            return

    # ------------- loop -------------
    while True:
        now = time.monotonic()

        if exiting["flag"]:
            st.playing = False
            st._need_audio_restart = True

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

        # audio control
        try:
            restart_audio_if_needed()
            if st.playing and audio_proc and (audio_proc.poll() is not None):
                _log_err("Audio loop exited unexpectedly; restarting")
                st._need_audio_restart = True
        except Exception as e:
            _log_err(f"Audio mgmt exception: {e!r}")
            _fatal("Audio error; stopping playback")
            st.playing = False
            st._need_audio_restart = True
            _emit_state(st)

        # heartbeat
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
