elif is_tone:
    pump = None
    try:
        if proc.stdout is None:
            raise RuntimeError("tone_generator requires stdout=PIPE")
        pump = StdoutJSONPump(proc.stdout, log)
    except Exception as e:
        log(f"[launcher] pump_init_failed: {e!r}")
        oled_message("Tone Generator", ["Pump init failed", str(e)[:21], ""], "BACK")
        try:
            proc.terminate()
        except Exception:
            pass
        time.sleep(1.0)
        oled_hard_wake()
        return

    state: Dict[str, Any] = {
        "page": "main",
        "ready": False,
        "playing": False,
        "freq_hz": 440,
        "volume": 70,
        "selection": "manual",
        "selection_label": "440Hz",
        "special_tone": "",
        "cursor_main": "frequency",
        "cursor_freq_menu": "manual",
        "idx_special_freq": 0,
        "idx_special_tone": 0,
        "fatal": "",
        "toast": "",
        "toast_until": 0.0,
    }

    # For display
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

    MAIN_ROWS_UI = ["frequency", "volume", "play"]
    FREQ_MENU_ROWS_UI = ["manual", "special_freq", "special_tone"]

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

    # ------- Renderers -------
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
            for i, label in enumerate(window):
                selected = (start + i) == idx
                _draw_row(draw, y0 + i * row_h, label, selected)

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
        oled_message("Tone Generator", ["ERROR", msg, ""], "BACK")

    # ------- Loop -------
    while proc.poll() is None:
        msgs = pump.pump(max_bytes=65536, max_lines=160)
        if msgs:
            last_msg_time = time.time()

        exit_requested = False

        for msg in msgs:
            t = msg.get("type")

            if t == "page":
                state["page"] = msg.get("name", state.get("page", "main"))

            elif t == "state":
                for k in (
                    "page", "ready", "playing", "freq_hz", "volume",
                    "selection", "selection_label", "special_tone",
                    "cursor_main", "cursor_freq_menu",
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
        silent_s = now - last_msg_time

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
            for _ in range(25):
                if proc.poll() is not None:
                    break
                time.sleep(0.02)
            if proc.poll() is not None:
                break

        # forward buttons
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
            for _ in range(50):
                if proc.poll() is not None:
                    break
                time.sleep(0.02)
            if proc.poll() is None:
                try:
                    proc.terminate()
                except Exception:
                    pass
            break

        if silent_s > 15.0:
            log("[launcher] watchdog: tone_generator silent >15s; terminating")
            try:
                proc.terminate()
            except Exception:
                pass
            break

        time.sleep(0.02)

    try:
        if pump:
            pump.close()
    except Exception:
        pass
