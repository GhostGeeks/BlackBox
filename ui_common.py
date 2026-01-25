# ~/oled/ui_common.py
from luma.core.render import canvas

LINE_H = 10
HEADER_H = 20  # 2 lines (title + divider)

def draw_header(d, title: str):
    d.text((0, 0), title, fill=255)
    d.text((0, 10), "────────────", fill=255)

def draw_row(d, y: int, text: str, selected: bool = False):
    if selected:
        d.rectangle((0, y, 127, y + 9), fill=255, outline=255)
        d.text((2, y), text, fill=0)
    else:
        d.text((2, y), text, fill=255)

def draw_row_lr(d, y: int, left: str, right: str, selected: bool = False, right_x: int = 80):
    """
    left at x=2, right aligned-ish by fixed x (works well for short values like '150ms').
    """
    if selected:
        d.rectangle((0, y, 127, y + 9), fill=255, outline=255)
        d.text((2, y), left, fill=0)
        d.text((right_x, y), right, fill=0)
    else:
        d.text((2, y), left, fill=255)
        d.text((right_x, y), right, fill=255)

def draw_centered(d, y: int, text: str, invert: bool = False):
    # Approx centering for default font (6px per char-ish)
    x = max(0, (128 - (len(text) * 6)) // 2)
    if invert:
        d.rectangle((0, y, 127, y + 11), fill=255, outline=255)
        d.text((x, y), text, fill=0)
    else:
        d.text((x, y), text, fill=255)

def render(device, draw_fn):
    with canvas(device) as d:
        draw_fn(d)
