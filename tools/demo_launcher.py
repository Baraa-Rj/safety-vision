"""سلامتك — demo scenario launcher.

A small control panel for presenting the system: three buttons, one per
recorded scenario (PPE violations, fall detection, zone breach). Clicking a
button plays that clip through the REAL pipeline (python3 -m src.main with
CAMERA_RTSP_URL pointed at the file), so everything on screen — detection,
compliance states, zone polygons, alerts — is the live system, not a recording
of it.

Built on OpenCV + PIL (no tkinter/Qt on this machine); PIL is compiled with
libraqm, which shapes the Arabic logo correctly.

Usage:
    python3 tools/demo_launcher.py
    python3 tools/demo_launcher.py --render-only ui.png   # save the UI image and exit

One scenario runs at a time; press q in the video window to end it (or the
launcher will re-enable the buttons when the clip finishes). q/Esc in the
launcher quits and stops any running scenario.
"""
import argparse
import os
import subprocess
import sys

import cv2
import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFont

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LOGO_PATH = os.path.join(_ROOT, "data", "assets", "salamtak_logo.png")


def _env_file():
    """KEY=VALUE pairs from the gitignored .env (backend address lives there,
    never in source). Explicitly-exported variables win over the file."""
    env = {}
    path = os.path.join(_ROOT, ".env")
    if os.path.exists(path):
        for line in open(path):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()
    env.update(os.environ)
    return env
_CLIPS = "/home/dark/ppe_finetune_dataset/drive_folder/records from camira"

# (title, arabic label, clip, accent color, zones on?) — the restricted-zone
# polygon is only relevant to the zone-breach scenario; the other clips run
# with ZONES_ENABLED=0 so no zone is drawn or checked.
SCENARIOS = [
    ("PPE Violations", "مخالفات معدات الحماية",
     f"{_CLIPS}/001-20260707220200355.mp4", (217, 119, 6), False),   # amber
    ("Fall Detection", "كشف السقوط",
     f"{_CLIPS}/001-20260707222316677.mp4", (220, 38, 38), False),   # red
    ("Zone Breach", "اختراق المنطقة المحظورة",
     f"{_CLIPS}/001-20260707225316918.mp4", (37, 99, 235), True),    # blue
]

W, H = 520, 640
BG = (16, 24, 32)
CARD = (30, 41, 55)
CARD_HOVER = (45, 60, 80)
FG = (240, 244, 248)
ACCENT = (22, 163, 74)        # safety green
_AR_BOLD = "/usr/share/fonts/truetype/noto/NotoSansArabic-Bold.ttf"
_AR_REG = "/usr/share/fonts/truetype/noto/NotoSansArabic-Regular.ttf"
_LAT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
_LAT_REG = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

BTN_X, BTN_W, BTN_H, BTN_GAP = 40, W - 80, 96, 24
BTN_Y0 = 236
LOGO_BLUE = (66, 133, 181)     # sampled from the logo artwork


def _load_logo(width=300):
    """The artwork is RGBA on a huge transparent canvas -> crop to content."""
    im = Image.open(_LOGO_PATH).convert("RGBA")
    im = im.crop(im.getchannel("A").getbbox())
    h = int(im.height * width / im.width)
    return im.resize((width, h), Image.LANCZOS)


_LOGO = _load_logo()


def _button_rect(i):
    y = BTN_Y0 + i * (BTN_H + BTN_GAP)
    return BTN_X, y, BTN_X + BTN_W, y + BTN_H


def render(hover=-1, running=-1):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)

    # Header: the Salamtak logo artwork + the name in English.
    img.paste(_LOGO, ((W - _LOGO.width) // 2, 24), _LOGO)
    name_font = ImageFont.truetype(_LAT_BOLD, 30)
    d.text((W // 2, 24 + _LOGO.height + 26), "Salamtak", font=name_font,
           fill=LOGO_BLUE, anchor="mm")
    sub_font = ImageFont.truetype(_LAT_REG, 16)
    d.text((W // 2, 24 + _LOGO.height + 56), "Safety Vision — Demo Scenarios",
           font=sub_font, fill=(148, 163, 184), anchor="mm")
    d.line([(BTN_X, BTN_Y0 - 32), (W - BTN_X, BTN_Y0 - 32)],
           fill=(51, 65, 85), width=1)

    title_font = ImageFont.truetype(_LAT_BOLD, 24)
    ar_font = ImageFont.truetype(_AR_REG, 20)
    tag_font = ImageFont.truetype(_LAT_REG, 15)

    for i, (title, ar, _path, color, _zones) in enumerate(SCENARIOS):
        x1, y1, x2, y2 = _button_rect(i)
        fill = CARD_HOVER if i == hover and running < 0 else CARD
        d.rounded_rectangle([x1, y1, x2, y2], radius=14, fill=fill,
                            outline=color if i == running else (51, 65, 85),
                            width=2)
        d.rounded_rectangle([x1, y1, x1 + 10, y2], radius=5, fill=color)
        d.text((x1 + 30, y1 + 32), title, font=title_font, fill=FG, anchor="lm")
        if i != running:   # the RUNNING text below needs the full row width
            d.text((x2 - 24, y2 - 26), ar, font=ar_font, fill=(148, 163, 184),
                   anchor="rm", direction="rtl", language="ar")
        state = ("RUNNING — press q in the video window to stop"
                 if i == running else "click to run")
        d.text((x1 + 30, y2 - 24), state, font=tag_font,
               fill=color if i == running else (100, 116, 139), anchor="lm")

    foot_font = ImageFont.truetype(_LAT_REG, 14)
    d.text((W // 2, H - 28),
           "one scenario at a time  ·  q / Esc quits the launcher",
           font=foot_font, fill=(100, 116, 139), anchor="mm")
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--render-only", metavar="PNG",
                    help="save the UI image and exit (no window)")
    args = ap.parse_args()
    if args.render_only:
        cv2.imwrite(args.render_only, render())
        print(f"UI rendered to {args.render_only}")
        return

    missing = [p for _, _, p, _, _ in SCENARIOS if not os.path.exists(p)]
    if missing:
        sys.exit("missing scenario video(s):\n  " + "\n  ".join(missing))

    state = {"hover": -1, "click": None}

    def on_mouse(event, x, y, flags, _):
        state["hover"] = -1
        for i in range(len(SCENARIOS)):
            x1, y1, x2, y2 = _button_rect(i)
            if x1 <= x <= x2 and y1 <= y <= y2:
                state["hover"] = i
                if event == cv2.EVENT_LBUTTONDOWN:
                    state["click"] = i

    # ASCII only: OpenCV's Qt backend can't resolve non-ASCII window names
    # (setMouseCallback fails with a NULL window handler).
    win = "Salamatak - Safety Vision"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)

    proc = None
    running = -1
    while True:
        if proc is not None and proc.poll() is not None:
            proc, running = None, -1

        clicked = state.pop("click", None)
        if clicked is not None and proc is None:
            env = dict(_env_file(), CAMERA_RTSP_URL=SCENARIOS[clicked][2],
                       ZONES_ENABLED="1" if SCENARIOS[clicked][4] else "0")
            # Inherit stdout/stderr: [ALERT SENT]/[ALERT FAILED] lines from the
            # pipeline show in the launcher's terminal for verification.
            proc = subprocess.Popen(
                [sys.executable, "-m", "src.main"], cwd=_ROOT, env=env)
            running = clicked

        cv2.imshow(win, render(hover=state["hover"], running=running))
        key = cv2.waitKey(50) & 0xFF
        if key in (ord("q"), 27):
            break

    if proc is not None and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
