"""Make the README's demo GIF from the live Space's real output (Phase 8).

Sends one image to https://huggingface.co/spaces/addy0606/zsad, then turns what comes back into frames: the
photo, its ground-truth mask (if given), the fused map and each detector's map, captioned with that method's
verdict from the app's own score table.

    python report/demo_gif.py --image <png> --mask <mask png>   # writes report/figures/demo.gif
Needs the Space to be running and network access; not part of scripts/reproduce.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "report" / "figures" / "demo.gif"
SPACE = "addy0606/zsad"
SIDE, BAR = 480, 64  # frame: SIDE x SIDE image + caption bar
INK, SURFACE = (11, 11, 11), (252, 252, 251)


def font(size: int) -> ImageFont.FreeTypeFont:
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def frame(img: Image.Image, title: str, subtitle: str = "") -> Image.Image:
    img = img.convert("RGB")
    img.thumbnail((SIDE, SIDE))
    out = Image.new("RGB", (SIDE, SIDE + BAR), SURFACE)
    out.paste(img, ((SIDE - img.width) // 2, (SIDE - img.height) // 2))
    d = ImageDraw.Draw(out)
    d.text((14, SIDE + 10), title, fill=INK, font=font(17))
    d.text((14, SIDE + 36), subtitle, fill=(82, 81, 78), font=font(14))
    return out


def mask_overlay(image: Image.Image, mask: Image.Image) -> Image.Image:
    """Ground-truth defect outlined in red over the photo."""
    m = np.asarray(mask.convert("L").resize(image.size)) > 127
    edge = m ^ np.asarray(Image.fromarray(m).resize((m.shape[1] // 4, m.shape[0] // 4)).resize(image.size))
    arr = np.asarray(image.convert("RGB")).copy()
    arr[m] = (0.6 * arr[m] + 0.4 * np.array([230, 0, 0])).astype(np.uint8)
    arr[edge & m] = (230, 0, 0)
    return Image.fromarray(arr)


def verdicts(table: str) -> dict[str, str]:
    """Method label -> 'defect flagged (0.97 > threshold 0.53)' from the app's markdown table."""
    out = {}
    for line in table.splitlines():
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) == 6 and cells[1][:1].isdigit() or (len(cells) == 6 and cells[1].startswith("-")):
            label, score, thr, verdict = cells[0], cells[1], cells[2], cells[3].replace("*", "")
            sign = ">" if "flagged" in verdict else "<="
            out[label] = f"{verdict} (score {score} {sign} threshold {thr})"
    return out


def gallery_path(item) -> str:
    img = item["image"] if isinstance(item, dict) else item[0]
    return img["path"] if isinstance(img, dict) else img


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--mask", default=None)
    p.add_argument("--title", default="MVTec AD bottle, broken rim (test image)")
    a = p.parse_args()

    from gradio_client import Client, handle_file

    gallery, table = Client(SPACE, verbose=False).predict(
        handle_file(a.image), "bottle", "industrial", False, api_name="/analyse"
    )
    v = verdicts(table)
    photo = Image.open(a.image)
    frames = [(frame(photo, "Input photo", a.title), 1600)]
    if a.mask:
        frames.append(
            (
                frame(
                    mask_overlay(photo, Image.open(a.mask)),
                    "Ground truth (not shown to any method)",
                    "the broken part, from the dataset's mask",
                ),
                1600,
            )
        )
    for item in gallery:
        caption = item["caption"] if isinstance(item, dict) else item[1]
        label = caption.split(":")[0]
        sub = next((s for k, s in v.items() if k.startswith(label.split(" (")[0])), "")
        if caption.startswith("Fused"):
            sub = "average of the four normalised maps"
        frames.append((frame(Image.open(gallery_path(item)), caption.split(" (")[0], sub), 1500))
    imgs, durations = zip(*frames, strict=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    imgs[0].save(OUT, save_all=True, append_images=list(imgs[1:]), duration=list(durations), loop=0, optimize=True)
    print(f"wrote {OUT} ({len(imgs)} frames, {OUT.stat().st_size // 1024} KB)")
    print(table)


if __name__ == "__main__":
    main()
