"""Image operations for the regenerated §4.1.3 and §4.2 benchmark."""
from io import BytesIO

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

MAIN = ("clean", "jpeg50", "jpeg25", "crop75", "crop50", "rotation30", "rotation75",
        "crop75+jpeg50", "crop75+jpeg25", "crop50+jpeg50", "crop50+jpeg25",
        "rotation30+jpeg50", "rotation30+jpeg25", "rotation75+jpeg50", "rotation75+jpeg25",
        "rotation30+crop75", "rotation30+crop50", "rotation75+crop75", "rotation75+crop50")
EXTRA = tuple(f"crop{area}-{location}" for area in (50, 75)
              for location in ("random", "upper_left", "upper_right", "lower_left",
                               "lower_right", "offcenter", "asymmetric"))
EXTRA += ("blur2", "noise005", "brightness15")


def attack(image, condition, seed=0):
    """Apply operations in their listed order, with per-image attack randomness."""
    rng = np.random.default_rng(seed)
    for operation in condition.split("+"):
        image = apply(image, operation, rng)
    return image


def apply(image, op, rng):
    if op == "clean":
        return image
    if op.startswith("jpeg"):
        with BytesIO() as buffer:
            image.save(buffer, format="JPEG", quality=int(op[4:]))
            buffer.seek(0)
            with Image.open(buffer) as decoded:
                return decoded.convert("RGB")
    if op.startswith("rotation"):
        return image.rotate(int(op[8:]), resample=Image.Resampling.BICUBIC,
                            expand=False, fillcolor=(0, 0, 0))
    if op.startswith("crop"):
        label, _, place = op[4:].partition("-")
        return crop(image, int(label)/100, place, rng)
    if op == "blur2":
        return image.filter(ImageFilter.GaussianBlur(2))
    if op == "noise005":
        src = np.asarray(image, dtype=np.float64)/255
        result = np.clip(src + rng.normal(0, 0.05, src.shape), 0, 1)
        return Image.fromarray(np.rint(result*255).astype(np.uint8))
    if op == "brightness15":
        return ImageEnhance.Brightness(image).enhance(1.5)
    raise ValueError(f"Unknown image operation: {op}")


def crop(image, area, place, rng):
    width, height = image.size
    shape = (1, area) if place == "asymmetric" else (area**0.5, area**0.5)
    w, h = max(1, round(width*shape[0])), max(1, round(height*shape[1]))
    dx, dy = width-w, height-h
    offsets = {
        "": (dx//2, dy//2), "upper_left": (0, 0), "upper_right": (dx, 0),
        "lower_left": (0, dy), "lower_right": (dx, dy),
        "offcenter": (dx//4, 3*dy//4), "asymmetric": (0, dy//2),
    }
    if place == "random":
        x, y = int(rng.integers(0, dx+1)), int(rng.integers(0, dy+1))
    else:
        x, y = offsets[place]
    return image.crop((x, y, x+w, y+h)).resize(image.size, Image.Resampling.BICUBIC)
