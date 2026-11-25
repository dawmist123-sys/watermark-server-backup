"""HVS-GM core for Eqs. (4), (10), and (12)–(17) of the revised manuscript."""
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter
from scipy.special import expit

PROTOCOL = json.loads(Path(__file__).with_name("protocol.json").read_text(encoding="utf-8"))


def fft(x):
    return np.fft.fftshift(np.fft.fft2(x, axes=(-2, -1)), axes=(-2, -1))


def ifft(x):
    return np.fft.ifft2(np.fft.ifftshift(x, axes=(-2, -1)), axes=(-2, -1)).real


def latent(x):
    array = np.asarray(x, dtype=np.float64)
    if array.shape != tuple(PROTOCOL["latent_shape"]) or not np.isfinite(array).all():
        raise ValueError("Expected a finite latent of shape (4, 64, 64)")
    return array


def strength(feature, spec=PROTOCOL):
    x = latent(feature)
    w = spec["local_window"]
    local_mean = uniform_filter(x, (1, w, w), mode="reflect")
    local_mean_square = uniform_filter(x * x, (1, w, w), mode="reflect")
    texture = np.sqrt(np.maximum(local_mean_square - local_mean**2, 0)).mean(axis=0)
    scale = np.quantile(texture, spec["texture_quantile"])
    u = np.clip(texture / scale, 0, 1) if scale > 0 else np.zeros_like(texture)
    low, high = spec["strength_min"], spec["strength_max"]
    return low + (high - low) * expit(spec["sigmoid_coefficient"] * (2 * u - 1))


class Watermark:
    def __init__(self, key_latent, spec=PROTOCOL):
        self.spec = spec
        self.key = latent(key_latent)
        self.template = fft(self.key)
        y, x = np.meshgrid(np.linspace(-1, 1, 64), np.linspace(-1, 1, 64), indexing="ij")
        self.mask = np.exp(-(x*x + y*y) / (2 * spec["sigma_M"]**2))
        self.window = np.exp(-(x*x + y*y) / (2 * spec["sigma_w"]**2))
        fy, fx = np.meshgrid(np.arange(64)-32, np.arange(64)-32, indexing="ij")
        self.band = fx*fx + fy*fy <= PROTOCOL["frequency_radius"]**2
        self.key_spectrum = fft(self.mask * ifft(self.template))[:, self.band]

    def modulation(self, initial, feature, variant="full"):
        initial = latent(initial)
        spectrum = fft(initial)
        spectrum[:, self.band] = self.template[:, self.band]
        candidate = ifft(spectrum) - initial
        adaptive = strength(feature, self.spec)
        full = candidate * self.mask * self.window * adaptive
        if variant == "full":
            return full
        if variant not in {"no_fixed_mask", "no_window", "no_mcr", "no_mcre",
                           "fixed_1_75", "fixed_equal_energy"}:
            raise ValueError(f"Unknown variant: {variant}")
        spatial = 1 if variant == "no_fixed_mask" else self.mask
        window = 1 if variant in {"no_window", "no_mcre"} else self.window
        allocation = 1.75 if variant.startswith("fixed_") else adaptive
        change = candidate * spatial * window * allocation
        if variant in {"no_mcr", "no_mcre"}:
            change[1:] = 0
        if variant == "fixed_equal_energy":
            energy = np.linalg.norm(change)
            if energy == 0:
                if np.linalg.norm(full) == 0:
                    return change
                raise ValueError("Cannot match a nonzero reference to a zero perturbation")
            change *= np.linalg.norm(full) / energy
        return change

    def score(self, recovered, rule="pearson", variant="full"):
        """Section 3.7 uses Pearson; Tables 4 and 6 identify negative magnitude L1."""
        actual = fft(self.mask * latent(recovered))[:, self.band]
        reference = self.key_spectrum
        if variant in {"no_mcr", "no_mcre"}:
            actual, reference = actual[:1], reference[:1]
        if rule == "negative_complex_l1":
            return float(-np.mean(np.abs(actual - reference)))
        actual, reference = np.abs(actual), np.abs(reference)
        if rule == "negative_l1":
            return float(-np.mean(np.abs(actual - reference)))
        if rule != "pearson":
            raise ValueError(f"Unknown score rule: {rule}")
        a, b = actual.ravel(), reference.ravel()
        a, b = a - a.mean(), b - b.mean()
        denominator = np.linalg.norm(a) * np.linalg.norm(b)
        if denominator == 0:
            raise ValueError("Pearson correlation is undefined for constant vectors")
        return float(np.dot(a, b) / denominator)
