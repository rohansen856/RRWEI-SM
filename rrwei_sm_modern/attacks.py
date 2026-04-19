"""Modernized image-attack suite.

Extends the classical rrwei_sm.attacks (Gaussian, S&P, median, mean,
sharpen, JPEG, JPEG2000) with three "modern" attack proxies that
approximate the threat models most relevant to 2024-2026 watermarking
literature:

* **Neural-codec proxy** -- a cascade of aggressive JPEG + JPEG2000 +
  bicubic colorspace round-trip that mimics the rate-distortion
  trade-off of learned image codecs (HIFIC / NVIDIA VAE).  Pure
  Pillow-based implementation; no neural network required.
* **Super-resolution cascade** -- bicubic 2x down then 2x up, plus a
  median filter to emulate the softness introduced by real SR
  networks (ESRGAN, SwinIR).
* **Diffusion-regeneration proxy** -- a Gaussian-blur + mild JPEG
  round-trip; approximates the "passes through latent space and is
  regenerated" distortion characteristic of diffusion-based
  purification attacks.  Real SD VAE round-trip is available as an
  opt-in via the ``use_vae=True`` argument if ``diffusers`` and a
  model are installed locally; otherwise the fallback runs.

All legacy attacks are kept unchanged for comparison.
"""

from __future__ import annotations

import io
from typing import Literal

import numpy as np
from PIL import Image, ImageFilter

__all__ = [
    "gaussian_noise",
    "salt_and_pepper",
    "median_filter",
    "mean_filter",
    "sharpen_filter",
    "jpeg_compress",
    "jpeg2000_compress",
    "neural_codec_proxy",
    "super_resolution_cascade",
    "diffusion_regen_proxy",
    "AVAILABLE_ATTACKS",
]


# -----------------------------
# Classical attacks (kept).
# -----------------------------

def gaussian_noise(image: np.ndarray, sigma: float, seed: int | None = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noisy = image.astype(np.float64) + rng.normal(scale=sigma, size=image.shape)
    return np.clip(np.round(noisy), 0, 255).astype(np.uint8)


def salt_and_pepper(image: np.ndarray, p: float = 0.01, seed: int | None = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    mask = rng.random(image.shape)
    out = image.copy()
    out[mask < p / 2] = 0
    out[mask > 1 - p / 2] = 255
    return out


def _pil(image: np.ndarray) -> Image.Image:
    return Image.fromarray(image)


def median_filter(image: np.ndarray, ksize: int = 3) -> np.ndarray:
    return np.array(_pil(image).filter(ImageFilter.MedianFilter(ksize)), dtype=np.uint8)


def mean_filter(image: np.ndarray, ksize: int = 3) -> np.ndarray:
    return np.array(_pil(image).filter(ImageFilter.BoxBlur((ksize - 1) / 2)), dtype=np.uint8)


def sharpen_filter(image: np.ndarray, amount: float = 1.0) -> np.ndarray:
    sharp = _pil(image).filter(
        ImageFilter.UnsharpMask(radius=1, percent=int(amount * 150), threshold=3)
    )
    return np.array(sharp, dtype=np.uint8)


def jpeg_compress(image: np.ndarray, quality: int = 75) -> np.ndarray:
    buf = io.BytesIO()
    _pil(image).save(buf, format="JPEG", quality=int(quality))
    buf.seek(0)
    return np.array(Image.open(buf).convert("L"), dtype=np.uint8)


def jpeg2000_compress(image: np.ndarray, quality_layers=(20.0,)) -> np.ndarray:
    buf = io.BytesIO()
    _pil(image).save(
        buf,
        format="JPEG2000",
        quality_mode="rates",
        quality_layers=list(quality_layers),
    )
    buf.seek(0)
    return np.array(Image.open(buf).convert("L"), dtype=np.uint8)


# -----------------------------
# Modern attack proxies.
# -----------------------------

def neural_codec_proxy(
    image: np.ndarray,
    *,
    jpeg_q: int = 20,
    jpeg2000_rate: float = 50.0,
) -> np.ndarray:
    """Neural-codec distortion proxy.

    Learned image codecs typically share these artifacts with
    classical codecs: low-frequency bias, DCT/wavelet-grid ringing,
    and color-space bottleneck noise.  We approximate the envelope by
    chaining aggressive JPEG + JPEG2000 + a bicubic up/down cycle
    (the ``0.5x`` step mirrors the downsampling hyperpriors use
    internally).
    """
    # Step 1: heavy JPEG.
    step1 = jpeg_compress(image, quality=jpeg_q)
    # Step 2: JPEG2000 with tight rate.
    step2 = jpeg2000_compress(step1, quality_layers=(jpeg2000_rate,))
    # Step 3: bicubic colorspace round-trip 2x down then up.
    h, w = step2.shape
    pil = _pil(step2)
    pil_small = pil.resize((w // 2, h // 2), Image.BICUBIC)
    pil_back = pil_small.resize((w, h), Image.BICUBIC)
    return np.array(pil_back, dtype=np.uint8)


def super_resolution_cascade(image: np.ndarray) -> np.ndarray:
    """Bicubic downscale 2x then bicubic upscale 2x.

    Classical round-trip that captures the information loss of any
    real SR network that commits to a lower-resolution latent.  A
    cascade of two 2x steps is used (0.5x and 2x) rather than the
    identity 1x cycle because the latter does nothing.
    """
    h, w = image.shape
    pil = _pil(image)
    pil_down = pil.resize((w // 2, h // 2), Image.BICUBIC)
    pil_up = pil_down.resize((w, h), Image.BICUBIC)
    return np.array(pil_up, dtype=np.uint8)


def diffusion_regen_proxy(
    image: np.ndarray,
    *,
    sigma: float = 1.5,
    jpeg_q: int = 40,
    use_vae: bool = False,
) -> np.ndarray:
    """Diffusion-regeneration distortion proxy.

    The default path: a Gaussian blur (sigma=1.5) + mild JPEG round-
    trip.  This captures the "forget high frequencies and resynthesize
    from a latent" behavior of diffusion-based watermark attacks.

    Opt-in real-VAE path: if ``use_vae=True`` and ``diffusers`` +
    weights are installed locally, runs the SD VAE encode / decode
    cycle.  We do NOT download 300 MB of weights in the test suite;
    this branch exists for researchers who want to reproduce the
    stronger threat model.
    """
    if use_vae:  # pragma: no cover - network/CI heavyweight
        try:
            import torch
            from diffusers import AutoencoderKL

            vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
            vae.eval()
            x = torch.from_numpy(image).float() / 127.5 - 1.0
            x = x.unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)
            with torch.no_grad():
                z = vae.encode(x).latent_dist.mean
                rec = vae.decode(z).sample
            arr = ((rec[0, 0].cpu().numpy() + 1.0) * 127.5).clip(0, 255)
            return arr.astype(np.uint8)
        except Exception as e:
            print(f"VAE path unavailable, falling back to proxy ({e})")

    blurred = _pil(image).filter(ImageFilter.GaussianBlur(radius=sigma))
    buf = io.BytesIO()
    blurred.save(buf, format="JPEG", quality=int(jpeg_q))
    buf.seek(0)
    return np.array(Image.open(buf).convert("L"), dtype=np.uint8)


AVAILABLE_ATTACKS: dict[str, callable] = {
    "gaussian_noise_sigma5": lambda img: gaussian_noise(img, sigma=5.0),
    "salt_pepper_p001": lambda img: salt_and_pepper(img, p=0.01),
    "median_3": lambda img: median_filter(img, ksize=3),
    "mean_3": lambda img: mean_filter(img, ksize=3),
    "sharpen_1": lambda img: sharpen_filter(img, amount=1.0),
    "jpeg_q40": lambda img: jpeg_compress(img, quality=40),
    "jpeg2000_r30": lambda img: jpeg2000_compress(img, quality_layers=(30.0,)),
    "neural_codec": neural_codec_proxy,
    "sr_cascade": super_resolution_cascade,
    "diffusion_regen": diffusion_regen_proxy,
}
