"""Modern perceptual quality metrics.

Extends the classical PSNR / SSIM with two perceptual-learning-based
distances used in modern image-quality literature:

* **LPIPS** -- Learned Perceptual Image Patch Similarity (Zhang et al.
  CVPR 2018), implemented via the ``lpips`` pip package wrapped around
  a pre-trained AlexNet backbone.  Correlates strongly with human
  perceptual judgments and is the de-facto quality metric for image
  generation / restoration since 2018.
* **DISTS** -- Deep Image Structure and Texture Similarity (Ding et
  al.  TPAMI 2020), approximated here by a multi-scale combination of
  structural (gradient-based) and texture (local-moment) similarity.
  The hand-rolled proxy avoids the extra dependency while preserving
  the qualitative behavior (monotonic in perceptual distortion).

Classical metrics (PSNR, SSIM, NPCR, UACI, pixel correlation, PEE
capacity) remain in ``rrwei_sm.metrics`` until Step 8 rename.

Lazy imports
------------
LPIPS pulls in PyTorch which is heavyweight.  We lazy-import it so
``evaluate.py`` can still run without LPIPS when torch is absent or
when the AlexNet weights are not reachable.
"""

from __future__ import annotations

import functools
import numpy as np

try:  # skimage is guaranteed by requirements; import at module load.
    from skimage.metrics import structural_similarity as _ski_ssim
except Exception:  # pragma: no cover
    _ski_ssim = None

__all__ = [
    "psnr",
    "ssim",
    "lpips_distance",
    "dists_proxy",
]


def psnr(a: np.ndarray, b: np.ndarray, peak: float = 255.0) -> float:
    """Peak-signal-to-noise ratio in dB."""
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    mse = float(np.mean((a - b) ** 2))
    if mse == 0:
        return 99.0
    return 10.0 * float(np.log10(peak**2 / mse))


def ssim(a: np.ndarray, b: np.ndarray, data_range: float = 255.0) -> float:
    if _ski_ssim is None:  # pragma: no cover
        raise RuntimeError("scikit-image is required for SSIM")
    return float(_ski_ssim(a.astype(np.float64), b.astype(np.float64), data_range=data_range))


@functools.lru_cache(maxsize=1)
def _lpips_model():
    """Import and build an LPIPS model once per process."""
    import torch  # noqa: F401
    import lpips

    model = lpips.LPIPS(net="alex", verbose=False)
    model.eval()
    return model


def _to_lpips_tensor(image: np.ndarray):
    import torch

    if image.ndim == 2:
        image = np.stack([image, image, image], axis=0)
    elif image.ndim == 3 and image.shape[-1] == 3:
        image = image.transpose(2, 0, 1)
    elif image.ndim == 3 and image.shape[0] == 3:
        pass
    else:
        raise ValueError(f"unexpected image shape: {image.shape}")
    # Normalize to [-1, 1] as expected by lpips.
    arr = image.astype(np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr).unsqueeze(0)


def lpips_distance(a: np.ndarray, b: np.ndarray) -> float:
    """LPIPS (AlexNet) distance, non-negative; 0 for identical inputs.

    Raises RuntimeError if torch / lpips are unavailable.
    """
    try:
        import torch  # noqa: F401
    except ImportError as e:
        raise RuntimeError(f"LPIPS requires torch: {e}")

    model = _lpips_model()
    ta = _to_lpips_tensor(a)
    tb = _to_lpips_tensor(b)
    import torch

    with torch.no_grad():
        d = model(ta, tb)
    return float(d.item())


# ---------------------------------------------------------------------------
# DISTS-proxy (multi-scale structure + texture similarity).
# ---------------------------------------------------------------------------

def _gaussian_downsample(x: np.ndarray) -> np.ndarray:
    """2x downsample with a 1:2:1 separable filter (matches DISTS pyramid)."""
    # Pad
    padded = np.pad(x, 1, mode="reflect")
    # Horizontal conv
    k = np.array([1, 2, 1], dtype=np.float64) / 4.0
    h = np.zeros_like(x, dtype=np.float64)
    for dx, kk in enumerate(k):
        h += kk * padded[1:-1, dx : dx + x.shape[1]]
    # Vertical conv
    h_padded = np.pad(h, 1, mode="reflect")
    v = np.zeros_like(x, dtype=np.float64)
    for dy, kk in enumerate(k):
        v += kk * h_padded[dy : dy + x.shape[0], 1:-1]
    return v[::2, ::2]


def _grad_magnitude(x: np.ndarray) -> np.ndarray:
    gx = np.zeros_like(x, dtype=np.float64)
    gy = np.zeros_like(x, dtype=np.float64)
    gx[:, 1:-1] = (x[:, 2:] - x[:, :-2]) / 2.0
    gy[1:-1, :] = (x[2:, :] - x[:-2, :]) / 2.0
    return np.sqrt(gx * gx + gy * gy)


def _texture_similarity(a: np.ndarray, b: np.ndarray, eps: float = 1e-6) -> float:
    """Similarity of per-block means (proxy for DISTS texture term)."""
    # 8x8 block means.
    def block_means(x: np.ndarray) -> np.ndarray:
        h, w = x.shape
        bh = h - (h % 8)
        bw = w - (w % 8)
        x = x[:bh, :bw]
        return x.reshape(bh // 8, 8, bw // 8, 8).mean(axis=(1, 3))

    ma = block_means(a.astype(np.float64))
    mb = block_means(b.astype(np.float64))
    num = 2 * ma * mb + eps
    den = ma * ma + mb * mb + eps
    return float(np.mean(num / den))


def _structure_similarity(a: np.ndarray, b: np.ndarray, eps: float = 1e-6) -> float:
    """Similarity of gradient magnitudes (proxy for DISTS structure term)."""
    ga = _grad_magnitude(a.astype(np.float64))
    gb = _grad_magnitude(b.astype(np.float64))
    num = 2 * ga * gb + eps
    den = ga * ga + gb * gb + eps
    return float(np.mean(num / den))


def dists_proxy(a: np.ndarray, b: np.ndarray, n_scales: int = 3) -> float:
    """Hand-rolled Deep-Image-Structure-and-Texture-Similarity proxy.

    Returns a distance in ``[0, 1]``; 0 means identical, 1 means
    completely dissimilar.  At each of ``n_scales`` resolutions we
    combine a structural similarity term (on gradient magnitudes)
    and a textural similarity term (on 8x8 block means).
    """
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    pyramid_a = [a.astype(np.float64)]
    pyramid_b = [b.astype(np.float64)]
    for _ in range(n_scales - 1):
        pyramid_a.append(_gaussian_downsample(pyramid_a[-1]))
        pyramid_b.append(_gaussian_downsample(pyramid_b[-1]))
    sims: list[float] = []
    for la, lb in zip(pyramid_a, pyramid_b):
        # Only compute texture term if the level is big enough (>= 8).
        if min(la.shape) >= 8:
            t = _texture_similarity(la, lb)
        else:
            t = _structure_similarity(la, lb)
        s = _structure_similarity(la, lb)
        sims.append(0.5 * (t + s))
    sim = float(np.mean(sims))
    return max(0.0, 1.0 - sim)
