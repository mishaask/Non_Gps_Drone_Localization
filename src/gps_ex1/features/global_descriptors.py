"""Global image descriptor backends.

Backends included in this project:

* ``basic``: lightweight OpenCV grayscale-thumbnail + HSV histogram baseline.
* ``dinov2``: direct global descriptor from Meta DINOv2 CLS/global token.
* ``anyloc`` / ``anyloc-gem``: AnyLoc-style descriptor using DINOv2 patch
  tokens with GeM pooling. This is intentionally self-contained: it gives us a
  practical AnyLoc-inspired visual-place-recognition backend without needing to
  vendor the whole AnyLoc repository or precomputed VLAD cluster centers.

The heavy backends are optional. They require PyTorch and the first run may
need internet access so torch.hub can cache the DINOv2 repo and model weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from gps_ex1.segmentation.dynamic_masks import DynamicObjectMasker


SUPPORTED_DESCRIPTOR_BACKENDS = ("basic", "dinov2", "anyloc", "anyloc-gem")


def _require_cv2():
    try:
        import cv2  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError("OpenCV is required. Install it with: pip install opencv-python") from exc
    return cv2


class DescriptorExtractor(Protocol):
    """Common API used by reference indexing and query localization."""

    name: str

    def describe_path(self, image_path: str | Path) -> np.ndarray:
        """Return one normalized descriptor for an image file."""

    def describe_bgr(self, image_bgr: np.ndarray) -> np.ndarray:
        """Return one normalized descriptor for an in-memory OpenCV BGR image."""


@dataclass
class BasicDescriptorExtractor:
    """CPU-friendly baseline descriptor: grayscale thumbnail + HSV histogram."""

    name: str = "basic"

    def describe_path(self, image_path: str | Path) -> np.ndarray:
        cv2 = _require_cv2()
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Could not read image: {image_path}")
        return self.describe_bgr(image)

    def describe_bgr(self, image_bgr: np.ndarray) -> np.ndarray:
        cv2 = _require_cv2()
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        thumb = cv2.resize(gray, (48, 48), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        thumb = thumb.reshape(-1)
        thumb = (thumb - float(thumb.mean())) / (float(thumb.std()) + 1e-6)

        small = cv2.resize(image_bgr, (160, 90), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 4, 4], [0, 180, 0, 256, 0, 256]).astype(np.float32)
        hist = hist.reshape(-1)
        hist = hist / (float(np.linalg.norm(hist)) + 1e-6)

        descriptor = np.concatenate([thumb, hist], axis=0).astype(np.float32)
        descriptor = descriptor / (float(np.linalg.norm(descriptor)) + 1e-6)
        return descriptor


class _DinoV2Base:
    """Shared DINOv2 loading and preprocessing logic."""

    def __init__(self, model_name: str = "dinov2_vits14", device: str | None = None, image_size: int = 518):
        try:
            import torch  # type: ignore
            from PIL import Image  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "DINOv2/AnyLoc backends require torch and pillow. Install optional dependencies with: "
                "python -m pip install -r requirements-modern.txt"
            ) from exc

        self._torch = torch
        self._Image = Image
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # DINOv2 ViT/14 works best when size is divisible by 14. 518 = 37*14.
        self.image_size = int(image_size)
        if self.image_size % 14 != 0:
            self.image_size = int(round(self.image_size / 14.0) * 14)
            self.image_size = max(224, self.image_size)

        try:
            self.model = torch.hub.load("facebookresearch/dinov2", model_name, trust_repo=True)
        except TypeError:  # older torch has no trust_repo kwarg
            self.model = torch.hub.load("facebookresearch/dinov2", model_name)
        self.model.eval().to(self.device)

    def _pil_from_path(self, image_path: str | Path):
        return self._Image.open(image_path).convert("RGB")

    def _pil_from_bgr(self, image_bgr: np.ndarray):
        cv2 = _require_cv2()
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        return self._Image.fromarray(rgb)

    def _preprocess_pil(self, image):
        torch = self._torch
        image = image.resize((self.image_size, self.image_size))
        arr = np.asarray(image).astype(np.float32) / 255.0
        arr = (arr - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array([0.229, 0.224, 0.225], dtype=np.float32)
        return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self.device)

    @staticmethod
    def _l2(vec: np.ndarray) -> np.ndarray:
        vec = vec.astype(np.float32).reshape(-1)
        return vec / (float(np.linalg.norm(vec)) + 1e-6)


class DinoV2DescriptorExtractor(_DinoV2Base):
    """Direct DINOv2 global descriptor backend.

    This uses the model output / CLS-style representation as a single image
    descriptor. It is a strong upgrade over the basic descriptor, but it does
    not explicitly aggregate local patch descriptors.
    """

    name = "dinov2"

    def describe_path(self, image_path: str | Path) -> np.ndarray:
        image = self._pil_from_path(image_path)
        return self._describe_pil(image)

    def describe_bgr(self, image_bgr: np.ndarray) -> np.ndarray:
        image = self._pil_from_bgr(image_bgr)
        return self._describe_pil(image)

    def _describe_pil(self, image) -> np.ndarray:
        torch = self._torch
        tensor = self._preprocess_pil(image)
        with torch.no_grad():
            features = self.model(tensor)
        vec = features.detach().cpu().numpy().reshape(-1).astype(np.float32)
        return self._l2(vec)


class AnyLocGemDescriptorExtractor(_DinoV2Base):
    """AnyLoc-style DINOv2 patch descriptor with GeM pooling.

    The AnyLoc paper shows that self-supervised DINOv2 patch features combined
    with unsupervised aggregation such as VLAD/GeM are highly useful for visual
    place recognition. Full AnyLoc-VLAD needs cluster centers and extra assets;
    this backend implements a practical self-contained GeM version so we can
    compare "direct DINOv2" versus "AnyLoc-style patch aggregation" in the
    assignment pipeline.
    """

    name = "anyloc-gem"

    def __init__(
        self,
        model_name: str = "dinov2_vits14",
        device: str | None = None,
        image_size: int = 518,
        gem_p: float = 3.0,
        include_cls: bool = True,
    ):
        super().__init__(model_name=model_name, device=device, image_size=image_size)
        self.gem_p = float(gem_p)
        self.include_cls = bool(include_cls)

    def describe_path(self, image_path: str | Path) -> np.ndarray:
        image = self._pil_from_path(image_path)
        return self._describe_pil(image)

    def describe_bgr(self, image_bgr: np.ndarray) -> np.ndarray:
        image = self._pil_from_bgr(image_bgr)
        return self._describe_pil(image)

    def _describe_pil(self, image) -> np.ndarray:
        torch = self._torch
        tensor = self._preprocess_pil(image)
        with torch.no_grad():
            if hasattr(self.model, "forward_features"):
                output = self.model.forward_features(tensor)
            else:  # pragma: no cover - DINOv2 models normally expose forward_features
                output = self.model(tensor)

        if isinstance(output, dict) and "x_norm_patchtokens" in output:
            patches = output["x_norm_patchtokens"]  # [1, num_patches, dim]
            cls = output.get("x_norm_clstoken", None)
        elif isinstance(output, dict) and "x_prenorm" in output:
            # Fallback for similar ViT APIs: token 0 is CLS, remaining are patches.
            tokens = output["x_prenorm"]
            cls = tokens[:, 0]
            patches = tokens[:, 1:]
        else:
            # Last-resort fallback: behave like direct DINOv2, but keep name so
            # the CSV/report still indicates which backend was attempted.
            vec = output.detach().cpu().numpy().reshape(-1).astype(np.float32)
            return self._l2(vec)

        patches = torch.nn.functional.normalize(patches.float(), dim=-1)
        # Signed GeM: DINO features can be negative. Standard GeM assumes
        # non-negative activations, so use sign(abs(x)^p) aggregation.
        p = max(self.gem_p, 1.0)
        gem = torch.sign(patches) * torch.clamp(torch.abs(patches), min=1e-6).pow(p)
        gem = torch.mean(gem, dim=1).sign() * torch.clamp(torch.abs(torch.mean(gem, dim=1)), min=1e-6).pow(1.0 / p)
        gem = torch.nn.functional.normalize(gem, dim=-1)

        parts = [gem]
        if self.include_cls and cls is not None:
            cls = torch.nn.functional.normalize(cls.float(), dim=-1)
            parts.append(cls)
        descriptor = torch.cat(parts, dim=-1)
        vec = descriptor.detach().cpu().numpy().reshape(-1).astype(np.float32)
        return self._l2(vec)


class MaskedDescriptorExtractor:
    """Descriptor wrapper that suppresses dynamic objects before describing."""

    def __init__(self, base: DescriptorExtractor, masker: DynamicObjectMasker):
        self.base = base
        self.masker = masker
        self.name = f"{base.name}+dynamic-mask"

    def describe_path(self, image_path: str | Path) -> np.ndarray:
        masked = self.masker.mask_path(image_path)
        return self.base.describe_bgr(masked)

    def describe_bgr(self, image_bgr: np.ndarray) -> np.ndarray:
        masked = self.masker.mask_bgr(image_bgr)
        return self.base.describe_bgr(masked)


def create_descriptor_extractor(
    name: str,
    masker: DynamicObjectMasker | None = None,
    model_name: str = "dinov2_vits14",
    device: str | None = None,
    image_size: int = 518,
) -> DescriptorExtractor:
    """Create a descriptor backend by CLI-friendly name."""

    normalized = name.strip().lower()
    if normalized in {"basic", "opencv", "baseline"}:
        extractor: DescriptorExtractor = BasicDescriptorExtractor()
    elif normalized in {"dinov2", "dino", "dino-v2"}:
        extractor = DinoV2DescriptorExtractor(model_name=model_name, device=device, image_size=image_size)
    elif normalized in {"anyloc", "anyloc-gem", "dinov2-gem", "dino-gem"}:
        extractor = AnyLocGemDescriptorExtractor(model_name=model_name, device=device, image_size=image_size)
    else:
        raise ValueError(
            f"Unknown descriptor backend: {name}. Use one of: basic, dinov2, anyloc-gem."
        )

    if masker is not None:
        return MaskedDescriptorExtractor(extractor, masker)
    return extractor
