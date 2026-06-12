"""Dynamic-object masking for aerial visual localization.

This module is deliberately conservative. Generic YOLO/COCO models often make
mistakes on high-altitude drone footage: tiny top-view cars may be missed, while
large rooftops or shadows may be classified as trucks/buses. The default setup
therefore masks only small ``car`` detections, uses bounding boxes by default
instead of large segmentation blobs, and filters detections by area/size.

The rest of the project imports this module lazily. Ultralytics/PyTorch are only
required when masking is explicitly enabled.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


COCO_CLASS_IDS = {
    "person": 0,
    "bicycle": 1,
    "car": 2,
    "motorcycle": 3,
    "airplane": 4,
    "bus": 5,
    "train": 6,
    "truck": 7,
    "boat": 8,
}
COCO_ID_TO_NAME = {value: key for key, value in COCO_CLASS_IDS.items()}

# Safer aerial default. The previous broad default included bus/train/truck and
# caused large rooftops/buildings to be masked in our drone frames.
DEFAULT_DYNAMIC_CLASSES = "car"


def _require_cv2():
    try:
        import cv2  # type: ignore
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("OpenCV is required. Install it with: python -m pip install opencv-python") from exc
    return cv2


def parse_class_filter(text: str | None) -> list[int]:
    """Parse a comma-separated class list into COCO class IDs.

    Accepts class names like ``car,truck`` or numeric IDs like ``2,7``.
    """

    if text is None or not text.strip():
        text = DEFAULT_DYNAMIC_CLASSES
    ids: list[int] = []
    for part in text.split(","):
        token = part.strip().lower().replace(" ", "_")
        if not token:
            continue
        if token.isdigit():
            class_id = int(token)
        else:
            if token not in COCO_CLASS_IDS:
                valid = ", ".join(sorted(COCO_CLASS_IDS.keys()))
                raise ValueError(f"Unknown COCO class '{token}'. Valid common names: {valid}. Or pass numeric IDs.")
            class_id = COCO_CLASS_IDS[token]
        if class_id not in ids:
            ids.append(class_id)
    return ids


@dataclass(frozen=True)
class DynamicDetection:
    class_id: int
    class_name: str
    confidence: float
    xyxy: tuple[int, int, int, int]
    box_area_px: int
    box_area_frac: float
    mask_area_px: int
    mask_area_frac: float
    accepted: bool
    reason: str


@dataclass
class DynamicObjectMasker:
    """Conservative YOLO/SAM-style dynamic object masker.

    Key defaults were chosen for 1080p aerial footage:
    - mask only ``car`` by default;
    - use YOLO boxes rather than segmentation masks by default;
    - reject detections that are too large to plausibly be a car.

    Set ``classes``/``max_area_frac``/``mask_source`` manually for experiments.
    """

    model_name: str = "yolov8n-seg.pt"
    classes: str = DEFAULT_DYNAMIC_CLASSES
    confidence: float = 0.35
    iou: float = 0.7
    dilate_px: int = 4
    fill: str = "blur"
    imgsz: int = 960
    device: str | None = None
    min_area_px: int = 20
    max_area_frac: float = 0.015
    max_width_frac: float = 0.20
    max_height_frac: float = 0.20
    mask_source: str = "box"  # box, segmentation, auto

    def __post_init__(self) -> None:
        try:
            from ultralytics import YOLO  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "Dynamic-object masking requires Ultralytics. Install it with: "
                "python -m pip install ultralytics"
            ) from exc

        self.class_ids = parse_class_filter(self.classes)
        self.model = YOLO(self.model_name)
        self._path_cache: dict[str, np.ndarray] = {}
        self._mask_cache: dict[str, np.ndarray] = {}
        self._detections_cache: dict[str, list[DynamicDetection]] = {}

        normalized = self.mask_source.strip().lower()
        if normalized not in {"box", "segmentation", "auto"}:
            raise ValueError("mask_source must be one of: box, segmentation, auto")
        self.mask_source = normalized

    def mask_path(self, image_path: str | Path) -> np.ndarray:
        """Read an image path and mask dynamic objects without retaining frame caches.

        Reference-index builds may process thousands of 1080p frames. Keeping every
        masked image/mask in memory can exhaust RAM on normal laptops, so path-based
        masking intentionally avoids persistent caching.
        """

        cv2 = _require_cv2()
        key = str(Path(image_path))
        image = cv2.imread(key, cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Could not read image for masking: {image_path}")
        return self.mask_bgr(image, cache_key=None)

    def mask_bgr(self, image_bgr: np.ndarray, cache_key: str | None = None) -> np.ndarray:
        """Return a copy of ``image_bgr`` with accepted dynamic objects suppressed."""

        mask = self.dynamic_mask_bgr(image_bgr, cache_key=cache_key)
        return apply_mask_fill(image_bgr, mask, fill=self.fill)

    def allowed_feature_mask_bgr(self, image_bgr: np.ndarray, cache_key: str | None = None) -> np.ndarray:
        """Return an OpenCV feature mask where 255 means usable and 0 means ignore."""

        cv2 = _require_cv2()
        dynamic = self.dynamic_mask_bgr(image_bgr, cache_key=cache_key)
        if dynamic is None or not np.any(dynamic):
            return np.full(image_bgr.shape[:2], 255, dtype=np.uint8)
        return cv2.bitwise_not(dynamic)

    def dynamic_mask_bgr(self, image_bgr: np.ndarray, cache_key: str | None = None) -> np.ndarray:
        """Return a uint8 binary mask where accepted dynamic objects are 255."""

        cv2 = _require_cv2()
        # Do not use persistent mask caching here. Full reference-index builds can
        # process thousands of large images, and one uint8 1080p mask is about 2 MB.
        cache_key = None

        h, w = image_bgr.shape[:2]
        combined = np.zeros((h, w), dtype=np.uint8)
        detections, per_instance_masks = self._predict_detections_and_masks(image_bgr, cache_key=cache_key)

        for i, det in enumerate(detections):
            if not det.accepted:
                continue
            x1, y1, x2, y2 = det.xyxy
            if x2 <= x1 or y2 <= y1:
                continue

            # For aerial footage, boxes are often safer than masks. Segmentation
            # masks can spill onto rooftops/trees if the detector is uncertain.
            instance_mask: np.ndarray | None = None
            if self.mask_source in {"segmentation", "auto"} and i < len(per_instance_masks):
                instance_mask = per_instance_masks[i]

            if instance_mask is not None:
                mask_area_frac = float(np.count_nonzero(instance_mask)) / float(max(1, h * w))
                if self.mask_source == "segmentation" and mask_area_frac <= self.max_area_frac:
                    combined = cv2.bitwise_or(combined, instance_mask)
                    continue
                if self.mask_source == "auto" and 0 < mask_area_frac <= self.max_area_frac:
                    combined = cv2.bitwise_or(combined, instance_mask)
                    continue

            # Box fallback / default.
            combined[y1 : y2 + 1, x1 : x2 + 1] = 255

        if self.dilate_px > 0 and np.any(combined):
            kernel_size = max(1, int(self.dilate_px) * 2 + 1)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            combined = cv2.dilate(combined, kernel, iterations=1)

        return combined

    def detections_bgr(self, image_bgr: np.ndarray, cache_key: str | None = None) -> list[DynamicDetection]:
        """Return filtered/raw detection metadata for debugging."""

        detections, _ = self._predict_detections_and_masks(image_bgr, cache_key=cache_key)
        return detections

    def debug_overlay_bgr(self, image_bgr: np.ndarray, cache_key: str | None = None, show_rejected: bool = True) -> np.ndarray:
        """Draw mask overlay plus class/confidence labels for debugging."""

        cv2 = _require_cv2()
        output = image_bgr.copy()
        mask = self.dynamic_mask_bgr(image_bgr, cache_key=cache_key)
        detections = self.detections_bgr(image_bgr, cache_key=cache_key)

        if np.any(mask):
            overlay = output.copy()
            overlay[mask > 0] = (90, 90, 90)
            cv2.addWeighted(overlay, 0.45, output, 0.55, 0, output)

        for det in detections:
            if not det.accepted and not show_rejected:
                continue
            x1, y1, x2, y2 = det.xyxy
            color = (0, 255, 0) if det.accepted else (0, 0, 255)
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
            status = "OK" if det.accepted else f"REJ:{det.reason}"
            label = f"{det.class_name} {det.confidence:.2f} area={det.box_area_frac:.3f} {status}"
            y_text = max(18, y1 - 6)
            cv2.putText(output, label, (x1, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(output, label, (x1, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        return output

    def _predict_detections_and_masks(self, image_bgr: np.ndarray, cache_key: str | None = None) -> tuple[list[DynamicDetection], list[np.ndarray]]:
        cv2 = _require_cv2()
        if cache_key is not None and cache_key in self._detections_cache:
            # Masks are cheap enough to recompute with the cached detection list unavailable;
            # the caller mostly uses the cached list for labels. Fall through for masks.
            pass

        h, w = image_bgr.shape[:2]
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        kwargs: dict[str, Any] = {
            "classes": self.class_ids,
            "conf": self.confidence,
            "iou": self.iou,
            "imgsz": self.imgsz,
            "verbose": False,
            "retina_masks": self.mask_source in {"segmentation", "auto"},
        }
        if self.device:
            kwargs["device"] = self.device
        results = self.model.predict(rgb, **kwargs)
        if not results:
            return [], []
        result = results[0]

        boxes = getattr(result, "boxes", None)
        if boxes is None or getattr(boxes, "xyxy", None) is None:
            return [], []

        try:
            xyxy_arr = boxes.xyxy.detach().cpu().numpy()
        except AttributeError:
            xyxy_arr = np.asarray(boxes.xyxy)
        try:
            cls_arr = boxes.cls.detach().cpu().numpy().astype(int)
        except AttributeError:
            cls_arr = np.asarray(getattr(boxes, "cls", np.zeros(len(xyxy_arr)))).astype(int)
        try:
            conf_arr = boxes.conf.detach().cpu().numpy()
        except AttributeError:
            conf_arr = np.asarray(getattr(boxes, "conf", np.ones(len(xyxy_arr))))

        raw_masks: list[np.ndarray | None] = [None] * len(xyxy_arr)
        masks = getattr(result, "masks", None)
        if masks is not None and getattr(masks, "data", None) is not None:
            data = masks.data
            try:
                mask_arr = data.detach().cpu().numpy()
            except AttributeError:
                mask_arr = np.asarray(data)
            for i, one in enumerate(mask_arr[: len(raw_masks)]):
                one_mask = (one > 0.5).astype(np.uint8) * 255
                if one_mask.shape[:2] != (h, w):
                    one_mask = cv2.resize(one_mask, (w, h), interpolation=cv2.INTER_NEAREST)
                raw_masks[i] = one_mask

        detections: list[DynamicDetection] = []
        instance_masks: list[np.ndarray] = []
        image_area = float(max(1, h * w))
        for i, xyxy in enumerate(xyxy_arr):
            x1, y1, x2, y2 = xyxy
            x1i = max(0, min(w - 1, int(round(float(x1)))))
            y1i = max(0, min(h - 1, int(round(float(y1)))))
            x2i = max(0, min(w - 1, int(round(float(x2)))))
            y2i = max(0, min(h - 1, int(round(float(y2)))))
            box_w = max(0, x2i - x1i + 1)
            box_h = max(0, y2i - y1i + 1)
            box_area = int(box_w * box_h)
            box_area_frac = float(box_area) / image_area
            mask = raw_masks[i] if i < len(raw_masks) else None
            mask_area = int(np.count_nonzero(mask)) if mask is not None else 0
            mask_area_frac = float(mask_area) / image_area
            class_id = int(cls_arr[i]) if i < len(cls_arr) else -1
            class_name = COCO_ID_TO_NAME.get(class_id, str(class_id))
            conf = float(conf_arr[i]) if i < len(conf_arr) else 0.0

            accepted = True
            reason = "accepted"
            if class_id not in self.class_ids:
                accepted = False
                reason = "class_filtered"
            elif box_area < self.min_area_px:
                accepted = False
                reason = "too_small"
            elif box_area_frac > self.max_area_frac:
                accepted = False
                reason = "too_large_area"
            elif (box_w / max(1, w)) > self.max_width_frac:
                accepted = False
                reason = "too_wide"
            elif (box_h / max(1, h)) > self.max_height_frac:
                accepted = False
                reason = "too_tall"

            detections.append(
                DynamicDetection(
                    class_id=class_id,
                    class_name=class_name,
                    confidence=conf,
                    xyxy=(x1i, y1i, x2i, y2i),
                    box_area_px=box_area,
                    box_area_frac=box_area_frac,
                    mask_area_px=mask_area,
                    mask_area_frac=mask_area_frac,
                    accepted=accepted,
                    reason=reason,
                )
            )
            if mask is not None:
                instance_masks.append(mask)
            else:
                instance_masks.append(np.zeros((h, w), dtype=np.uint8))

        return detections, instance_masks


def apply_mask_fill(image_bgr: np.ndarray, mask: np.ndarray, fill: str = "blur") -> np.ndarray:
    """Replace masked pixels so descriptor extractors do not latch onto them."""

    cv2 = _require_cv2()
    output = image_bgr.copy()
    if mask is None or not np.any(mask):
        return output

    normalized = fill.strip().lower()
    masked = mask > 0
    if normalized == "black":
        output[masked] = (0, 0, 0)
    elif normalized == "gray":
        output[masked] = (127, 127, 127)
    elif normalized == "blur":
        blurred = cv2.GaussianBlur(output, (0, 0), sigmaX=25, sigmaY=25)
        output[masked] = blurred[masked]
    elif normalized == "median":
        unmasked_pixels = output[~masked]
        if unmasked_pixels.size == 0:
            fill_color = np.array([127, 127, 127], dtype=np.uint8)
        else:
            fill_color = np.median(unmasked_pixels.reshape(-1, 3), axis=0).astype(np.uint8)
        output[masked] = fill_color
    else:
        raise ValueError("Unknown mask fill mode. Use median, gray, black, or blur.")
    return output


def create_dynamic_object_masker(
    enabled: bool,
    model_name: str = "yolov8n-seg.pt",
    classes: str = DEFAULT_DYNAMIC_CLASSES,
    confidence: float = 0.35,
    iou: float = 0.7,
    dilate_px: int = 4,
    fill: str = "blur",
    imgsz: int = 960,
    device: str | None = None,
    min_area_px: int = 20,
    max_area_frac: float = 0.015,
    max_width_frac: float = 0.20,
    max_height_frac: float = 0.20,
    mask_source: str = "box",
) -> DynamicObjectMasker | None:
    if not enabled:
        return None
    return DynamicObjectMasker(
        model_name=model_name,
        classes=classes,
        confidence=confidence,
        iou=iou,
        dilate_px=dilate_px,
        fill=fill,
        imgsz=imgsz,
        device=device,
        min_area_px=min_area_px,
        max_area_frac=max_area_frac,
        max_width_frac=max_width_frac,
        max_height_frac=max_height_frac,
        mask_source=mask_source,
    )
