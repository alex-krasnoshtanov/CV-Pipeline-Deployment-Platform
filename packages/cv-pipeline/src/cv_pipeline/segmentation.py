"""Segmentation model loading and patch-based inference.

Loads a U-Net checkpoint and runs inference on arbitrarily large images
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)

# Default inference settings matching the configuration the released
# unet-v1 checkpoint was trained with. Changing either silently degrades
# results against that checkpoint.
_DEFAULT_PATCH_SIZE: int = 256
_DEFAULT_OVERLAP: float = 0.5
_DEFAULT_BATCH_SIZE: int = 16
_DEFAULT_THRESHOLD: float = 0.5


class SegmentationModel:
    """U-Net segmentation model with patch-based inference.

    Loads a ResNet34-backed U-Net checkpoint once and exposes a
    ``predict`` method that can be called repeatedly on different
    images. Large images are handled by padding, patching, predicting,
    and reconstructing with overlap averaging.

    Args:
        model_path: Path to the ``.pth`` checkpoint file.
        patch_size: Side length of the square patches in pixels.
        overlap: Overlap ratio between adjacent patches (0.0 to 1.0).
        device: Torch device string. If ``None``, CUDA is used when
            available, otherwise CPU.

    Raises:
        ValueError: If ``patch_size`` is not positive, ``overlap``
            is outside [0, 1), or the computed step is less than 1.
    """

    def __init__(
        self,
        model_path: str | Path,
        patch_size: int = _DEFAULT_PATCH_SIZE,
        overlap: float = _DEFAULT_OVERLAP,
        device: str | None = None,
    ) -> None:
        if patch_size <= 0:
            raise ValueError("patch_size must be greater than 0.")
        if overlap < 0 or overlap >= 1:
            raise ValueError("overlap must satisfy 0 <= overlap < 1.")
        step = int(patch_size * (1 - overlap))
        if step < 1:
            raise ValueError(
                "Computed patch step must be at least 1. Adjust patch_size or overlap."
            )

        self.patch_size = patch_size
        self.overlap = overlap
        self.step = step

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        logger.info(
            "Initialising segmentation model — patch %dx%d, overlap %.0f%%, device %s.",
            patch_size,
            patch_size,
            overlap * 100,
            self.device,
        )

        # Resolve weights source: version string (via registry) or direct path.
        # A bare version name is looked up in REGISTRY; anything else is a path.
        # This keeps the CLI's --model <path> contract while letting callers
        # use SegmentationModel("unet-v1") for convenience.
        from cv_pipeline.weights import REGISTRY, get_weights

        model_path_str = str(model_path)
        path_obj = Path(model_path)
        is_version_name = (
            model_path_str in REGISTRY
            and "/" not in model_path_str
            and "\\" not in model_path_str
            and not path_obj.exists()
        )
        if is_version_name:
            resolved_path = get_weights(model_path_str)
            logger.info("Resolved version '%s' to '%s'.", model_path_str, resolved_path)
        else:
            resolved_path = path_obj

        self._requested_version = model_path_str if is_version_name else None

        self._model_path = resolved_path

        # Load checkpoint ONCE, pass to all methods.
        checkpoint = self._load_checkpoint()
        self.in_channels = self._detect_input_channels(checkpoint)
        self.model_version = self._read_model_version(checkpoint)
        self._build_model(checkpoint)

    # ---- public API --------------------------------------------------

    def predict(
        self,
        image: np.ndarray,
        batch_size: int = _DEFAULT_BATCH_SIZE,
    ) -> np.ndarray:
        """Run segmentation on a single image.

        Args:
            image: Input image as a numpy array, either (H, W) grayscale
                or (H, W, 3) RGB, dtype ``uint8``.
            batch_size: Number of patches to process per forward pass.

        Returns:
            Probability map (H, W) as ``float32`` in [0, 1]. Each pixel
            value is the model's confidence that the pixel is root tissue.
        """
        h, w = image.shape[:2]
        logger.info("Running inference on %dx%d image.", h, w)

        image = self._prepare_channels(image)

        padding = self._calculate_padding(h, w)
        padded = self._pad_image(image, padding)
        patches = self._create_patches(padded)

        logger.debug("Created %d patches.", len(patches))

        predictions = self._predict_patches(patches, batch_size)
        padded_h, padded_w = padded.shape[:2]
        reconstructed = self._reconstruct(predictions, padded_h, padded_w)
        result = self._remove_padding(reconstructed, padding)

        if result.shape != (h, w):
            raise RuntimeError(
                f"Shape mismatch after reconstruction: expected ({h}, {w}), "
                f"got {result.shape}.",
            )

        return result

    def predict_mask(
        self,
        image: np.ndarray,
        threshold: float = _DEFAULT_THRESHOLD,
        batch_size: int = _DEFAULT_BATCH_SIZE,
    ) -> tuple[np.ndarray, float]:
        """Run segmentation and return a binary mask with confidence.

        Convenience method that calls ``predict``, binarises the
        probability map, and computes the mean confidence across
        root-classified pixels.

        Args:
            image: Input image (H, W) or (H, W, 3), dtype ``uint8``.
            threshold: Probability threshold for binarisation.
            batch_size: Number of patches per forward pass.

        Returns:
            A tuple of (binary_mask, mask_confidence).
            ``binary_mask`` is ``uint8`` with 0 for background and 255
            for root. ``mask_confidence`` is the mean probability of
            root-classified pixels, or 0.0 if no pixels are classified
            as root.
        """
        prob_map = self.predict(image, batch_size=batch_size)

        binary = (prob_map >= threshold).astype(np.uint8) * 255

        root_pixels = prob_map[binary == 255]
        if len(root_pixels) == 0:
            mask_confidence = 0.0
        else:
            mask_confidence = float(np.mean(root_pixels))

        return binary, mask_confidence

    # ---- model loading -----------------------------------------------

    def _load_checkpoint(self) -> dict:
        """Load the checkpoint file from disk once.

        Returns:
            The checkpoint dictionary.

        Raises:
            RuntimeError: If the checkpoint cannot be loaded.
        """
        # weights_only=False is required because our checkpoints contain
        # non-tensor metadata (model_version string, epoch integer).
        # Only load checkpoints from trusted sources.
        try:
            checkpoint = torch.load(
                self._model_path,
                map_location="cpu",
                weights_only=False,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Cannot load checkpoint at '{self._model_path}': {exc}",
            ) from exc
        return checkpoint

    def _detect_input_channels(self, checkpoint: dict) -> int:
        """Detect the number of input channels from the checkpoint.

        Args:
            checkpoint: The loaded checkpoint dictionary.

        Returns:
            1 for grayscale or 3 for RGB.
        """
        state_dict = checkpoint.get("model_state_dict", checkpoint)

        if "encoder.conv1.weight" in state_dict:
            channels = state_dict["encoder.conv1.weight"].shape[1]
            logger.info(
                "Detected %d input channel(s) from checkpoint.",
                channels,
            )
            return int(channels)

        logger.warning(
            "Could not detect input channels — defaulting to 1 (grayscale).",
        )
        return 1

    def _read_model_version(self, checkpoint: dict) -> str:
        """Read the model version string from checkpoint metadata.

        Falls back to ``'unet-v0'`` if the checkpoint does not contain
        a ``model_version`` key.

        When a registry version string was requested (e.g. ``"unet-v1"``),
        prefer that value over stale checkpoint metadata so API responses
        reflect the configured deployment version.

        Args:
            checkpoint: The loaded checkpoint dictionary.

        Returns:
            A version string in the format ``<architecture>-v<version>``.
        """
        metadata_version = str(checkpoint.get("model_version", "unet-v0"))
        if self._requested_version and metadata_version != self._requested_version:
            logger.warning(
                "Checkpoint metadata version '%s' differs from requested '%s'; "
                "using requested version.",
                metadata_version,
                self._requested_version,
            )
            return self._requested_version

        logger.info("Model version: %s.", metadata_version)
        return metadata_version

    def _build_model(self, checkpoint: dict) -> None:
        """Initialise the U-Net architecture and load trained weights.

        Args:
            checkpoint: The loaded checkpoint dictionary.
        """
        import segmentation_models_pytorch as smp

        self.model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=self.in_channels,
            classes=1,
            activation=None,
        ).to(self.device)

        state_dict = checkpoint.get("model_state_dict", checkpoint)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info("Model loaded — %s parameters.", f"{total_params:,}")

    # ---- channel preparation -----------------------------------------

    def _prepare_channels(self, image: np.ndarray) -> np.ndarray:
        """Match the image channel count to what the model expects.

        Args:
            image: (H, W) or (H, W, 3) numpy array.

        Returns:
            Image with a trailing channel dimension matching
            ``self.in_channels``.
        """
        if self.in_channels == 1:
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            image = np.expand_dims(image, axis=-1)
        elif self.in_channels == 3:
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        return image

    # ---- patching and reconstruction ---------------------------------

    def _calculate_padding(
        self,
        h: int,
        w: int,
    ) -> tuple[int, int, int, int]:
        """Calculate reflection padding so the image is patch-compatible.

        Args:
            h: Image height.
            w: Image width.

        Returns:
            A tuple (top, bottom, left, right) of padding amounts.
        """

        def _pad_needed(size: int) -> int:
            if size < self.patch_size:
                return self.patch_size - size
            remainder = (size - self.patch_size) % self.step
            if remainder != 0:
                return self.step - remainder
            return 0

        h_pad = _pad_needed(h)
        w_pad = _pad_needed(w)
        return (
            h_pad // 2,
            h_pad - h_pad // 2,
            w_pad // 2,
            w_pad - w_pad // 2,
        )

    def _pad_image(
        self,
        image: np.ndarray,
        padding: tuple[int, int, int, int],
    ) -> np.ndarray:
        """Pad the image with reflection to avoid edge artefacts.

        Args:
            image: Image array (H, W, C).
            padding: (top, bottom, left, right).

        Returns:
            Padded image array.
        """
        top, bottom, left, right = padding
        padded = cv2.copyMakeBorder(
            image,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_REFLECT_101,
        )
        # cv2 may drop a trailing channel dim of 1.
        if len(image.shape) == 3 and len(padded.shape) == 2:
            padded = np.expand_dims(padded, axis=-1)
        return padded

    def _create_patches(self, image: np.ndarray) -> np.ndarray:
        """Slide a window over the image and collect patches.

        Args:
            image: Padded image (H, W, C).

        Returns:
            Patches as (N, C, patch_size, patch_size) in PyTorch format.
        """
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)

        h, w, c = image.shape
        patches: list[np.ndarray] = []

        for y in range(0, h - self.patch_size + 1, self.step):
            for x in range(0, w - self.patch_size + 1, self.step):
                patch = image[
                    y : y + self.patch_size,
                    x : x + self.patch_size,
                    :,
                ]
                patches.append(patch)

        # (N, H, W, C) → (N, C, H, W)
        batch = np.stack(patches, axis=0).transpose(0, 3, 1, 2)
        return batch

    def _predict_patches(
        self,
        patches: np.ndarray,
        batch_size: int,
    ) -> np.ndarray:
        """Run the model on a batch of patches.

        Args:
            patches: (N, C, patch_size, patch_size) as ``uint8``.
            batch_size: Forward-pass batch size.

        Returns:
            Predictions (N, patch_size, patch_size) as ``float32``
            in [0, 1].
        """
        tensor = torch.from_numpy(patches).float() / 255.0
        results: list[np.ndarray] = []

        with torch.no_grad():
            for i in range(0, len(tensor), batch_size):
                batch = tensor[i : i + batch_size].to(self.device)
                output = torch.sigmoid(self.model(batch))
                results.append(output.cpu().numpy()[:, 0, :, :])

        return np.concatenate(results, axis=0)

    def _reconstruct(
        self,
        predictions: np.ndarray,
        padded_h: int,
        padded_w: int,
    ) -> np.ndarray:
        """Stitch patches back together with overlap averaging.

        Args:
            predictions: (N, patch_size, patch_size) probability maps.
            padded_h: Height of the padded image.
            padded_w: Width of the padded image.

        Returns:
            Reconstructed probability map (padded_h, padded_w).
        """
        accumulated = np.zeros(
            (padded_h, padded_w),
            dtype=np.float32,
        )
        counts = np.zeros(
            (padded_h, padded_w),
            dtype=np.float32,
        )

        n_w = (padded_w - self.patch_size) // self.step + 1
        n_h = (padded_h - self.patch_size) // self.step + 1

        idx = 0
        for i in range(n_h):
            for j in range(n_w):
                y = i * self.step
                x = j * self.step
                y_slice = slice(y, y + self.patch_size)
                x_slice = slice(x, x + self.patch_size)
                accumulated[y_slice, x_slice] += predictions[idx]
                counts[y_slice, x_slice] += 1
                idx += 1

        counts = np.maximum(counts, 1)
        return accumulated / counts

    def _remove_padding(
        self,
        array: np.ndarray,
        padding: tuple[int, int, int, int],
    ) -> np.ndarray:
        """Strip the reflection padding to restore the original size.

        Args:
            array: Padded array (H, W).
            padding: (top, bottom, left, right).

        Returns:
            Cropped array matching the original image dimensions.
        """
        top, bottom, left, right = padding
        h, w = array.shape[:2]
        y_end = h - bottom if bottom > 0 else h
        x_end = w - right if right > 0 else w
        return array[top:y_end, left:x_end]
