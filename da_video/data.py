from __future__ import annotations

import gzip
import struct
import urllib.request
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.datasets import load_digits
from torch.utils.data import Dataset


MNIST_IMAGE_URLS = {
    "train": (
        "https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz",
        "train-images-idx3-ubyte.gz",
    ),
    "val": (
        "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
    ),
    "test": (
        "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
    ),
}


def _download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    urllib.request.urlretrieve(url, destination)


def _read_idx_images(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as handle:
        magic, num_images, rows, cols = struct.unpack(">IIII", handle.read(16))
        if magic != 2051:
            raise ValueError(f"Unexpected IDX magic number {magic} in {path}")
        data = np.frombuffer(handle.read(), dtype=np.uint8)
    return data.reshape(num_images, rows, cols)


def _resize_digits(images: np.ndarray, size: int = 28) -> np.ndarray:
    resized = []
    for image in images:
        pil_image = Image.fromarray(image)
        resized_image = pil_image.resize((size, size), Image.Resampling.BILINEAR)
        resized.append(np.asarray(resized_image, dtype=np.float32))
    resized_array = np.stack(resized, axis=0)
    return resized_array / max(resized_array.max(), 1.0)


def load_digit_bank(root: str | Path, split: str, source: str = "auto") -> np.ndarray:
    if split not in {"train", "val", "test"}:
        raise ValueError(f"Unsupported split: {split}")

    root = Path(root)
    if source not in {"auto", "mnist", "sklearn"}:
        raise ValueError(f"Unsupported source: {source}")

    if source in {"auto", "mnist"}:
        try:
            url, filename = MNIST_IMAGE_URLS[split]
            archive_path = root / "mnist" / "raw" / filename
            _download_file(url, archive_path)
            images = _read_idx_images(archive_path).astype(np.float32) / 255.0
            return images
        except Exception:
            if source == "mnist":
                raise

    digits = load_digits().images.astype(np.float32)
    digits = _resize_digits((digits / digits.max() * 255.0).astype(np.uint8))
    split_index = int(0.8 * len(digits))
    if split == "train":
        return digits[:split_index]
    return digits[split_index:]


class MovingMNISTDataset(Dataset):
    def __init__(
        self,
        root: str | Path = "data",
        split: str = "train",
        num_sequences: int = 10_000,
        seq_len: int = 20,
        image_size: int = 64,
        digit_size: int = 28,
        num_digits: int = 2,
        seed: int = 42,
        source: str = "auto",
        speed_range: tuple[float, float] = (2.0, 4.0),
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.num_sequences = num_sequences
        self.seq_len = seq_len
        self.image_size = image_size
        self.digit_size = digit_size
        self.num_digits = num_digits
        self.seed = seed
        self.source = source
        self.speed_range = speed_range
        self.digit_bank = load_digit_bank(self.root, split=split, source=source)

    def __len__(self) -> int:
        return self.num_sequences

    def _make_rng(self, index: int) -> np.random.Generator:
        split_offset = {"train": 0, "val": 100_000, "test": 200_000}[self.split]
        return np.random.default_rng(self.seed + split_offset + index)

    def _sample_digit(self, rng: np.random.Generator) -> np.ndarray:
        digit_index = int(rng.integers(0, len(self.digit_bank)))
        digit = self.digit_bank[digit_index]
        if digit.shape[0] != self.digit_size:
            digit = _resize_digits((digit[None] * 255.0).astype(np.uint8), self.digit_size)[0]
        return digit.astype(np.float32)

    def _sample_velocity(self, rng: np.random.Generator) -> np.ndarray:
        speed = rng.uniform(*self.speed_range)
        angle = rng.uniform(0.0, 2.0 * np.pi)
        return np.array([np.cos(angle), np.sin(angle)], dtype=np.float32) * speed

    def _bounce(self, position: np.ndarray, velocity: np.ndarray, limit: float) -> tuple[np.ndarray, np.ndarray]:
        next_position = position + velocity
        for axis in range(2):
            if next_position[axis] <= 0.0:
                next_position[axis] = -next_position[axis]
                velocity[axis] *= -1.0
            elif next_position[axis] >= limit:
                next_position[axis] = 2.0 * limit - next_position[axis]
                velocity[axis] *= -1.0
        return next_position, velocity

    def _render_sequence(self, rng: np.random.Generator) -> np.ndarray:
        canvas_limit = self.image_size - self.digit_size
        digits = [self._sample_digit(rng) for _ in range(self.num_digits)]
        positions = [rng.uniform(0.0, canvas_limit, size=2).astype(np.float32) for _ in range(self.num_digits)]
        velocities = [self._sample_velocity(rng) for _ in range(self.num_digits)]

        frames = np.zeros((self.seq_len, self.image_size, self.image_size), dtype=np.float32)
        for t in range(self.seq_len):
            frame = np.zeros((self.image_size, self.image_size), dtype=np.float32)
            for idx, digit in enumerate(digits):
                y, x = positions[idx].round().astype(np.int32)
                frame[y : y + self.digit_size, x : x + self.digit_size] = np.maximum(
                    frame[y : y + self.digit_size, x : x + self.digit_size],
                    digit,
                )
                positions[idx], velocities[idx] = self._bounce(positions[idx], velocities[idx], canvas_limit)
            frames[t] = frame
        return frames

    def __getitem__(self, index: int) -> torch.Tensor:
        rng = self._make_rng(index)
        sequence = self._render_sequence(rng)
        return torch.from_numpy(sequence[:, None, :, :])

