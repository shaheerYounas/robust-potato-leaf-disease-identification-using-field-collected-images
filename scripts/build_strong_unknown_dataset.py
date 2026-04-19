"""
Build a STRONG 'Unknown' class dataset for real-world potato leaf OOD rejection.

Three categories of negatives:
  1. Real-world objects (CIFAR-10): cars, planes, cats, dogs, trucks, ships, etc.
  2. Real textures (DTD): fabric, wood, metal, brick, carpet, etc.
  3. Synthetic fallback: noise, gradients, patterns (kept for diversity)

All images are resized to 224x224 to match training pipeline.
"""
import sys, os, shutil, random
from pathlib import Path
from PIL import Image
import numpy as np

# ── Config ─────────────────────────────────────────────────────────────
PROJECT = Path(__file__).resolve().parent.parent
DATASET_ROOT = PROJECT / "data" / "processed" / "augmented_dataset"
UNKNOWN_DIR = DATASET_ROOT / "Unknown"
TARGET_COUNT = 748  # match other classes

# How many from each source
N_CIFAR = 350       # real objects (50%)
N_DTD   = 250       # real textures (33%)
N_SYNTH = 148       # synthetic fallback (17%)
assert N_CIFAR + N_DTD + N_SYNTH == TARGET_COUNT

IMG_SIZE = (224, 224)
SEED = 42
random.seed(SEED)
np.random.seed(SEED)


def download_cifar10_negatives(n: int, out_dir: Path):
    """Download CIFAR-10 and save n random images as 224x224 PNGs."""
    print(f"  Downloading CIFAR-10 ({n} images)...")
    from torchvision.datasets import CIFAR10
    ds = CIFAR10(root=str(PROJECT / "data" / "cache"), train=True, download=True)
    # CIFAR-10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
    # ALL are non-leaf — perfect negatives
    indices = list(range(len(ds)))
    random.shuffle(indices)
    count = 0
    for idx in indices:
        if count >= n:
            break
        img, label = ds[idx]  # img is PIL 32x32
        img_resized = img.resize(IMG_SIZE, Image.LANCZOS)
        img_resized.save(out_dir / f"cifar10_{count:04d}.png")
        count += 1
    print(f"    Saved {count} CIFAR-10 images")
    return count


def download_dtd_negatives(n: int, out_dir: Path):
    """Download DTD (Describable Textures Dataset) and save n random images."""
    print(f"  Downloading DTD ({n} images)...")
    from torchvision.datasets import DTD
    ds = DTD(root=str(PROJECT / "data" / "cache"), split="train", download=True)
    indices = list(range(len(ds)))
    random.shuffle(indices)
    count = 0
    for idx in indices:
        if count >= n:
            break
        img, label = ds[idx]  # img is PIL, variable size
        img_resized = img.convert("RGB").resize(IMG_SIZE, Image.LANCZOS)
        img_resized.save(out_dir / f"dtd_{count:04d}.png")
        count += 1
    print(f"    Saved {count} DTD images")
    return count


def generate_synthetic_negatives(n: int, out_dir: Path):
    """Generate diverse synthetic non-leaf images."""
    print(f"  Generating {n} synthetic negatives...")
    count = 0
    w, h = IMG_SIZE

    def save(arr, name):
        nonlocal count
        Image.fromarray(arr.astype(np.uint8)).save(out_dir / f"synth_{name}_{count:04d}.png")
        count += 1

    # Uniform random noise (like camera sensor noise)
    for _ in range(n // 6):
        save(np.random.randint(0, 256, (h, w, 3), dtype=np.uint8), "noise")

    # Smooth color gradients (like sky/wall photos)
    for _ in range(n // 6):
        c1, c2 = np.random.randint(0, 256, 3), np.random.randint(0, 256, 3)
        t = np.linspace(0, 1, h).reshape(-1, 1, 1)
        arr = (c1 * (1 - t) + c2 * t).astype(np.uint8)
        arr = np.broadcast_to(arr, (h, w, 3)).copy()
        save(arr, "grad")

    # Random geometric shapes on solid backgrounds (like icons/UI)
    for _ in range(n // 6):
        bg = np.full((h, w, 3), np.random.randint(0, 256, 3), dtype=np.uint8)
        for _ in range(random.randint(3, 10)):
            x1, y1 = random.randint(0, w-20), random.randint(0, h-20)
            x2, y2 = min(x1+random.randint(10, 80), w), min(y1+random.randint(10, 80), h)
            bg[y1:y2, x1:x2] = np.random.randint(0, 256, 3)
        save(bg, "shapes")

    # Perlin-like blobs (compressed JPEG artifacts, out of focus)
    for _ in range(n // 6):
        from PIL import ImageFilter
        arr = np.random.randint(0, 256, (h // 4, w // 4, 3), dtype=np.uint8)
        img = Image.fromarray(arr).resize(IMG_SIZE, Image.BILINEAR)
        img = img.filter(ImageFilter.GaussianBlur(radius=8))
        save(np.array(img), "blur")

    # Solid colors with text-like patterns
    for _ in range(n // 6):
        bg = np.full((h, w, 3), np.random.randint(0, 256, 3), dtype=np.uint8)
        # add horizontal line patterns (like text)
        for y in range(0, h, random.randint(8, 20)):
            thickness = random.randint(1, 4)
            length = random.randint(w // 4, w)
            x_start = random.randint(0, w - length)
            bg[y:y+thickness, x_start:x_start+length] = np.random.randint(0, 256, 3)
        save(bg, "text")

    # Fill remaining
    while count < n:
        # Checkerboard / grid patterns
        block = random.randint(8, 32)
        arr = np.zeros((h, w, 3), dtype=np.uint8)
        c1, c2 = np.random.randint(0, 256, 3), np.random.randint(0, 256, 3)
        for i in range(0, h, block):
            for j in range(0, w, block):
                c = c1 if ((i // block) + (j // block)) % 2 == 0 else c2
                arr[i:i+block, j:j+block] = c
        save(arr, "grid")

    print(f"    Saved {count} synthetic images")
    return count


def main():
    print("=== Building Strong Unknown Dataset ===")
    print(f"Target: {TARGET_COUNT} images ({N_CIFAR} CIFAR-10 + {N_DTD} DTD + {N_SYNTH} synthetic)")

    # Clear old Unknown folder
    if UNKNOWN_DIR.exists():
        old_count = len(list(UNKNOWN_DIR.glob("*")))
        print(f"Removing old Unknown folder ({old_count} files)...")
        shutil.rmtree(UNKNOWN_DIR)
    UNKNOWN_DIR.mkdir(parents=True, exist_ok=True)

    total = 0

    # 1. CIFAR-10 real objects
    total += download_cifar10_negatives(N_CIFAR, UNKNOWN_DIR)

    # 2. DTD real textures
    total += download_dtd_negatives(N_DTD, UNKNOWN_DIR)

    # 3. Synthetic fallback
    total += generate_synthetic_negatives(N_SYNTH, UNKNOWN_DIR)

    print(f"\n=== Done! {total} images in {UNKNOWN_DIR} ===")
    print("Breakdown:")
    for prefix in ["cifar10", "dtd", "synth"]:
        c = len(list(UNKNOWN_DIR.glob(f"{prefix}_*")))
        print(f"  {prefix}: {c}")


if __name__ == "__main__":
    main()
