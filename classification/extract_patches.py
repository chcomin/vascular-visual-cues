"""Extract the 9x9 vessel and background patches used in the classification experiment.

For each image, the label (or its complement, for background patches) is convolved with
a filter filled with ones. Pixels whose output equals int(patch_size**2 * fraction) are
candidate patch centers. Up to `num_patches` centers far enough from each other are
randomly drawn per image.

Example:
    python extract_patches.py --fraction 1.0
    python extract_patches.py --fraction 0.75
    python extract_patches.py --fraction 0.5
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from PIL import Image


def ones_convolution(img, size):
    """Convolve a binary image with a size x size filter filled with ones."""
    kernel = torch.ones(1, 1, size, size)
    img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
    return functional.conv2d(img, kernel, padding="same")


def add_if_far(points, point, size=9):
    """Append `point` to `points` if it is far enough from all points already selected."""
    for p in points:
        distance_r = np.abs(point[0] - p[0])
        distance_c = np.abs(point[1] - p[1])
        if distance_c < size or distance_r < size or (distance_r < 5 and distance_c < 5):
            return points
    points.append(point)
    return points


def read_label(path):
    """Read a label image as a 0/1 array. Works for PNG and TIFF files."""
    return (np.array(Image.open(path)) > 0).astype(int)


def extract_patches(images_dir, labels_dir, target="vessel", size=9, num_patches=7, fraction=1.0, seed=2):
    """Extract patches from all images in `images_dir`.

    Returns
    -------
    patches : np.ndarray
        Array of shape (n, size, size) with the original intensities.
    names : list of str
        Identifier of each patch, in the format "<image> <class> - <index>".
    """

    images = np.sort(os.listdir(images_dir))
    class_name = "vaso" if target == "vessel" else "fundo"
    k = size//2

    patches, names = [], []
    for name in images:
        stem = name.split(".")[0]
        img = np.array(Image.open(Path(images_dir) / name))
        label = read_label(Path(labels_dir) / f"{stem}.png")

        if target != "vessel":
            label = np.logical_not(label).astype(int)

        # Candidate centers: windows having the expected number of target pixels
        conv = ones_convolution(label, size)
        cond = (conv == int((size**2)*fraction)).long()
        indices = np.array(cond.nonzero(as_tuple=False)[:, 1:3])
        if len(indices) == 0:
            print(f"Could not extract patches from image {stem}")
            continue

        # Keep only centers whose window fits inside the image
        shape = [np.max(indices[:, 0]), np.max(indices[:, 1])]
        indices = indices[np.all(np.greater(indices, [size, size]), axis=1)]
        indices = indices[np.all(np.less_equal(indices, [shape[0]-size, shape[1]-size]), axis=1)]
        if len(indices) == 0:
            print(f"Could not extract patches from image {stem}")
            continue

        # Random selection of centers that are far from each other
        np.random.seed(seed)
        rand = np.random.randint(0, len(indices))
        points = [list(indices[rand])]
        cont = 0
        while cont < len(indices) and len(points) < num_patches:
            cont += 1
            rand = np.random.randint(0, len(indices), size=1)[0]
            points = add_if_far(points, list(indices[rand]), size)

        for idx, p in enumerate(points):
            patches.append(img[p[0]-k:p[0]+k+1, p[1]-k:p[1]+k+1])
            names.append(f"{stem} {class_name} - {idx}")

    print(f"{target}: {len(patches)} patches")

    return np.array(patches), names


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images_dir", default="../data/VessMAP/images")
    parser.add_argument("--labels_dir", default="../data/VessMAP/annotator1/labels")
    parser.add_argument("--output_dir", default="patches")
    parser.add_argument("--fraction", type=float, default=1.0, help="Fraction f of vessel pixels in the vessel patches")
    parser.add_argument("--size", type=int, default=9, help="Patch size")
    parser.add_argument("--num_patches", type=int, default=7, help="Maximum number of patches per image")
    parser.add_argument("--seed", type=int, default=2)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    patches, names = extract_patches(args.images_dir, args.labels_dir, "vessel", args.size,
                                     args.num_patches, args.fraction, args.seed)
    np.savez(output_dir / f"vessel_f{int(args.fraction*100)}.npz", patches=patches, names=names)

    # Background patches only contain background pixels
    patches, names = extract_patches(args.images_dir, args.labels_dir, "background", args.size,
                                     args.num_patches, 1.0, args.seed)
    np.savez(output_dir / "background.npz", patches=patches, names=names)


if __name__ == "__main__":
    main()
