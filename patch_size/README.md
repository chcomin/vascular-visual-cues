# Segmentation performance as a function of patch size

This experiment restricts the spatial context available to the network by training it on patches of different sizes, with the goal of finding the patch size from which the segmentation performance stops improving. It complements the receptive field experiment (`src/train_all.py`) with a direct measure of the required context.

Each sample of the VessMAP and DRIVE datasets is divided into non-overlapping patches of size $W\times W$, with $W \in \{4, 8, 16, 32, 64, 128, 256, 512\}$ (512 only for DRIVE), and the base model is trained to segment the vessels. Training parameters, data splits and augmentations are the same as in the receptive field experiment (`src/train_all.py`), except for the batch size, which is set to

$$B = 4\left(\frac{H}{W}\right)^2,$$

where $H=256$ for VessMAP and $H=512$ for DRIVE, so that every batch contains the same number of pixels.

## Running

The experiment uses the same training code as the other segmentation experiments, through the `--window_size` option of `src/train.py` and `src/generate_results.py`. To train and evaluate all patch sizes, run inside the `src` directory:

```bash
python run_windows.py
```

Each patch size is trained once (seed 0) for 1000 epochs, and the model with the highest validation Dice score is used for testing. The test metrics are saved in `data/results/{dataset}_window_{W}/perf/test_performance.json`. A GPU is recommended: on a CPU, the run with $W=4$ alone takes about 3 days.

## Files

- `patch_size_data.pkl`: test metrics (`auc`, `acc`, `dice/F1`, `MCC`, `spec`, `sens`) for each patch size obtained in the paper, as a dict with one pandas DataFrame per dataset (`"VessMAP"` and `"DRIVE"`), indexed by $W$.
- `analyze_windows.ipynb`: generates `dice_vs_patch_size.pdf` from `patch_size_data.pkl`.
