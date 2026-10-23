# Visual Cues for Vascular Segmentation

Code for the paper **Investigating the Visual Cues of CNNs for Vascular Segmentation: A Case Study in Microscopy and Fundus Imaging** by Weslley dos Santos Silva and Cesar Henrique Comin.

> ### Abstract
> Vascular segmentation is a standard procedure for clinical diagnosis, yet the specific visual features determining model decisions remain poorly understood. This paper investigates the visual cues Convolutional Neural Networks (CNNs) use to segment blood vessels across two distinct imaging domains: fluorescence microscopy and retinal fundus photography. We employ a series of experiments to quantify the influence of shape, texture, and receptive field on segmentation performance.
> First, we isolate texture and intensity by evaluating performance on patches subjected to pixel shuffling and normalization. Second, we assess global shape relevance by training models on sparse contours and centerlines. Lastly, we quantify the required spatial context by systematically varying the network's theoretical and effective receptive fields.
> Within the scope of the evaluated datasets, we found that pixel intensity is more relevant than texture, though networks maintain surprisingly high accuracy even when both cues are removed. Furthermore, CNNs struggle to extrapolate full vessel geometry from shape cues alone, typically relying on a relatively small effective receptive field of around 20 pixels, though global context provides a modest benefit for fundus images. While specific to the modalities studied, this methodology offers a quantitative foundation to audit and refine deep learning systems in vascular imaging.


# Repository structure

| Path | Description |
|---|---|
| `src/` (base code) | Training, prediction and evaluation code shared by all segmentation experiments. |
| `data/` | Train/validation/test splits of DRIVE and VessMAP. The datasets, trained models (`data/experiments`) and predictions (`data/results`) are also placed here. |
| [`classification/`](classification/README.md) | Classifies 9×9 patches as vessel or background after selectively removing intensity (normalization) and texture (pixel shuffling), with the goal of measuring how much each of these cues contributes to recognizing vessels without shape information. |
| `src/train.py` (`--transform erosion` or `--transform skeleton`) | Trains the network on images that keep only the contours or the centerlines of the vessels and evaluates it on the original images, with the goal of evaluating whether the network can recover the full vessel geometry from sparse shape cues. |
| `src/train_all.py` | Trains 160 U-Net configurations with different kernel sizes, dilations and pooling, with the goal of relating the theoretical and effective receptive fields of the network to its segmentation performance. |
| `src/run_windows.py` | Trains the base model on non-overlapping patches of increasing size, with the goal of finding how much spatial context is needed for good segmentation. |
| [`patch_size/`](patch_size/README.md) | Test metrics obtained by `run_windows.py` and the notebook that plots the Dice score as a function of patch size. |

The minimal instructions for running each experiment are given below. See the README of each directory for more details.

# Instructions

Download the [DRIVE](https://drive.grand-challenge.org/) and [VessMAP](https://zenodo.org/records/10045265) datasets and place them in their respective folders inside the `data` directory. 

Prepare the Python environment and install the required packages indicated in the `requirements.txt` file. 


## Patch classification experiment (texture and intensity)

This experiment does not use the segmentation code and only uses the VessMAP dataset. Run inside the `classification` directory:

```bash
python extract_patches.py --fraction 1.0
python extract_patches.py --fraction 0.75
python extract_patches.py --fraction 0.5
python train_classification.py --all
python summarize.py
```

The first three commands extract the 9×9 patches, the fourth trains all configurations of the paper (4 perturbations × 3 fractions of vessel pixels × 5 seeds) and the last one computes the mean accuracies and the statistical tests. See [`classification/`](classification/README.md) for details.

## Segmentation experiments

The experiments below use the segmentation code in `src/`.

### Single model training

To do a single training run and evaluate the model, run the following commands inside the `src` directory:

```bash
python train.py --csv_train ../data/DRIVE/train.csv --save_path dic_unet_drive --model_name dic_unet
python generate_results.py --config_file ../data/experiments/dic_unet_drive/config.cfg --dataset DRIVE
python analyze_results.py --path_preds ../data/results/dic_unet_drive --dataset DRIVE
```

You can change `DRIVE` to `VessMAP` in the commands above to train and evaluate on the VessMAP dataset. `unet`, `wnet` and `dic_unet` are the available model names. 

The first command will train the model and save it in the `data/experiments/<save_path>` folder. It will also save in the same folder a config.cfg file with the training parameters. The second command will generate predictions for all images in the dataset (training, validation and test) and save them in the `data/results/<save_path>` folder. The third command will analyze the predictions and save the performance metrics in the folder `data/results/<save_path>/perf`.

### Segmentation using shape cues

To train the model on images containing only the contours (`--transform erosion`) or the centerlines (`--transform skeleton`) of the vessels, add the `--transform` option to the training command. For example, to train the U-Net on the contours of the VessMAP dataset with the parameters of the paper and evaluate it:

```bash
python train.py --csv_train ../data/VessMAP/train.csv --save_path unet_vessmap_erosion --model_name unet --im_size 256 --in_c 1 --epochs 2500 --metric dice --transform erosion
python generate_results.py --config_file ../data/experiments/unet_vessmap_erosion/config.cfg --dataset VessMAP --im_size 256 --in_c 1
python analyze_results.py --path_preds ../data/results/unet_vessmap_erosion --dataset VessMAP
```

Use `--model_name wnet` for the W-Net.

### Receptive field variation

To train and evaluate models with all the configurations used in the paper, run the following command inside the `src` directory:

```bash
python train_all.py
```

### Segmentation performance vs. patch size

To train and evaluate the base model on non-overlapping patches of different sizes, run inside the `src` directory:

```bash
python run_windows.py
```

The figure of the paper is generated by the notebook `patch_size/analyze_windows.ipynb`. See [`patch_size/`](patch_size/README.md) for details.

The training and evaluation code of the segmentation experiments is based on the [Little W-Net](https://github.com/agaldran/lwnet) repository.