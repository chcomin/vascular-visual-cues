# Visual Cues for Vascular Segmentation

Code for the paper **Investigating the Visual Cues of CNNs for Vascular Segmentation: A Case Study in Microscopy and Fundus Imaging** by Weslley dos Santos Silva and Cesar Henrique Comin.

> ### Abstract
> Vascular segmentation in medical images is a standard procedure for clinical diagnosis, yet the specific visual features determining model decisions are poorly understood. This paper presents a case study investigating the visual cues used by Convolutional Neural Networks (CNNs) for segmenting blood vessels across two distinct imaging domains: fluorescence microscopy and retinal fundus photography. We employ a series of experiments to quantify the relative influence of shape, texture, and receptive field on segmentation performance.
> First, we isolate the roles of texture and intensity by evaluating performance on patches subjected to pixel shuffling and normalization. Second, we assess the influence of global shape by training models to segment vessels using only sparse contours or centerlines as input. Lastly, we quantify the spatial context required for robust segmentation by systematically varying the network's theoretical and effective receptive fields. Within the scope of the evaluated datasets and architectures, we found that pixel intensity appears more relevant than vessel texture, though networks maintain surprisingly high accuracy even when both cues are removed. Furthermore, the results indicate that CNNs struggle to extrapolate full vessel geometry from shape cues alone and typically rely on a relatively small effective receptive field of around 20 pixels, though global context provides a modest benefit for fundus images. While these findings are specific to the modalities and models studied, the proposed methodology offers a quantitative foundation for future efforts to audit and refine deep learning systems in vascular imaging.


# Instructions

Download the [DRIVE](https://drive.grand-challenge.org/) and [VessMAP](https://zenodo.org/records/10045265) datasets and place them in their respective folders inside the `data` directory. 

Prepare the Python environment and install the required packages indicated in the `requirements.txt` file. 


## Single model training

To do a single training run and evaluate the model, run the following commands inside the `src` directory:

```bash
python train.py --csv_train ../data/DRIVE/train.csv --save_path dic_unet_drive --model_name dic_unet
python generate_results.py --config_file ../data/experiments/dic_unet_drive/config.cfg --dataset DRIVE
python analyze_results.py --path_preds ../data/results/dic_unet_drive --dataset DRIVE
```

You can change `DRIVE` to `VessMAP` in the commands above to train and evaluate on the VessMAP dataset. `unet`, `wnet` and `dic_unet` are the available model names. 

The first command will train the model and save it in the `data/experiments/<save_path>` folder. It will also save in the same folder a config.cfg file with the training parameters. The second command will generate predictions for all images in the dataset (training, validation and test) and save them in the `data/results/<save_path>` folder. The third command will analyze the predictions and save the performance metrics in the folder `data/results/<save_path>/perf`.

## Multiple models training

To train and evaluate models with all the configurations used in the paper, run the following command inside the `src` directory:

```bash
python train_all
```

The training and evaluation code is based on the [Little W-Net](https://github.com/agaldran/lwnet) repository.