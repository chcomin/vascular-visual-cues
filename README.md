# Visual Cues for Vascular Segmentation

Code for the paper **Investigating the Visual Cues of CNNs for Vascular Segmentation: A Case Study in Microscopy and Fundus Imaging** by Weslley dos Santos Silva and Cesar Henrique Comin.

> ### Abstract
> Vascular segmentation is a standard procedure for clinical diagnosis, yet the specific visual features determining model decisions remain poorly understood. This paper investigates the visual cues Convolutional Neural Networks (CNNs) use to segment blood vessels across two distinct imaging domains: fluorescence microscopy and retinal fundus photography. We employ a series of experiments to quantify the influence of shape, texture, and receptive field on segmentation performance.
> First, we isolate texture and intensity by evaluating performance on patches subjected to pixel shuffling and normalization. Second, we assess global shape relevance by training models on sparse contours and centerlines. Lastly, we quantify the required spatial context by systematically varying the network's theoretical and effective receptive fields.
> Within the scope of the evaluated datasets, we found that pixel intensity is more relevant than texture, though networks maintain surprisingly high accuracy even when both cues are removed. Furthermore, CNNs struggle to extrapolate full vessel geometry from shape cues alone, typically relying on a relatively small effective receptive field of around 20 pixels, though global context provides a modest benefit for fundus images. While specific to the modalities studied, this methodology offers a quantitative foundation to audit and refine deep learning systems in vascular imaging.


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