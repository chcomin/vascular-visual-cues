"""
Train, generate predictions and compute metrics for a U-Net trained on
non-overlapping windows (patches) of different sizes. Results are saved in
../data/experiments/{dataset}_window_{window_size} and ../data/results/{dataset}_window_{window_size}.
The test metrics used in the paper are in
../data/results/{dataset}_window_{window_size}/perf/test_performance.json.
"""
import os

def dict_to_argv(params):
    """
    Convert a dictionary of parameters to a list of command line arguments.
    """
    args = []
    for key, value in params.items():
        args.append(f"{key} {value}")
    return " ".join(args)

def run(dataset):

    base_bs = 4
    conf_name = ""
    experiment = "../data/experiments"
    results = "../data/results"

    # 1 channel for VessMAP, 3 channels for DRIVE
    in_c = 1 if dataset == "VessMAP" else 3
    # size 256 for VessMAP, 512 for DRIVE
    im_size = 256 if dataset == "VessMAP" else 512
    # Window sizes used in the paper. Window size 0 means no windowing (the full image is used)
    window_sizes = [256, 128, 64, 32, 16, 8, 4]
    if dataset == "DRIVE":
        window_sizes.insert(0, 512)

    params_train = {
        "--epochs": "1000",
        "--csv_train": f"../data/{dataset}/train.csv",
        "--model_name": "unet",
        "--im_size": f"{im_size}",
        "--in_c": f"{in_c}",
        "--num_workers": "9",
        "--experiment": experiment,
        # Metric used for selecting the best checkpoint
        "--metric": "dice",
    }

    params_gen = {
        "--dataset": dataset,
        "--im_size": f"{im_size}",
        "--in_c": f"{in_c}",
        "--result_path": results,
    }

    params_analyze = {
        "--dataset": dataset,
    }

    for window_size in window_sizes:
        if window_size == 0:
            bs = base_bs
        else:
            # Calculate batch size so that the same number of pixels are processed per batch
            # for each window size
            factor = im_size // window_size
            if im_size % window_size != 0:
                # +1 because when windows do not fit, there will be lots of zeros due to
                # padding
                factor += 1
            bs = factor**2*base_bs

        params_train["--window_size"] = f"{window_size}"
        params_train["--save_path"] = f"{dataset}_window_{window_size}{conf_name}"
        params_train["--batch_size"] = f"{bs}"

        params_gen["--config_file"] = f"{experiment}/{dataset}_window_{window_size}{conf_name}/config.cfg"
        params_gen["--window_size"] = f"{window_size}"

        params_analyze["--path_preds"] = f"{results}/{dataset}_window_{window_size}{conf_name}"

        cmd = f"python train.py {dict_to_argv(params_train)}"
        os.system(cmd)

        cmd = f"python generate_results.py {dict_to_argv(params_gen)}"
        os.system(cmd)

        cmd = f"python analyze_results.py {dict_to_argv(params_analyze)}"
        os.system(cmd)


if __name__ == "__main__":

    # A single training run (seed 0) was done for each window size
    run("VessMAP")
    run("DRIVE")
