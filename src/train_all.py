import os

def generate_combinations(base_values, new_values, flag = "kernels"):
    """Generate combinations of model configurations by replacing values in the base list with the values
    in the new_values."""

    n_layers = len(base_values)
    all_confs = {"base": base_values}

    for v in new_values:
        # [CH1] Change only one layer
        for idx in range(n_layers):
            values = base_values[:]  
            values[idx] = v
            dic = {f"CH1.{v}.{idx}": values}
            all_confs.update(dic)

        # [CH2] Change all layers up to idx
        values = base_values[:]
        for idx in range(n_layers):
            values[idx] = v
            dic = {f"CH2.{v}.{idx}": values[:]}
            all_confs.update(dic)

        if flag == "kernels":
            # [CH3] Change first conv of residual blocks
            values = base_values[:]
            for idx in range(n_layers):
                if idx%2==0:
                    values[idx] = v
            dic = {f"CH3.{v}.{1}": values[:]}
            all_confs.update(dic)

            # [CH4] Change second conv of residual blocks
            values = base_values[:]
            for idx in range(n_layers):
                if idx%2==1:
                    values[idx] = v
            dic = {f"CH4.{v}.{1}": values[:]}
            all_confs.update(dic)

            # [CH5] Change each residual block
            for idx in range(0, n_layers, 2):
                values = base_values[:]
                values[idx] = v
                values[idx+1] = v
                dic = {f"CH5.{v}.{idx}": values[:]}
                all_confs.update(dic)

            # [CH6] Change encoder
            values = base_values[:]
            for idx in range(0, 6):
                values[idx] = v
            dic = {f"CH6.{v}.{idx}": values[:]}
            all_confs.update(dic)

            # Change decoder
            values = base_values[:]
            for idx in range(6, n_layers):
                values[idx] = v
            dic = {f"CH7.{v}.{idx}": values[:]}
            all_confs.update(dic)
        
        elif flag == "residuals":
            # Change encoder residuals
            values = base_values[:]
            values[1] = v
            values[2] = v
            dic = {f"CH3.{v}.{idx}": values[:]}
            all_confs.update(dic)

            # Change decoder residuals
            values = base_values[:]
            values[3] = v
            values[4] = v
            dic = {f"CH4.{v}.{idx}": values[:]}
            all_confs.update(dic)
    
    return all_confs

def generate_configurations(base_k, base_dil, base_res, k_vals, dil_vals, res_vals, return_base = True):
    """Generate model configurations based on kernel sizes, dilation rates, and residual block configurations."""

    # Kernel configurations
    all_kernels = generate_combinations(base_k, k_vals)

    # Dilation configurations
    all_dilations = generate_combinations(base_dil, dil_vals)
    
    # Residual configurations
    all_residuals = generate_combinations(base_res, res_vals, flag = "residuals")

    # Downs configurations
    all_downs = [(2 ,2), (2, 1), (1, 2), (1, 1)]

    # Only for visualization
    base_dil = tuple(base_dil)
    base_k = tuple(base_k)
    base_downs = (2, 2)
    base_res = tuple(base_res)
    base_config = (base_k, base_dil, base_res, base_downs)
    if return_base == True:
        return [base_config]
    
    bases_kernels = [base_k]
    for v in k_vals:
        bases_kernels.append(all_kernels[f"CH2.{v}.9"])


    all_configs = []
    for kernels in all_kernels.values():
        all_configs.append((kernels, base_dil, base_res, base_downs))
        
    for dilations in all_dilations.values():
        all_configs.append((base_k, dilations, base_res, base_downs))
    
    #for residuals in all_residuals.values():
    #    all_configs.append((base_k, base_dil, residuals, base_downs))
    
    for downs in all_downs:
        all_configs.append((base_k, base_dil, base_res, downs))
                    
    return all_configs

def train_all(configs:list, seed:int = 0, transform = None, data = "DRIVE", model = "dic_unet"):
    """Train models for all architecture configurations and generate results."""
    
    csv_train = f"../data/{data}/train.csv"
    experiment = f"../data/experiments/{data}"
    results = f"../data/results/{data}"
    im_size = 256 if data == "VessMAP" else 512
    in_c = 1 if data == "VessMAP" else 3
    for idx in range(0, len(configs)):
        config = configs[idx]
        arch_conf_str = str(config)
        model_id = idx+1
        
        print("************* ID", model_id)
        print(arch_conf_str)
        cmd = f"python train.py --csv_train {csv_train} --seed {seed} --save_path {model_id} --model_id {model_id} --im_size {im_size} --in_c {in_c} --experiment {experiment} --model_name {model} --evaluate_every 200 --arch_conf_str \"{arch_conf_str}\" --transform \"{transform}\" " #--epochs 100 --evaluate_every 100"
        os.system(cmd)

        cmd = f"python generate_results.py --config_file {experiment}/{model_id}/config{seed}.cfg --dataset {data} --seed {seed} --im_size {im_size} --in_c {in_c} --result_path {results}"
        os.system(cmd)

        cmd = f"python analyze_results.py --path_preds {results}/{model_id} --dataset {data} --seed {seed}"
        os.system(cmd)

def train_all_transform(configs:list, seed:int = 0, transform = None, data = "DRIVE", model = "dic_unet"):
    """Train models for all architecture configurations with a specific transformation and generate results."""
    
    csv_train = f"../data/{data}/train.csv"
    experiment = f"../data/experiments/{data}"
    results = f"../data/results/{data}"
    im_size = 256 if data == "VessMAP" else 512
    in_c = 1 if data == "VessMAP" else 3
    config = configs[0]

    arch_conf_str = str(config)
    model_id = transform
               
    print("************* ID", model_id)
    print(arch_conf_str)
    cmd = f"python train.py --csv_train {csv_train} --seed {seed} --save_path {model_id} --model_id {model_id} --im_size {im_size} --in_c {in_c} --experiment {experiment} --model_name {model} --evaluate_every 200 --arch_conf_str \"{arch_conf_str}\" --transform \"{transform}\" " #--epochs 100 --evaluate_every 100"
    os.system(cmd)

    cmd = f"python generate_results.py --config_file {experiment}/{model_id}/config{seed}.cfg --dataset {data} --seed {seed} --im_size {im_size} --in_c {in_c} --result_path {results}"
    os.system(cmd)

    cmd = f"python analyze_results.py --path_preds {results}/{model_id} --dataset {data} --seed {seed}"
    os.system(cmd)


def main():

    USE_TRANSFORM = False

    # Kernel sizes
    n_layers = 10 
    base_k = [3]*n_layers
    # Kernel variations to test
    k_vals = [1, 5, 7]

    # Dilation values
    base_dil = [1]*n_layers
    # Dilation variations to test
    dil_vals = [2, 3, 7]

    # Kernel sizes for residual blocks. Not used in the paper.
    base_res = [1]*5 
    res_vals = [] #[3, 5, 7]

    configs = generate_configurations(base_k, base_dil, base_res, k_vals, dil_vals, res_vals, return_base = False)

    for seed in [2, 4, 8]:
        if USE_TRANSFORM:
            train_all_transform(configs, seed = seed, data = "DRIVE", model="dic_unet", transform='erosion')
        else:
            train_all(configs, seed = seed, data = "DRIVE", model="dic_unet")

        

main()