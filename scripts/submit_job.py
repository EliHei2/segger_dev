import yaml
import subprocess
import argparse
import os
import time

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="config.yaml", help="Path to the configuration YAML file")
args = parser.parse_args()

script_dir = os.path.dirname(os.path.realpath(__file__))
config_file_path = os.path.join(script_dir, args.config)

# Load the YAML configuration file
with open(config_file_path, "r") as file:
    config = yaml.safe_load(file)

# Get the base directory
repo_dir = config["container_dir"] if config.get("use_singularity", False) else config["local_repo_dir"]

time_stamp = time.strftime("%Y%m%d-%H%M%S")


# Function to get Singularity command if enabled
def get_singularity_command(use_gpu=False):
    if config.get("use_singularity", False):
        singularity_command = [
            "singularity",
            "exec",
            "--bind",
            f"{config['local_repo_dir']}:{config['container_dir']}",
            "--pwd",
            config["container_dir"],
        ]
        if use_gpu:
            singularity_command.append("--nv")
        singularity_command.append(config["singularity_image"])
        return singularity_command
    return []  # Return an empty list if Singularity is not enabled


# Function to get Python command
def get_python_command():
    return (
        ["python3", "-m", "debugpy", "--listen", "0.0.0.0:5678", "--wait-for-client"]
        if config.get("use_debugpy", False)
        else ["python3"]
    )


# Define the pipeline functions


# Run the data processing pipeline
def run_data_processing():
    command = (
        get_singularity_command(use_gpu=False)
        + get_python_command()
        + [
            f"{repo_dir}/src/segger/cli/create_dataset_fast.py",
            "--base_dir",
            config["preprocessing"]["base_dir"],
            "--data_dir",
            config["preprocessing"]["data_dir"],
            "--sample_type",
            config["preprocessing"]["sample_type"],
            "--k_bd",
            str(config["preprocessing"]["k_bd"]),
            "--dist_bd",
            str(config["preprocessing"]["dist_bd"]),
            "--k_tx",
            str(config["preprocessing"]["k_tx"]),
            "--dist_tx",
            str(config["preprocessing"]["dist_tx"]),
            "--neg_sampling_ratio",
            str(config["preprocessing"]["neg_sampling_ratio"]),
            "--frac",
            str(config["preprocessing"]["frac"]),
            "--val_prob",
            str(config["preprocessing"]["val_prob"]),
            "--test_prob",
            str(config["preprocessing"]["test_prob"]),
            "--n_workers",
            str(config["preprocessing"]["n_workers"]),
        ]
    )

    if config["preprocessing"].get("tile_size") is not None:
        command.extend(["--tile_size", str(config["preprocessing"]["tile_size"])])
    if config["preprocessing"].get("tile_width") is not None:
        command.extend(["--tile_width", str(config["preprocessing"]["tile_width"])])
    if config["preprocessing"].get("tile_height") is not None:
        command.extend(["--tile_height", str(config["preprocessing"]["tile_height"])])
    if config["preprocessing"].get("scrnaseq_file") is not None:
        command.extend(["--scrnaseq_file", config["preprocessing"]["scrnaseq_file"]])
    if config["preprocessing"].get("celltype_column") is not None:
        command.extend(["--celltype_column", config["preprocessing"]["celltype_column"]])

    if config.get("use_lsf", False):
        command = [
            "bsub",
            "-J",
            f"job_data_processing_{time_stamp}",
            "-o",
            config["preprocessing"]["output_log"],
            "-n",
            str(config["preprocessing"]["n_workers"]),
            "-R",
            f"rusage[mem={config['preprocessing']['memory']}]",
            "-q",
            "long",
        ] + command

    try:
        print(f"Running command: {command}")
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running data processing pipeline: {e}")


# Run the training pipeline
def run_training():
    command = (
        get_singularity_command(use_gpu=True)
        + get_python_command()
        + [
            f"{repo_dir}/src/segger/cli/train_model.py",
            "--dataset_dir",
            config["training"]["dataset_dir"],
            "--models_dir",
            config["training"]["models_dir"],
            "--sample_tag",
            config["training"]["sample_tag"],
            "--init_emb",
            str(config["training"]["init_emb"]),
            "--hidden_channels",
            str(config["training"]["hidden_channels"]),
            "--num_tx_tokens",
            str(config["training"]["num_tx_tokens"]),
            "--out_channels",
            str(config["training"]["out_channels"]),
            "--heads",
            str(config["training"]["heads"]),
            "--num_mid_layers",
            str(config["training"]["num_mid_layers"]),
            "--batch_size",
            str(config["training"]["batch_size"]),
            "--num_workers",
            str(config["training"]["num_workers"]),
            "--accelerator",
            config["training"]["accelerator"],
            "--max_epochs",
            str(config["training"]["max_epochs"]),
            "--devices",
            str(config["training"]["devices"]),
            "--strategy",
            config["training"]["strategy"],
            "--precision",
            config["training"]["precision"],
        ]
    )

    if config.get("use_lsf", False):
        command = [
            "bsub",
            "-J",
            f"job_training_{time_stamp}",
            "-o",
            config["training"]["output_log"],
            "-n",
            str(config["training"]["num_workers"]),
            "-R",
            f"rusage[mem={config['training']['memory']}]",
            "-R",
            "tensorcore",
            "-gpu",
            f"num={config['training']['devices']}:j_exclusive=no:gmem={config['training']['gpu_memory']}",
            "-q",
            "gpu",
        ] + command
        # only run training after data_processing
        if 1 in config["pipelines"]:
            command[3:3] = ["-w", f"done(job_data_processing_{time_stamp})"]

    try:
        print(f"Running command: {command}")
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running training pipeline: {e}")


# Run the prediction pipeline
def run_prediction():
    command = (
        get_singularity_command(use_gpu=True)
        + get_python_command()
        + [
            f"{repo_dir}/src/segger/cli/predict_fast.py",
            "--segger_data_dir",
            config["prediction"]["segger_data_dir"],
            "--models_dir",
            config["prediction"]["models_dir"],
            "--benchmarks_dir",
            config["prediction"]["benchmarks_dir"],
            "--transcripts_file",
            config["prediction"]["transcripts_file"],
            "--batch_size",
            str(config["prediction"]["batch_size"]),
            "--num_workers",
            str(config["prediction"]["num_workers"]),
            "--model_version",
            str(config["prediction"]["model_version"]),
            "--save_tag",
            config["prediction"]["save_tag"],
            "--min_transcripts",
            str(config["prediction"]["min_transcripts"]),
            "--cell_id_col",
            str(config["prediction"]["cell_id_col"]),
            "--use_cc",
            str(config["prediction"]["use_cc"]),
            "--knn_method",
            config["prediction"]["knn_method"],
            "--file_format",
            config["prediction"]["file_format"],
            "--k_bd",
            str(config["prediction"]["k_bd"]),
            "--dist_bd",
            str(config["prediction"]["dist_bd"]),
            "--k_tx",
            str(config["prediction"]["k_tx"]),
            "--dist_tx",
            str(config["prediction"]["dist_tx"]),
        ]
    )

    if config.get("use_lsf", False):
        command = [
            "bsub",
            "-J",
            f"job_prediction_{time_stamp}",
            "-o",
            config["prediction"]["output_log"],
            "-n",
            str(config["prediction"]["num_workers"]),
            "-R",
            f"rusage[mem={config['prediction']['memory']}]",
            "-R",
            "tensorcore",
            "-gpu",
            f"num=1:j_exclusive=no:gmem={config['prediction']['gpu_memory']}",
            "-q",
            "gpu",
        ] + command
        # only run prediction after training/data_processing
        if 2 in config["pipelines"]:
            command[3:3] = ["-w", f"done(job_training_{time_stamp})"]
        elif 1 in config["pipelines"]:
            command[3:3] = ["-w", f"done(job_data_processing_{time_stamp})"]

    try:
        print(f"Running command: {command}")
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running prediction pipeline: {e}")


# Run the selected pipelines
pipelines = config.get("pipelines", [])
for pipeline in pipelines:
    if pipeline == 1:
        run_data_processing()
    elif pipeline == 2:
        run_training()
    elif pipeline == 3:
        run_prediction()
    else:
        print(f"Invalid pipeline number: {pipeline}")
