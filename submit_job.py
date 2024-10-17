import yaml
import subprocess
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="config.yaml", help="Path to the configuration YAML file")
args = parser.parse_args()

# Load the YAML configuration file
with open(args.config, "r") as file:
    config = yaml.safe_load(file)


# Function to get Singularity command if enabled
def get_singularity_command(use_gpu=False):
    if config.get("use_singularity", False):
        singularity_command = [
            "singularity",
            "exec",
            "--bind",
            f"{config['path_mappings']['local_repo_dir']}:{config['path_mappings']['container_dir']}",
            "--pwd",
            config["path_mappings"]["container_dir"],
        ]
        if use_gpu:
            singularity_command.append("--nv")
        singularity_command.append(config["path_mappings"]["singularity_image"])
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
            "src/segger/cli/create_dataset_fast.py",
            "--base_dir",
            config["preprocessing"]["base_dir"],
            "--data_dir",
            config["preprocessing"]["data_dir"],
            "--sample_type",
            config["preprocessing"]["sample_type"],
            "--tile_width",
            str(config["preprocessing"]["tile_width"]),
            "--tile_height",
            str(config["preprocessing"]["tile_height"]),
            "--n_workers",
            str(config["preprocessing"]["workers"]),
        ]
    )

    if config.get("use_lsf", False):
        command = [
            "bsub",
            "-J",
            "job_data_processing",
            "-o",
            config["preprocessing"]["output_log"],
            "-n",
            str(config["preprocessing"]["workers"]),
            "-R",
            f"rusage[mem={config['preprocessing']['memory']}]",
            "-q",
            "medium",
        ] + command

    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running data processing pipeline: {e}")


# Run the training pipeline
def run_training():
    command = (
        get_singularity_command(use_gpu=True)
        + get_python_command()
        + [
            "src/segger/cli/train_model.py",
            "--dataset_dir",
            config["training"]["dataset_dir"],
            "--models_dir",
            config["training"]["models_dir"],
            "--sample_tag",
            config["training"]["sample_tag"],
            "--num_workers",
            str(config["training"]["workers"]),
            "--devices",
            str(config["training"]["gpus"]),
        ]
    )

    if config.get("use_lsf", False):
        command = [
            "bsub",
            "-J",
            "job_training",
            "-w",
            "done(job_data_processing)",
            "-o",
            config["training"]["output_log"],
            "-n",
            str(config["training"]["workers"]),
            "-R",
            f"rusage[mem={config['training']['memory']}]",
            "-R",
            "tensorcore",
            "-gpu",
            f"num={config['training']['gpus']}:j_exclusive=no:gmem={config['training']['gpu_memory']}",
            "-q",
            "gpu",
        ] + command

    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running training pipeline: {e}")


# Run the prediction pipeline
def run_prediction():
    command = (
        get_singularity_command(use_gpu=True)
        + get_python_command()
        + [
            "src/segger/cli/predict.py",
            "--segger_data_dir",
            config["prediction"]["segger_data_dir"],
            "--models_dir",
            config["prediction"]["models_dir"],
            "--benchmarks_dir",
            config["prediction"]["benchmarks_dir"],
            "--transcripts_file",
            config["prediction"]["transcripts_file"],
            "--knn_method",
            config["prediction"]["knn_method"],
            "--num_workers",
            str(config["prediction"]["workers"]),
        ]
    )

    if config.get("use_lsf", False):
        command = [
            "bsub",
            "-J",
            "job_prediction",
            "-w",
            "done(job_training)",
            "-o",
            config["prediction"]["output_log"],
            "-n",
            str(config["prediction"]["workers"]),
            "-R",
            f"rusage[mem={config['prediction']['memory']}]",
            "-R",
            "tensorcore",
            "-gpu",
            f"num=1:j_exclusive=no:gmem={config['prediction']['gpu_memory']}",
            "-q",
            "gpu",
        ] + command

    try:
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
