import yaml
import subprocess

# Load the YAML configuration file
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)


# Define the pipeline functions
def run_data_processing():
    subprocess.run(
        [
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
            "long",
            "singularity",
            "exec",
            "--bind",
            f"{config['paths']['local_repo_dir']}:{config['paths']['container_dir']}",
            "--pwd",
            config["paths"]["container_dir"],
            config["paths"]["singularity_image"],
            "python3",
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


def run_training():
    subprocess.run(
        [
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
            "singularity",
            "exec",
            "--nv",
            "--bind",
            f"{config['paths']['local_repo_dir']}:{config['paths']['container_dir']}",
            "--pwd",
            config["paths"]["container_dir"],
            config["paths"]["singularity_image"],
            "python3",
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


def run_prediction():
    subprocess.run(
        [
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
            "singularity",
            "exec",
            "--nv",
            "--bind",
            f"{config['paths']['local_repo_dir']}:{config['paths']['container_dir']}",
            "--pwd",
            config["paths"]["container_dir"],
            config["paths"]["singularity_image"],
            "python3",
            "src/segger/cli/predict.py",
            "--segger_data_dir",
            config["prediction"]["segger_data_dir"],
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


# Main script logic
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
