import yaml
import subprocess

# Load the YAML configuration file
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Helper function to wrap command with Singularity
def wrap_command_with_singularity(command):
    return [
        "singularity", "exec", "--bind",
        f"{config['paths']['local_repo_dir']}:{config['paths']['container_dir']}",
        "--pwd", config['paths']['container_dir'],
        config['paths']['singularity_image']
    ] + command

# Define the pipeline functions

# Run the data processing pipeline
def run_data_processing():
    python_command = [
        "python3", "src/segger/cli/create_dataset_fast.py",
        "--base_dir", config['preprocessing']['base_dir'],
        "--data_dir", config['preprocessing']['data_dir'],
        "--sample_type", config['preprocessing']['sample_type'],
        "--tile_width", str(config['preprocessing']['tile_width']),
        "--tile_height", str(config['preprocessing']['tile_height']),
        "--n_workers", str(config['preprocessing']['workers'])
    ]

    if config['use_singularity']:
        python_command = wrap_command_with_singularity(python_command)
    
    bsub_command = [
        "bsub", "-J", "job_data_processing", "-o", config['preprocessing']['output_log'],
        "-n", str(config['preprocessing']['workers']),
        "-R", f"rusage[mem={config['preprocessing']['memory']}]",
        "-q", "long"
    ] + python_command

    subprocess.run(bsub_command)

# Run the training pipeline
def run_training():
    python_command = [
        "python3", "src/segger/cli/train_model.py",
        "--dataset_dir", config['training']['dataset_dir'],
        "--models_dir", config['training']['models_dir'],
        "--sample_tag", config['training']['sample_tag'],
        "--num_workers", str(config['training']['workers']),
        "--devices", str(config['training']['gpus'])
    ]

    if config['use_singularity']:
        python_command = wrap_command_with_singularity(python_command)

    bsub_command = [
        "bsub", "-J", "job_training", "-w", "done(job_data_processing)",
        "-o", config['training']['output_log'],
        "-n", str(config['training']['workers']),
        "-R", f"rusage[mem={config['training']['memory']}]",
        "-R", "tensorcore",
        "-gpu", f"num={config['training']['gpus']}:j_exclusive=no:gmem={config['training']['gpu_memory']}",
        "-q", "gpu"
    ] + python_command

    subprocess.run(bsub_command)

# Run the prediction pipeline
def run_prediction():
    python_command = [
        "python3", "src/segger/cli/predict.py",
        "--segger_data_dir", config['prediction']['segger_data_dir'],
        "--benchmarks_dir", config['prediction']['benchmarks_dir'],
        "--transcripts_file", config['prediction']['transcripts_file'],
        "--knn_method", config['prediction']['knn_method'],
        "--num_workers", str(config['prediction']['workers'])
    ]

    if config['use_singularity']:
        python_command = wrap_command_with_singularity(python_command)

    bsub_command = [
        "bsub", "-J", "job_prediction", "-w", "done(job_training)",
        "-o", config['prediction']['output_log'],
        "-n", str(config['prediction']['workers']),
        "-R", f"rusage[mem={config['prediction']['memory']}]",
        "-R", "tensorcore",
        "-gpu", f"num=1:j_exclusive=no:gmem={config['prediction']['gpu_memory']}",
        "-q", "gpu"
    ] + python_command

    subprocess.run(bsub_command)

# Run the selected pipelines
pipelines = config.get('pipelines', [])
for pipeline in pipelines:
    if pipeline == 1:
        run_data_processing()
    elif pipeline == 2:
        run_training()
    elif pipeline == 3:
        run_prediction()
    else:
        print(f"Invalid pipeline number: {pipeline}")
