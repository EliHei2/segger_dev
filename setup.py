from setuptools import setup, find_packages

setup(
    name='segger_dev',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy==1.23.5',
        'pandas==1.5.3',
        'scipy==1.10.1',
        'scanpy==1.9.1',
        'matplotlib==3.7.1',
        'seaborn==0.12.2',
        'squidpy==1.1.1',
        'torch==1.13.1',
        'torchvision==0.14.1',
        'pytorch-lightning==2.0.0',
        'torch-geometric==2.2.0',
        'torchmetrics==0.11.4',
        'zarr==2.14.1',
        'tqdm==4.65.0',
        'pathlib==1.0.1',
        'scikit-learn==1.2.2',
        'adjustText==0.8.3',
        'argparse==1.4.0'
    ],
    entry_points={
        'console_scripts': [
            'create_dataset=scripts.create_dataset:main',
        ],
    },
)
