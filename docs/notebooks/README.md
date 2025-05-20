# README


Set the environment

```
conda create -n segger python=3.8
conda activate segger
pip install -r requirements.txt
```



Download files from a remote server using bash command
```
scp -r cuisen@linyi:/home/cuisen/users/fengyun/segger/docs/notebooks/figures/tobedownload /Users/finleyyu/Desktop/Germany/Master_thesis_2025/segger/docs/notebooks/figures
```

Interactive gene expression plots

```
cd docs/notebooks

streamlit run visualization/interactive_attention.py
```