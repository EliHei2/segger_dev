

from segger.validation.xenium_explorer import seg2explorer
import pandas as pd


transcripts_file = 'data_tidy/benchmarks/human_CRC_seg_nuclei/human_CRC_seg_nuclei_0.4_False_4_15_5_3_20250521/segger_transcripts.parquet'
transcripts_df = pd.read_parquet(transcripts_file)
# transcripts_df = transcripts_df.iloc[:10000]
seg2explorer(
    seg_df=transcripts_df, # this is your segger output
    source_path="/dkfz/cluster/gpu/data/OE0606/elihei/segger_experiments/data_raw/xenium_seg_kit/human_CRC_real", #raw xenium data
    output_dir="data_tidy/explorer/human_CRC_real_nuclei", #where you wanna save your xneium explorer file, could be the same as raw
    cells_filename="seg_cells1", #file names for cells.zarr
    analysis_filename="seg_analysis1", #file names for analysis.zarr
    xenium_filename="seg_experiment1.xenium", #xenium explorer file
    analysis_df=None,
    cell_id_columns="segger_cell_id", # segger cell id column in transcripts_df
)



XENIUM_DATA_DIR = Path( #raw data dir
    "/dkfz/cluster/gpu/data/OE0606/elihei/segger_experiments/data_raw/xenium_seg_kit/human_CRC_real"
)
transcripts_file = (
   XENIUM_DATA_DIR / "transcripts.parquet"
)

SEGGER_DATA_DIR = Path("data_tidy/pyg_datasets/human_CRC_seg_nuclei") # preprocessed data dir


seg_tag = "human_CRC_seg_nuclei"
model_version = 0
models_dir = Path("./models") / seg_tag #trained model dir


output_dir = Path( #output dir
    "/dkfz/cluster/gpu/data/OE0606/elihei/segger_experiments/data_tidy/benchmarks/human_CRC_seg_nuclei"
)

