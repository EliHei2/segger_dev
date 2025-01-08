df_path = Path("/omics/groups/OE0606/internal/gleb/xenium/elyas/rep1/baysor_rep1.csv")
df = dd.read_csv(df_path)

res = compute_cells_per_nucleus(df, new_cell_col="baysor_cell_id")


plot_gradient_nuclei_histogram(res, figures_path)


df_path = Path("data_tidy/Xenium_FFPE_Human_Breast_Cancer_Rep1_v9_segger.csv.gz")
df_v9 = dd.read_csv(df_path)
df_main = dd.read_parquet("data_raw/breast_cancer/Xenium_FFPE_Human_Breast_Cancer_Rep1/outs/transcripts.parquet")

ddf = df_v9.merge(df_main, on="transcript_id")
# df = dd.read_parquet(segger_emb_path/'segger_transcripts.parquet')
ddf1 = ddf.compute()
res = compute_cells_per_nucleus(df.compute(), new_cell_col="segger_cell_id")
plot_gradient_nuclei_histogram(res, figures_path, "segger")

dfff = df.compute()
dfff = dfff.dropna()
gene_dropout = calculate_gene_dropout_rate(dfff)


plot_z_distance_distribution(dfff, figures_path, title="Z-Distance Distribution")

plot_z_distance_boxplot(dfff, figures_path, title="Z-Distance Distribution")


tx_df = dd.read_csv("data_tidy/Xenium_FFPE_Human_Breast_Cancer_Rep2_v9_segger.csv.gz")
ddf = tx_df.merge(df_main, on="transcript_id")


from segger.prediction.boundary import generate_boundary
import geopandas as gpd
from tqdm import tqdm

bb = generate_boundaries(ddf, x="x_location", y="y_location", cell_id="segger_cell_id", n_jobs=8)


from pqdm.processes import pqdm  # or use pqdm.threads for threading-based parallelism


# Modify the function to work with a single group to use with pqdm
def process_group(group):
    cell_id, t = group
    return {"cell_id": cell_id, "length": len(t), "geom": generate_boundary(t, x="x_location", y="y_location")}


def generate_boundaries(df, x="x_location", y="y_location", cell_id="segger_cell_id", n_jobs=10):
    # Group by cell_id
    group_df = df.groupby(cell_id)
    # Use pqdm to process each group in parallel
    results = pqdm(tqdm(group_df, desc="Processing Groups"), process_group, n_jobs=n_jobs)
    # Convert results to GeoDataFrame
    return gpd.GeoDataFrame(
        data=[[res["cell_id"], res["length"]] for res in results],
        geometry=[res["geom"] for res in results],
        columns=["cell_id", "length"],
    )
