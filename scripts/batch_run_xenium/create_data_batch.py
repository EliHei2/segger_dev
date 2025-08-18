#!/usr/bin/env python
from pathlib import Path
from segger.data.parquet.sample import STSampleParquet
from segger.data.utils import calculate_gene_celltype_abundance_embedding
import scanpy as sc
import pandas as pd
import numpy as np
import argparse
import os
from pqdm.processes import pqdm
from tqdm import tqdm
from segger.data.parquet._utils import find_markers, find_mutually_exclusive_genes

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Preprocess Xenium sample for segger')
    parser.add_argument('--sample_id', type=str, required=True, help='Xenium sample ID to process')
    parser.add_argument('--project_dir', type=str, required=True, help='Base directory containing Xenium samples')
    parser.add_argument('--scrna_file', type=str, required=True, help='Path to scRNA-seq reference file')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for processed data')
    parser.add_argument('--celltype_column', type=str, default="Annotation_merged", help='Column name for cell types in scRNA-seq data')
    parser.add_argument('--n_workers', type=int, default=4, help='Number of workers for processing')
    parser.add_argument('--k_tx', type=int, default=5, help='Number of neighbors for transcript graph')
    parser.add_argument('--dist_tx', type=float, default=20.0, help='Distance threshold for transcript graph')
    parser.add_argument('--subsample_frac', type=float, default=0.1, help='Subsampling fraction for scRNA-seq data')
    
    args = parser.parse_args()

    # Convert paths to Path objects
    project_dir = Path(args.project_dir)
    scrnaseq_file = Path(args.scrna_file)
    output_dir = Path(args.output_dir)

    # Load reference data and compute embeddings
    scrnaseq = sc.read(scrnaseq_file)
    sc.pp.subsample(scrnaseq, args.subsample_frac)
    gene_celltype_abundance_embedding = calculate_gene_celltype_abundance_embedding(
        scrnaseq,
        args.celltype_column
    )

    # Process the sample
    xenium_data_dir = project_dir / args.sample_id
    segger_data_dir = output_dir / args.sample_id
    
    try:
        sample = STSampleParquet(
            base_dir=xenium_data_dir,
            n_workers=args.n_workers,
            sample_type="xenium", # xenium for typical xenium, xenium_v2 for v2
            weights=gene_celltype_abundance_embedding,
            # scale_factor=0.5 # this is to shrink the initial seg. masks (used for seg. kit)
        )
        
        genes = list(set(scrnaseq.var_names) & set(sample.transcripts_metadata['feature_names']))
        markers = find_markers(scrnaseq[:,genes], cell_type_column=args.celltype_column, pos_percentile=90, neg_percentile=20, percentage=20)
        # Find mutually exclusive genes based on scRNAseq data
        exclusive_gene_pairs = find_mutually_exclusive_genes(
            adata=scrnaseq,
            markers=markers,
            cell_type_column=args.celltype_column
        )

        sample.save(
            data_dir=segger_data_dir,
            k_bd=3,
            dist_bd=15,
            k_tx=args.k_tx,
            dist_tx=args.dist_tx,
            k_tx_ex=20,
            dist_tx_ex=20,
            tile_size=10_000,  # Tile size for processing
            neg_sampling_ratio=5.0,
            frac=1.0,
            val_prob=0.3,
            test_prob=0.0,
        )
        print(f"Successfully processed {args.sample_id}")
    except Exception as e:
        print(f"Failed to process {args.sample_id}: {str(e)}")
        raise

if __name__ == "__main__":
    main()