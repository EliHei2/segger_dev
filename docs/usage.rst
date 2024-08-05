Usage
=====

Using the package to create a dataset
-------------------------------------

You can use the `XeniumSample` class to load your data and create a dataset for Segger.

Using Jupyter Notebook
======================

.. nbinclude:: notebooks/create_dataset.ipynb

Using Bash Script
=================

You can also create the dataset using the provided bash script.

.. code-block:: bash

    ./scripts/create_dataset.sh

Using Python Script
===================

Alternatively, you can run the Python script directly with command-line arguments.

.. code-block:: bash

    python scripts/create_dataset.py \
        --raw_data_dir data_raw/pancreatic \
        --processed_data_dir data_tidy/pyg_datasets/pancreatic \
        --transcripts_url https://cf.10xgenomics.com/samples/xenium/1.3.0/xenium_human_pancreas/analysis/transcripts.csv.gz \
        --nuclei_url https://cf.10xgenomics.com/samples/xenium/1.3.0/xenium_human_pancreas/analysis/nucleus_boundaries.csv.gz \
        --min_qv 30 \
        --d_x 180 \
        --d_y 180 \
        --x_size 200 \
        --y_size 200 \
        --r 3 \
        --val_prob 0.1 \
        --test_prob 0.1 \
        --k_nc 3 \
        --dist_nc 10 \
        --k_tx 5 \
        --dist_tx 3 \
        --compute_labels True \
        --sampling_rate 1
