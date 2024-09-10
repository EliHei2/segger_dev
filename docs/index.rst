Welcome to Segger
=================

🚀 **Segger** is a cutting-edge tool for **cell segmentation** in **single-molecule spatial omics** datasets. By leveraging **graph neural networks (GNNs)** and heterogeneous graphs, Segger offers unmatched accuracy and scalability.

.. note::

   For installation instructions, visit the :doc:`Installation Guide <installation>`.

---


Quick Links
===========

- **Installation Guide**: Get started with installing Segger on your machine. See :doc:`installation`.
- **User Guide**: Learn how to use Segger for cell segmentation tasks. See :doc:`user_guide/index`.
- **Command-Line Interface (CLI)**: Explore the CLI options for working with Segger. See :doc:`cli`.
- **API reference**: Dive into the detailed API documentation for advanced usage. See :doc:`api/index`.

---

Why Segger?
===========

🔬 **Challenges in Segmentation**

Spatial omics segmentation faces issues like:

- **Over/Under-segmentation**
- **Transcript contamination**
- **Scalability limitations**

Segger tackles these with a **graph-based approach**, achieving superior segmentation accuracy.

---

How Segger Works
================

🧠 **Heterogeneous Graph Structure**

Segger models **nuclei and transcript relationships** using:

- **Nodes** for nuclei and transcripts
- **Edges** connecting spatially related elements
- **GNN** to propagate information and refine cell borders

🔄 **Nucleus-free transcripts**: Segger can effectively handle **nucleus-free cells**, improving transcript assignment in complex tissue environments.

---

Key Features
============

✨ **What makes Segger unique?**

- **Highly parallelizable** – Optimized for multi-GPU environments
- **Fast and efficient** – Trains in a fraction of the time compared to alternatives
- **Transfer learning** – Easily adaptable to new datasets and technologies

---

Technology Behind Segger
========================

⚙️ **Powered by:**

- **PyTorch Lightning & PyTorch Geometric**: Enables fast, efficient graph neural network (GNN) implementation for heterogeneous graphs.
- **RAPIDS**: Provides GPU-accelerated computation for tasks like k-nearest neighbors (KNN) graph construction.
- **AnnData & Scanpy**: Handles transcriptomic data integration and efficient data processing for large-scale datasets.
- **SciPy**: Facilitates spatial graph construction, including distance metrics and convex hull calculations for transcript clustering.

---

Get Started
===========

💡 **Ready to use Segger?**

Follow these guides to get started:

- :doc:`Installation Guide <installation>`
- :doc:`User Guide <user_guide/index>`

.. tip::

   Segger's ability to train across **multiple GPUs** makes it ideal for **large-scale projects**.

---

Use Cases
=========

🔍 **Benchmarked on:**

- **10X Xenium** and **MERSCOPE** – Segger shows **exceptional performance** in accuracy and speed.
  Its ability to **fine-tune across datasets** makes it adaptable for various projects.

🔗 Explore more in the :doc:`User Guide <user_guide/index>`.

---

Join the Community
==================

- **Source Code**: `GitHub <https://github.com/EliHei2/segger_dev>`_
- **Bug Tracker**: `Report Issues <https://github.com/EliHei2/segger_dev/issues>`_
- **Full Documentation**: :doc:`API Reference <api/index>`

Segger is **open-source** and welcomes contributions. Join us in advancing spatial omics segmentation!
