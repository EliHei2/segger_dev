# segger.models

The `segger.models` module provides the core machine learning models for the Segger framework, specifically designed for spatial transcriptomics data analysis using Graph Neural Networks (GNNs). This module implements attention-based convolutional architectures that can handle heterogeneous graphs with transcript and boundary nodes, formulating cell segmentation as a transcript-to-cell link prediction task.

> **📖 [Detailed Module Guide](README.md)** - For a comprehensive understanding of the models module, including architecture, examples, and best practices.

> **🔍 [Complete API Reference](api_reference.md)** - For detailed API documentation of all classes, functions, and modules.

## Overview

The models module serves as the machine learning engine for spatial transcriptomics analysis in Segger, offering:

- **Graph Neural Network Architecture**: Attention-based convolutional layers for spatial data analysis
- **Heterogeneous Graph Support**: Handles different node types (transcripts, boundaries) with specialized processing
- **Spatial Relationship Learning**: Learns complex spatial relationships between transcripts and cellular structures
- **Link Prediction Framework**: Formulates cell segmentation as a transcript-to-cell assignment problem
- **PyTorch Integration**: Seamless integration with PyTorch and PyTorch Geometric ecosystems
- **Scalable Training**: Support for both single and multi-GPU training workflows

## Core Architecture

The module is built around a sophisticated GNN architecture that processes spatial transcriptomics data:

- **`Segger`**: Main GNN model with attention-based convolutional layers using GATv2Conv
- **Heterogeneous Graph Processing**: Handles different node and edge types with specialized attention mechanisms
- **Attention Mechanisms**: Multi-head Graph Attention Networks for learning node relationships
- **Feature Engineering**: Automatic feature transformation and embedding for transcripts and boundaries

## Key Features

### **Advanced GNN Architecture**
- **Graph Attention Networks**: GATv2Conv layers for learning node relationships with flexible attention
- **Heterogeneous Processing**: Specialized handling for transcript and boundary nodes with different feature types
- **Multi-head Attention**: Parallel attention mechanisms for robust feature learning
- **Residual Connections**: Stabilized learning with configurable layer depth

### **Spatial Data Optimization**
- **Node Type Differentiation**: Automatic detection and processing of different node types
- **Spatial Relationship Learning**: Captures complex spatial interactions through graph structure
- **Feature Embedding**: Efficient transformation of spatial and transcript features
- **Memory Optimization**: Optimized for large spatial transcriptomics datasets

### **Training & Deployment**
- **PyTorch Integration**: Native PyTorch module compatibility with full CUDA support
- **PyTorch Geometric**: Optimized for graph-based operations and heterogeneous graphs
- **Multi-GPU Support**: Scalable training across multiple devices with PyTorch Lightning
- **Production Ready**: Optimized inference and deployment capabilities

## Submodules

- [Segger Model](segger_model.md): Core GNN architecture for spatial transcriptomics
- [Architecture Details](architecture.md): Detailed model architecture and design principles
- [Training Workflows](training.md): Training workflows and optimization strategies
- [Inference & Prediction](inference.md): Model inference and prediction utilities

## Use Cases

The models module is designed for:

- **Research Scientists**: Training GNNs on spatial transcriptomics data for cell segmentation
- **ML Engineers**: Building production-ready spatial analysis models with link prediction
- **Bioinformaticians**: Analyzing complex spatial gene expression patterns and cell relationships
- **Software Developers**: Integrating GNN models into spatial analysis pipelines

## API Documentation

::: src.segger.models
    options:
      show_root_heading: false
      show_root_full_path: false
      show_if_no_docstring: true


