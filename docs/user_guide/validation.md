# Validating the segmentations

This module provides utilities for validating segmentation methods in single-cell transcriptomics, focusing on evaluating performance across metrics such as sensitivity, specificity, and spatial localization.

---

## Benchmarking and Validation of Segmentation Methods

To rigorously evaluate segmentation performance, we use a suite of metrics grouped into four categories: **General Statistics**, **Sensitivity**, **Spatial Localization**, and **Specificity and Contamination**. These metrics provide a comprehensive framework for assessing the accuracy and precision of segmentation methods.

### General Statistics

- **Percent Assigned Transcripts**: Measures the proportion of transcripts correctly assigned to cells.
  
$$
\text{Percent Assigned} = \frac{N_{\text{assigned}}}{N_{\text{total}}} \times 100
$$

- **Transcript Density**: Assesses transcript counts relative to cell size.
  
$$
D = \frac{\text{transcript counts}}{\text{cell area}}
$$

### Sensitivity

- **F1 Score for Cell Type Purity**: Evaluates how well a segmentation method can identify cells based on marker genes.
  
$$
\text{F1}_{\text{purity}} = 2 \times \frac{\text{precision} \times \text{recall}}{\text{precision} + \text{recall}}
$$

- **Gene-specific Assignment Metrics**: Measures the proportion of correctly assigned transcripts for a specific gene.
  
$$
\text{Percent Assigned}_g = \frac{N_{\text{assigned}}^g}{N_{\text{total}}^g} \times 100
$$

### Spatial Localization

- **Percent Cytoplasmic and Percent Nucleus Transcripts**: Evaluates the spatial distribution of transcripts within cells.
  
$$
\text{Percent Cytoplasmic} = \frac{N_{\text{cytoplasmic}}}{N_{\text{assigned}}} \times 100
$$

$$
\text{Percent Nucleus} = \frac{N_{\text{nucleus}}}{N_{\text{assigned}}} \times 100
$$

- **Neighborhood Entropy**: Measures the diversity of neighboring cell types.
  
$$
E = -\sum_{c} p(c) \log(p(c))
$$

### Specificity and Contamination

- **Mutually Exclusive Co-expression Rate (MECR)**: Quantifies how mutually exclusive gene expression is across cells.
  
$$
\text{MECR}(g_1, g_2) = \frac{P(g_1 \cap g_2)}{P(g_1 \cup g_2)}
$$

- **Contamination from Neighboring Cells**: Assesses transcript contamination from adjacent cells.
  
$$
C_{ij} = \frac{\sum_{k \in \text{neighbors}} m_{ik} \cdot w_{kj}}{\sum_{k \in \text{neighbors}} m_{ik}}
$$

### Comparison Across Segmentation Methods

A correlation analysis is used to compare different segmentation methods based on metrics such as transcript count and cell area. The **Comparison Metric** is defined as:

$$
\text{Comparison Metric}(m_1, m_2) = \frac{\sum_{i=1}^{n} (M_1(i) - \bar{M_1}) (M_2(i) - \bar{M_2})}{\sqrt{\sum_{i=1}^{n} (M_1(i) - \bar{M_1})^2 \sum_{i=1}^{n} (M_2(i) - \bar{M_2})^2}}
$$
