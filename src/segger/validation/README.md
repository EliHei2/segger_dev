## Benchmarking and Validation of Segmentation Methods

To rigorously evaluate the performance of segmentation methods in single-cell transcriptomics, we developed a suite of metrics grouped into four key categories: **General Statistics**, **Sensitivity**, **Spatial Localization**, and **Specificity and Contamination**. These metrics collectively assess the accuracy, biological relevance, and technical precision of segmentation outcomes, providing a comprehensive framework for method validation. In total, we utilize **9 distinct metrics** across these categories.

### General Statistics (2 Metrics)

**Percent Assigned Transcripts**. A fundamental measure of segmentation accuracy is the percentage of transcripts that are correctly assigned to cells. This metric assesses the ability of the segmentation method to accurately associate detected transcripts with their respective cells. The **Percent Assigned Transcripts** is calculated as:

$$
\text{Percent Assigned} = \frac{N_{\text{assigned}}}{N_{\text{total}}} \times 100
$$

where $N_{\text{assigned}}$ represents the number of transcripts assigned to a cell, and $N_{\text{total}}$ is the total number of transcripts detected in the dataset. A higher percentage reflects a more precise segmentation, which is critical for downstream analyses such as gene expression profiling and cell type classification.

**Transcript Density**. Transcript density is an important general statistic that provides insight into the transcriptional activity within a cell relative to its size. This metric is computed as:

$$
D = \frac{\text{transcript counts}}{\text{cell area}}
$$

A higher transcript density might indicate a more transcriptionally active or smaller cell, while a lower density could suggest a quiescent or larger cell. This measure is particularly useful in understanding cellular states and comparing them across different conditions or cell types.

### Sensitivity (2 Metrics)

**F1 Score for Cell Type Purity**. Sensitivity in the context of cell type identification is assessed using the F1 score, which balances precision and recall of marker gene expression. This score is crucial for determining how well a segmentation method can correctly identify cells of a specific type based on known marker genes. The **F1 Score for Cell Type Purity** is calculated as:

$$
\text{F1}_{\text{purity}} = 2 \times \frac{\text{precision} \times \text{recall}}{\text{precision} + \text{recall}}
$$

where precision is the proportion of true positive marker genes among those identified as positive, and recall is the proportion of true positive marker genes correctly identified by the method. A high F1 score indicates that the method is sensitive to the true expression patterns of marker genes, making it reliable for cell type annotation.

**Gene-specific Assignment Metrics**. Sensitivity is also evaluated through gene-specific assignment metrics, which assess the proportion of a gene’s transcripts that are correctly assigned to cells. This metric ensures that the segmentation method consistently captures the expression of individual genes:

$$
\text{Percent Assigned}_g = \frac{N_{\text{assigned}}^g}{N_{\text{total}}^g} \times 100
$$

where $N_{\text{assigned}}^g$ and $N_{\text{total}}^g$ represent the assigned and total transcripts for a given gene $g$. This measure is crucial for validating the method's ability to detect low-abundance transcripts and accurately reflect gene expression levels.

### Spatial Localization (2 Metrics)

**Percent Cytoplasmic and Percent Nucleus Transcripts**. The spatial distribution of transcripts within a cell’s compartments—cytoplasm and nucleus—is critical for understanding cellular function and identity. Accurate segmentation should preserve these spatial patterns. We calculated the **Percent Cytoplasmic** and **Percent Nucleus Transcripts** as:

$$
\text{Percent Cytoplasmic} = \frac{N_{\text{cytoplasmic}}}{N_{\text{assigned}}} \times 100
$$

$$
\text{Percent Nucleus} = \frac{N_{\text{nucleus}}}{N_{\text{assigned}}} \times 100
$$

where $N_{\text{cytoplasmic}}$ and $N_{\text{nucleus}}$ denote the number of transcripts located in the cytoplasm and nucleus, respectively. These metrics assess the ability of the segmentation method to maintain the correct intracellular distribution of transcripts, which is essential for accurate interpretation of subcellular processes and gene regulation.

**Neighborhood Entropy**. The structural organization of cells within their spatial context is evaluated using **Neighborhood Entropy**, which measures the diversity of cell types in the immediate vicinity of each cell:

$$
E = -\sum_{c} p(c) \log(p(c))
$$

where $p(c)$ is the proportion of neighboring cells of type $c$. Higher entropy values suggest a more heterogeneous cellular neighborhood, which can be indicative of complex tissue architecture or microenvironmental interactions.

### Specificity and Contamination (3 Metrics)

**Mutually Exclusive Co-expression Rate (MECR)**. Specificity in the context of gene expression is assessed by the MECR, which quantifies the degree to which two genes are expressed in a mutually exclusive manner across different cells. The **MECR** is calculated as:

$$
\text{MECR}(g_1, g_2) = \frac{P(g_1 \cap g_2)}{P(g_1 \cup g_2)}
$$

where $P(g_1 \cap g_2)$ is the proportion of cells where both genes $g_1$ and $g_2$ are expressed, and $P(g_1 \cup g_2)$ is the proportion of cells where at least one of the genes is expressed. A lower MECR indicates higher specificity in gene expression, suggesting that the genes are likely to be markers for distinct cell populations.

**Contamination from Neighboring Cells**. In spatial transcriptomics, contamination from neighboring cells can occur due to signal overlap or physical proximity. We assessed this potential contamination by calculating a contamination score based on the expression of marker genes in adjacent cells:

$$
C_{ij} = \frac{\sum_{k \in \text{neighbors}} m_{ik} \cdot w_{kj}}{\sum_{k \in \text{neighbors}} m_{ik}}
$$

where $m_{ik}$ is the expression level of marker $k$ in cell $i$, and $w_{kj}$ represents the weight of the neighbor $k$ influencing cell $j$. This metric is critical for identifying and mitigating artifacts that arise from the close physical proximity of cells, ensuring that the segmentation method produces biologically meaningful results.

### Comparison Across Segmentation Methods

To compare different segmentation methods comprehensively, we employed correlation analyses and scatter plots focusing on key metrics such as transcript count, cell area, and transcript density. The **Comparison Metric** between two methods $m_1$ and $m_2$ is defined as:

$$
\text{Comparison Metric}(m_1, m_2) = \frac{\sum_{i=1}^{n} (M_1(i) - \bar{M_1}) (M_2(i) - \bar{M_2})}{\sqrt{\sum_{i=1}^{n} (M_1(i) - \bar{M_1})^2 \sum_{i=1}^{n} (M_2(i) - \bar{M_2})^2}}
$$

where $M_1$ and $M_2$ represent the metrics of interest for the methods being compared, and $\bar{M_1}$ and $\bar{M_2}$ are their respective means. This approach allows for a detailed examination of segmentation performance, highlighting method-specific strengths and potential biases.
