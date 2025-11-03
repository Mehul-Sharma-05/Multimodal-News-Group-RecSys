
# Multimodal News Group Recommendation System

A robust, research-driven framework for **news recommendation in group settings** using advanced **vision-language models** and a dual-tower neural architecture. This system integrates group user profiles and multimodal content (text and images) to deliver highly relevant news articles tailored for *collaborative consumption*.

---

## Overview

The exponential rise of online media has transformed news discovery into a social, multimedia experience. Existing recommender systems typically focus on personalized or multimodal recommendations for individuals but largely ignore the **group dynamics** where users consume news together. Our framework bridges this gap by recommending news to groups using **unified vision-language semantic embeddings** and aggregated group histories.

---

## Key Features

- **Unified Vision-Language Encoding:** Articles represented using both text and images via a vision-language model (SigLIP2).
- **Group User Profiling:** Aggregates behavioral histories of all group members for collaborative recommendations.
- **Dual-Tower Architecture:** Parallel neural networks encode group profiles and news content, projecting both to a shared embedding space.
- **Real-Time and Batch Serving:** Optimized for both online inference and offline batch recommendation scenarios.
- **Extensive Evaluation:** Outperforms conventional baselines on accuracy, diversity, and group satisfaction across industry-standard datasets.

---

## Architecture

- **Vision-Language Model (SigLIP2):** Converts news articles into 1024-dimensional embeddings using paired text/images.
- **Group Embedding Construction:** Profiles built by clustering user interaction histories (MiniBatchKMeans), using mean pooling or attention pooling for aggregation.
- **Dual-Tower Model:** 
  - *Group Tower*: Processes group embeddings.
  - *Article Tower*: Processes news item embeddings.
  - Matching score via cosine similarity.
- **Training Protocols:** Adam optimizer, cross-entropy (ranking) loss, contrastive (InfoNCE) loss, early stopping.

---

## Datasets

- **MIND:** Large-scale Microsoft News dataset (1M users, 160K articles, click logs).
- **VMIND:** Augments MIND with associated article images, improving multimodal representation.
- **Group Construction:** User clustering algorithms to create collaborative consumption groups.

Preprocessing steps include text normalization, image resizing/normalization, and user/group filtering based on activity logs.

---

## Installation

**Prerequisites:**
- Python 3.10+
- PyTorch 2.1.0+
- Hugging Face Transformers 4.38+
- Pandas, NumPy, PIL
- CUDA-enabled GPU (recommended)

**Setup:**

```bash
git clone https://github.com/<your-repo>/multimodal-group-news-recommendation
cd multimodal-group-news-recommendation
pip install -r requirements.txt
```

---

## Usage

**Training:**

1. Prepare datasets (MIND, VMIND).
2. Run data preprocessing scripts.
3. Extract multimodal embeddings using SigLIP2.
4. Construct group profiles.
5. Train dual-tower model:

```bash
python train.py --config configs/training.yaml
```

**Inference:**

```bash
python recommend.py --groups_path data/groups.json --output results/recommendations.json
```

---

## Evaluation

- **Ranking Quality:** Tested against NRMS, MM-Rec, and other baselines using AUC, nDCG, Precision/Recall metrics.
- **Group Size Sensitivity:** Stable gains observed for group sizes (3–6).
- **Ablations:** Removing group profiling or multimodal signals shows clear performance drops, highlighting importance of each component.
- **Fairness/Diversity:** Measured using entropy, coverage, and diversity metrics for robust and inclusive recommendations.

---

## Results (Sample)

| Model        | Group | Multimodal | AUC   | nDCG5 | Diversity | Coverage | Fairness |
|--------------|-------|------------|-------|-------|-----------|----------|----------|
| NRMS         | No    | No         | 0.62  | 0.24  | 0.36      | 35%      | 0.64     |
| MM-Rec       | No    | Yes        | 0.65  | 0.27  | 0.51      | 48%      | 0.72     |
| **Ours**     | Yes   | Yes        | 0.76  | 0.37  | 0.67      | 56%      | 0.82     |

---

## Reproducibility

- All random seeds fixed for experimental consistency.
- Scripts track GPU usage, training metrics with Weights & Biases.
- Data splits, preprocessing, and training protocols documented for fair comparisons.

---

## References

- Wu et al., “MIND: A Large-scale Dataset for News Recommendation,” ACL 2020.
- Wu et al., “MM-Rec: Multimodal News Recommendation,” SIGIR 2022.
- Vaswani et al., “Attention Is All You Need,” NeurIPS 2017.
- More cited works in codebase and full documentation.

---


For more details or troubleshooting, refer to [docs/](docs/) or contact the maintainers. 

---

**NOTE:** For full reproducibility and additional experiments, including cold start handling and aggregation strategy ablations, see the complete research paper and supplementary materials in this repository.
