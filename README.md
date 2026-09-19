# CMEA: Candidate-Level Cross-Modal Concordance for Reliable Multimodal Entity Alignment

> Anonymous review repository for the manuscript **"Candidate-Level Cross-Modal Concordance for Reliable Multimodal Entity Alignment"**.

This repository provides a lightweight reference package for anonymous review. It includes the environment setup, dataset organization, basic preprocessing/training/evaluation interfaces, and selected experimental results. To preserve anonymity and avoid releasing incomplete research artifacts during review, some implementation details, full experiment scripts, and additional resources are withheld at this stage. A cleaned and complete version will be released after acceptance/publication.

---

## 1. Overview

CMEA is a multimodal entity alignment framework designed to improve the reliability of multimodal fusion by jointly considering modality certainty and cross-modal agreement in the candidate space.

The method operates on structural, textual, and visual information and produces adaptive pair-specific modality weights for entity alignment.

### Motivation

<p align="center">
  <img src="./source_rdm/intro.jpg" width="100%" alt="Motivation Figure">
</p>


### Framework

<p align="center">
  <img src="./source_rdm/model.png" width="100%" alt="CMEA Framework">
</p>



For additional methodological details, please refer to the submitted manuscript. The full implementation and extended documentation will be released after acceptance/publication.





### Comparison with Reliability-Aware and Adaptive Fusion Methods

To further clarify the technical positioning of CMEA, we summarize representative reliability-aware and adaptive multimodal entity alignment methods from several complementary perspectives, including adaptation/reliability mechanism, modeling granularity, shared candidate-space modeling, explicit cross-modal agreement, and pair-specific weighting. The comparison highlights that CMEA differs from prior approaches by explicitly modeling modality-specific candidate preference distributions in a shared cross-graph candidate space and using candidate-level cross-modal concordance to derive directional reliability and pair-specific modality weights.

| Method | Main Adaptation / Reliability Mechanism | Granularity | Shared Candidate-Space Modeling | Explicit Cross-Modal Agreement | Pair-Specific Weighting |
|---|---|---|---|---|---|
| AMF2SEA | Adaptive selection of fusion strategies | Entity level | Not explicitly modeled | Not explicitly modeled at candidate-distribution level | Not explicitly pair-specific |
| RICEA | Relative interaction + uncertainty-based weight recalibration | Entity level | Not explicitly modeled | Relative modality interaction, rather than candidate-distribution concordance | Not explicitly pair-specific |
| CDMEA | Counterfactual mitigation of visual modality bias | Modality-bias / causal-effect level | Not explicitly modeled | Not explicitly modeled at candidate-distribution level | Not explicitly pair-specific |
| HUMEA | Hierarchical MoE + unimodal distillation | Entity / representation level | Not explicitly modeled | Intra-/inter-modal interaction, rather than candidate-space concordance | Not explicitly pair-specific |
| **CMEA** | **Candidate certainty + leave-one-modality-out concordance** | **Directional entity level → candidate-pair level** | **Yes, shared candidate distributions** | **Yes, candidate-level concordance** | **Yes, bidirectionally informed pair-specific weights** |












## 2. Environment

The experiments were implemented in PyTorch and conducted on NVIDIA GPUs.

We recommend creating a Conda environment:

```bash
conda create -n cmea python=3.11 -y
conda activate cmea
```

Install PyTorch according to your CUDA version, and then install the remaining dependencies:

```text
annotated-doc
anyio
bzip2
ca-certificates
certifi
click
colorama
conda-pack
filelock
fsspec
h11
hf-xet
httpcore
httpx
huggingface-hub
idna
jinja2
libexpat
libffi
liblzma
libsqlite
libzlib
markdown-it-py
markupsafe
mdurl
mpmath
networkx
numpy
openssl
packaging
pillow
pip
pygments
pypdf
python
pyyaml
regex
rich
safetensors
setuptools
shellingham
sympy
tk
tokenizers
torch
torchaudio
torchvision
tqdm
transformers
typer
typing-extensions
tzdata
ucrt
vc
vc14_runtime
vcomp14
wheel
```

## 3. Data Preparation and Usage

The experiments use the following multimodal entity alignment benchmarks:

- FB15K--DB15K
- FB15K--YG15K
- DBP15K ZH--EN
- DBP15K JA--EN
- DBP15K FR--EN

Please organize the datasets under:

```text
data/
├── FB15K-DB15K/
├── FB15K-YG15K/
├── DBP15K_ZH-EN/
├── DBP15K_JA-EN/
└── DBP15K_FR-EN/
```


### Table 1: Results on FB15K-DB15K

| Method | Ref. | 20% Hits@1 | 20% Hits@10 | 20% MRR | 50% Hits@1 | 50% Hits@10 | 50% MRR | 80% Hits@1 | 80% Hits@10 | 80% MRR |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PoE | ESWA'19 | 0.126 | 0.251 | 0.170 | 0.464 | 0.658 | 0.533 | 0.666 | 0.820 | 0.721 |
| MMEA | KSEM'20 | 0.265 | 0.541 | 0.357 | 0.417 | 0.703 | 0.512 | 0.590 | 0.869 | 0.685 |
| EVA | AAAI'21 | 0.199 | 0.448 | 0.283 | 0.334 | 0.589 | 0.422 | 0.484 | 0.696 | 0.563 |
| MCLEA | COLING'22 | 0.295 | 0.582 | 0.393 | 0.555 | 0.784 | 0.637 | 0.735 | 0.890 | 0.790 |
| MEAformer | MM'23 | 0.417 | 0.715 | 0.518 | 0.619 | 0.843 | 0.698 | 0.765 | 0.916 | 0.820 |
| ACK-MMEA | WWW'23 | 0.304 | 0.549 | 0.387 | 0.560 | 0.736 | 0.624 | 0.682 | 0.874 | 0.752 |
| DESAlign | ICDE'24 | 0.497 | 0.750 | 0.586 | 0.718 | <u>0.889</u> | 0.782 | 0.805 | 0.926 | 0.850 |
| PCMEA | AAAI'24 | 0.556 | 0.807 | 0.672 | 0.706 | 0.884 | 0.785 | 0.794 | 0.924 | 0.848 |
| RICEA | ACL'25 | 0.471 | 0.720 | 0.589 | 0.648 | 0.852 | 0.721 | 0.776 | 0.916 | 0.829 |
| CDMEA | SIGIR'25 | <u>0.621</u> | 0.813 | <u>0.707</u> | 0.712 | 0.885 | 0.793 | <u>0.835</u> | 0.930 | <u>0.864</u> |
| HSP | PR'26 | 0.542 | **0.837** | 0.666 | 0.716 | 0.886 | <u>0.797</u> | 0.815 | 0.932 | 0.858 |
| MyGRAM | AAAI'26 | 0.619 | 0.807 | 0.702 | <u>0.724</u> | 0.877 | 0.784 | 0.822 | 0.931 | 0.859 |
| LLMEA | AAAI'26 | 0.613 | 0.827 | 0.692 | 0.641 | 0.873 | 0.725 | 0.792 | 0.925 | 0.851 |
| HUMEA | AAAI'26 | 0.511 | 0.764 | 0.598 | 0.694 | 0.880 | 0.763 | 0.809 | <u>0.935</u> | 0.856 |
| **CMEA (Ours)** | - | **0.633** | <u>0.835</u> | **0.718** | **0.738** | **0.893** | **0.806** | **0.840** | **0.941** | **0.872** |
| Std. | - | ± .007 | ± .004 | ± .005 | ± .005 | ± .003 | ± .004 | ± .004 | ± .003 | ± .003 |
| Best Imp. (%) | - | 1.93 | -- | 1.56 | 1.93 | 0.45 | 1.13 | 0.60 | 0.64 | 0.93 |

**Table 1.** Experimental results on the FB15K-DB15K dataset. The best and second-best results are highlighted in **bold** and <u>underlined</u>, respectively. Hits@1 and Hits@10 denote the proportion of ground-truth entities ranked within the top-1 and top-10 positions, respectively. The "Best Imp. (%)" indicates the percentage improvement of our CMEA model over the strongest baseline.

---

### Table 2: Results on FB15K-YG15K

| Method | Ref. | 20% Hits@1 | 20% Hits@10 | 20% MRR | 50% Hits@1 | 50% Hits@10 | 50% MRR | 80% Hits@1 | 80% Hits@10 | 80% MRR |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PoE | ESWA'19 | 0.250 | 0.495 | 0.334 | 0.411 | 0.669 | 0.498 | 0.492 | 0.705 | 0.572 |
| MMEA | KSEM'20 | 0.234 | 0.480 | 0.317 | 0.403 | 0.645 | 0.486 | 0.598 | 0.839 | 0.682 |
| EVA | AAAI'21 | 0.153 | 0.361 | 0.224 | 0.311 | 0.534 | 0.388 | 0.491 | 0.692 | 0.565 |
| MCLEA | COLING'22 | 0.254 | 0.484 | 0.332 | 0.501 | 0.705 | 0.574 | 0.667 | 0.824 | 0.722 |
| MEAformer | MM'23 | 0.327 | 0.595 | 0.417 | 0.560 | 0.778 | 0.639 | 0.703 | 0.873 | 0.766 |
| ACK-MMEA | WWW'23 | 0.289 | 0.496 | 0.360 | 0.535 | 0.699 | 0.593 | 0.676 | 0.864 | 0.744 |
| DESAlign | ICDE'24 | 0.410 | 0.660 | 0.495 | 0.612 | 0.799 | 0.680 | 0.728 | 0.877 | 0.782 |
| PCMEA | AAAI'24 | 0.509 | 0.684 | <u>0.601</u> | 0.656 | 0.805 | 0.691 | 0.745 | 0.902 | 0.793 |
| RICEA | ACL'25 | 0.411 | 0.658 | 0.497 | 0.617 | 0.811 | 0.687 | 0.734 | 0.892 | 0.792 |
| CDMEA | SIGIR'25 | 0.521 | <u>0.708</u> | 0.584 | 0.668 | 0.857 | 0.705 | 0.772 | 0.907 | 0.813 |
| HSP | PR'26 | 0.527 | 0.684 | 0.595 | 0.674 | 0.844 | 0.706 | 0.766 | **0.926** | 0.822 |
| MyGRAM | AAAI'26 | <u>0.529</u> | 0.683 | 0.589 | <u>0.715</u> | <u>0.864</u> | <u>0.765</u> | 0.766 | 0.917 | 0.826 |
| LLMEA | AAAI'26 | 0.347 | 0.604 | 0.426 | 0.581 | 0.787 | 0.660 | 0.712 | 0.896 | 0.775 |
| HUMEA | AAAI'26 | 0.439 | 0.698 | 0.528 | 0.649 | 0.847 | 0.720 | <u>0.773</u> | 0.919 | <u>0.827</u> |
| **CMEA (Ours)** | - | **0.548** | **0.728** | **0.610** | **0.731** | **0.869** | **0.778** | **0.782** | <u>0.925</u> | **0.836** |
| Std. | - | ± .008 | ± .005 | ± .006 | ± .006 | ± .004 | ± .005 | ± .004 | ± .003 | ± .004 |
| Best Imp. (%) | - | 3.59 | 2.82 | 1.50 | 2.24 | 0.46 | 1.70 | 1.16 | -- | 1.09 |

**Table 2.** Experimental results on the FB15K-YG15K dataset. The best and second-best results are highlighted in **bold** and <u>underlined</u>, respectively. Hits@1 and Hits@10 denote the proportion of ground-truth entities ranked within the top-1 and top-10 positions, respectively. The "Best Imp. (%)" indicates the percentage improvement of our CMEA model over the strongest baseline.


## Default Experimental Configuration

| Parameter | Setting | Parameter | Setting |
|---|---:|---|---:|
| Learning rate | `1e-4` | Weight decay | `1e-5` |
| Batch size | `64` | Dropout | `0.1` |
| Optimizer | `AdamW` | LR scheduler | `Cosine annealing` |
| Max epochs | `1000` | Early stopping | `10 epochs` |
| Candidate size \(K\) | `100` | Candidate temperature \(\tau_p\) | `0.10` |
| Text hidden dimension | `768` |
| Visual encoder | `VGG-16` | Encoder status | `Frozen` |
| Structural encoder | `GCN` | Graph encoder sharing | `Shared across KGs` |
| Projection dimension \(d\) | `256` | Projection normalization | `L2 normalization` |
| Text pooling | `[CLS]` | Similarity function | `Cosine similarity` |
| Candidate support | `Multi-modal Top-K union` | Candidate refresh | `Periodic` |
| Certainty measure | `Normalized entropy` | Concordance measure | `JS divergence` |
| Consensus strategy | `Leave-one-modality-out` | Pair aggregation | `Geometric mean` |
| Missing visual feature | `One-hop mean aggregation` | Completion gradient | `Detached` |
| Alignment objective | `Bidirectional` | Auxiliary supervision | `Enabled` |
| Number of runs | `5` | Evaluation metrics | `Hits@1 / Hits@10 / MRR` |


## 5. Reproducibility Notes

The main experiments are evaluated over multiple random seeds. For reproducibility, we recommend recording:

- Python version
- PyTorch version
- CUDA version
- GPU model
- dataset split
- configuration file
- random seed
- checkpoint

The pretrained textual and visual encoders used in the main experiments are kept frozen during CMEA training.





## 6. Code Release Plan

To maintain a clean and stable release during the review stage, the current anonymous repository contains only the review-oriented reference implementation and selected experiment resources.

Upon acceptance/publication, we plan to release:

- the complete cleaned source code;
- full preprocessing scripts;
- complete training and evaluation scripts;
- experiment configuration files;
- additional checkpoints;
- extended reproduction instructions;
- supplementary implementation details.

The post-acceptance release will be linked from this repository.

---



## 7. Citation

For anonymous review:

```bibtex
@article{cmea2026,
  title   = {Candidate-Level Cross-Modal Concordance for Reliable Multimodal Entity Alignment},
  author  = {Anonymous Authors},
  journal = {Under Review},
  year    = {2026}
}
```












