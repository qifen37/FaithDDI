# FaithDDI
## Overview
This is the official implementation for the manuscript:

> **Faithful Molecular Representation Learning and Knowledge-Grounded Mechanism Reasoning for Drug-Drug Interaction Prediction**

FaithDDI is a multi-class drug-drug interaction (DDI) prediction framework that combines molecular motif representations with biomedical knowledge-graph context.

## Repository contents

```
|-- main.py                 # Cross-validation experiment configuration and entry point
|-- train_test.py           # Training, calibration, and evaluation routines
|-- model/
|   |-- FaithDDI.py         # FaithDDI model and cross-drug co-attention
|   |-- hgnn.py             # Heterogeneous knowledge-graph encoder
|   |-- decoder.py          # KG-anchored path-memory decoder
|-- utils/
|   |-- metrics.py          # Multi-class evaluation metrics
|   |-- pytorchtools.py     # Macro-F1 early stopping
|-- data/
    |-- Deng/
    |   |-- ddi.tsv
    |   |-- smiles.tsv
    |-- Ryu/
        |-- ddi.tsv
        |-- smiles.tsv
```

## Model overview

The implementation is organized into five main stages:

1. **Knowledge-graph encoding.** `HGNN` performs relation-aware message passing over a heterogeneous biomedical graph.
2. **Molecular encoding.** A motif hypergraph encoder produces atom-level and motif-level molecular representations.
3. **Multimodal fusion.** Knowledge-graph and molecular embeddings are combined through a learnable gate.
4. **Pairwise interaction modeling.** Bidirectional multi-head co-attention lets each drug attend to the other drug's motifs.
5. **Mechanism-aware decoding.** A bank of path prototypes, initialized from knowledge-graph relation embeddings, supplies a knowledge-grounded context for DDI classification.

## Data

The repository includes two standard multi-class DDI benchmarks:

| Dataset | Drugs | DDI pairs | Classes | Drug ID range | Label range |
|:--------|------:|----------:|--------:|:--------------|:------------|
| Deng    | 570   | 37,264    | 65      | 0-569         | 0-64        |
| Ryu     | 1,700 | 191,570   | 86      | 0-1699        | 0-85        |

Each `ddi.tsv` file is tab-separated and contains no header:

```text
left_drug_id    right_drug_id    ddi_class_id
```

Each `smiles.tsv` file is also tab-separated and contains no header:

```text
drug_id    SMILES
```

## Data sources
| Resource |  Source                                                       |
| -------- |  ------------------------------------------------------------ |
| Deng     | [Bioinformatics 2020](https://academic.oup.com/bioinformatics/article/36/15/4316/5837109) |
| Ryu      | [PNAS 2018](https://www.pnas.org/doi/10.1073/pnas.1803294115) |
| DRKG     | [Official repository](https://github.com/gnn4dr/DRKG)        |

## Package dependencies

The project was developed with the following environment:

```
Python          3.10
PyTorch         2.2.1+cu121
torch-scatter   2.1.2+pt22cu121
DGL             2.2.1
NumPy           1.26.4
NetworkX        2.6
RDKit           2024.9.6
scikit-learn    1.6.1
```

Reported experiments were run on an Intel Xeon Gold 6330 CPU and an NVIDIA RTX 5880 Ada GPU.

## Experiment configuration

`main.py` defines five-fold evaluation on the Deng or Ryu dataset under three split conditions. The main arguments are:

| Argument | Default | Description |
|:---------|--------:|:------------|
| `--ddi_name` | `Deng` | DDI dataset: `Deng` or `Ryu` |
| `--condition` | `S1` | Evaluation condition: `S1`, `S2`, or `S3` |
| `--fold_num` | `5` | Number of cross-validation folds |
| `--hidden_dim` | `256` | Hidden representation size |
| `--num_layer` | `3` | Number of graph/hypergraph encoder layers |
| `--batch_size` | `4096` | Batch size |
| `--epoch` | `1000` | Maximum number of training epochs |
| `--patience` | `50` | Early-stopping patience after the warm-up period |

Condition-specific defaults are applied to the learning rate, weight decay, label smoothing, dropout rates, and class-prior calibration strength. Training uses Adam, gradient clipping, a fixed random seed of 42, and macro-F1 for early stopping. Reported metrics are accuracy, macro-F1, macro-precision, macro-recall.

## Citation and license

Citation metadata and a software license have not yet been included in this snapshot. Add both before public release.