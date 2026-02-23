# Plasma proteome-derived aging subtypes stratify divergent multimorbidity risk and healthspan

This repository contains the code used in the study:

> **Plasma proteome-derived aging subtypes stratify divergent multimorbidity risk and healthspan**

The project identifies aging subtypes from plasma proteomics using a **Mixture Gaussian Radial Basis Function (MixtureGRBF)** model and evaluates their clinical and biological associations.

The workflow consists of three major parts:

1. Data preprocessing
2. Subtype discovery using MixtureGRBF
3. Downstream analysis & visualization

---

## Repository Structure

```
.
├── preprocess/            # Data cleaning and preprocessing
├── mixtureGRBF/           # Core subtype discovery algorithm
├── run_algo.m             # Entry script for subtype inference
├── analysis/              # Statistical analysis and figures
├── utils/                 # Shared utility functions
├── output/                # Generated intermediate & final results
└── README.md
```

---

## Pipeline Overview

```
Raw proteomics data
        ↓
Preprocess (normalization, filtering)
        ↓
MixtureGRBF model (subtype inference)
        ↓
Subtype assignments & stages
        ↓
Association analysis (disease, survival, healthspan)
        ↓
Figures & tables
```

---

## 1. Environment Setup

### MATLAB (required for subtype discovery)

The subtype model is implemented in MATLAB.

Recommended:
- MATLAB R2021a or later
- Statistics and Machine Learning Toolbox

No external toolbox installation is required beyond standard MATLAB packages.

---

### Python (required for preprocessing & analysis)

Recommended Python ≥ 3.9

Install dependencies:

```bash
pip install numpy pandas scipy scikit-learn matplotlib seaborn lifelines tqdm statsmodels anndata
```

If survival analysis is used:

```bash
pip install lifelines
```

---

## 2. Data Preprocessing

All preprocessing scripts are located in:

```
preprocess/
```

This step prepares the plasma proteome matrix used by the model.

Typical preprocessing includes:

- Sample filtering (missingness / QC)
- Protein filtering
- Normalization
- Covariate adjustment
- Feature selection

After preprocessing, the script should output a matrix:

```
subjects × proteins
```

and save it to a file used by `run_algo.m`.

---

## 3. Subtype Discovery (Core Algorithm)

The main entry point:

```
run_algo.m
```

This script runs the MixtureGRBF model to learn latent aging subtypes.

The script will:

1. Load preprocessed proteomics data
2. Train the MixtureGRBF model
3. Estimate subtype membership
4. Estimate biological aging stage
5. Save outputs

Core algorithm implementation:

```
mixtureGRBF/
```

---

### Outputs

Results will be saved to:

```
output/
```

Typical files include:

| File | Description |
|----|----|
| subtype_stage | continuous biological aging stage |
| model_parameters | trained model parameters |


---

## 4. Downstream Analysis

All statistical analysis and figures are in:

```
analysis/
```

This section includes:

- Survival analysis
- Multimorbidity risk
- Disease association
- Cell-type enrichment
- Visualization

---

## 5. Reproducing the Study

Recommended execution order:

### Step 1 — Preprocess data
```
preprocess/*.py
```

### Step 2 — Train subtype model
```
run_algo.m
```

### Step 3 — Run analysis
```
analysis/*.py
```

---


## Citation

If you use this code, please cite the associated paper:

```
Plasma proteome-derived aging subtypes stratify divergent multimorbidity risk and healthspan
```

---

## Contact

For questions about the algorithm or reproduction, please open an issue in this repository.