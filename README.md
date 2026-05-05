# Replication Kit: IJF-D-25-00886R1

**Project Title:** Designing Cost-effective Climate Policies through Preference Prediction: Evidence from Chinese Households' Low-carbon Alternatives

**Date:** May 2026  
**Author:** Yuanyuan Tang (School of Applied Economics, Renmin University of China)  
**Contact:** tangyuanyuan@ruc.edu.cn

---

## 1. Overview

This replication kit contains all the data and code necessary to reproduce the empirical results, including tables and figures presented in the manuscript. The analysis combines R (for econometric modeling and post-stratification weighting) and Python (for machine learning and predictive analysis).

---

## 2. Repository Structure

Replication_Kit_IJF/
├── run_all.py              # Master script to execute the entire pipeline
├── README.md               # This documentation
├── Data/
│   ├── energy_wta.csv      # Used survey data
│   └── Intermediate/       # Folder for generated intermediate datasets
├── Code/
│   ├── section4_1/         # Section 4.1 and its related checks
│   ├── Section4_2/         # Section 4.2 and its related checks
│   └── Section4_3/         # Section 4.3 and its related checks analysis
└── Results/
├── tables/             # Output folder for CSV/LaTeX tables
└── pictures/           # Output folder for generated figures

---

## 3. Computational Environment

### Python
- **Language:** Python 3.10.0
- **Key Libraries** (see `requirements.txt`):
  - pandas==2.0.0
  - scipy==1.10.1
  - numpy==1.23.5
  - sklearn==1.6.1
  - xgboost==3.1.2
  - optuna==4.7.0
  - matplotlib==3.8.2
  - tqdm==4.65.0

### R
- **Language:** R 4.5.2
- **Necessary Packages:** MASS, dplyr, stargazer, brant, survey, VGAM, ggplot2, tidyr, broom, knitr

---

## 4. Instructions for Reproduction

**Step 1: Install Dependencies**

```bash
pip install -r requirements.txt
```

Ensure R is installed and the required packages listed above are available in your R environment.

**Step 2: Run the Master Script**

```bash
python run_all.py
```

> **Important:**
> - The script will first attempt to detect your Rscript executable automatically.
> - If detection fails, you will be prompted to manually enter the full path (e.g., `C:\Program Files\R\R-4.x.x\bin\Rscript.exe`).

**Step 3: Verify Results**

Once completed, all outputs will be populated in the `Results/` folder:
- **Tables:** `Results/tables/`
- **Figures:** `Results/pictures/`

---

## 5. Data Notes

- **Source Data:** `Data/energy_wta.csv` contains the survey responses from 1,487 Chinese households, translated into English text or numerical version.
- **Intermediate Data:** Temporary data generated during the processing stage.

---

## 6. Runtime

Estimated runtime: **over 5 hours**.
