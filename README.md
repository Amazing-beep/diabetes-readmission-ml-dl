# Comparing Traditional Machine Learning and Deep Learning for Predicting 30-Day Hospital Readmission Among Patients with Diabetes

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Amazing-beep/diabetes-readmission-ml-dl/blob/main/notebooks/diabetes_readmission_classification.ipynb)
[![W&B Offline](https://img.shields.io/badge/W%26B-Offline%20Tracking-orange.svg)](https://wandb.ai/)

---

## Important Submission Links
- **GitHub Repository**: [https://github.com/Amazing-beep/diabetes-readmission-ml-dl](https://github.com/Amazing-beep/diabetes-readmission-ml-dl)
- **Google Drive Artifacts Folder**: [https://drive.google.com/drive/folders/1LoVTMlsopfQ8C3oAqFJ3xGavl8eiNfFD?usp=drive_link](https://drive.google.com/drive/folders/1LoVTMlsopfQ8C3oAqFJ3xGavl8eiNfFD?usp=drive_link)
  *(Contains the raw dataset zip and saved best model weights: `best_traditional_model.joblib` and `best_deep_learning_model.keras`)*
- **Presentation Video**: *(Recorded 5–10 minute demonstration link)*
- **Academic Report**: [report/report_outline.md](report/report_outline.md)

---

## Executive Summary & Research Question
- **Author**: Software Engineering student specializing in healthcare machine learning applications.
- **Mission Alignment**: Hospital readmission within 30 days is a primary indicator of inpatient discharge vulnerability, post-discharge care fragmentation, and acute clinical decompensation among patients with diabetes mellitus. Under the CMS Hospital Readmissions Reduction Program (HRRP), hospitals face severe financial penalties for excess 30-day readmissions. Our mission is to build an operational clinical risk stratification pipeline to identify high-risk diabetic patients at discharge, enabling targeted transitional care interventions (e.g., nurse follow-up calls, timely primary care appointments, medication reconciliation).
- **Core Research Question**: *Can deep neural networks with tf.data input pipelines outperform tuned cost-sensitive classical machine learning algorithms (e.g., LightGBM, Random Forest) in predicting 30-day hospital readmission under severe class imbalance (~1:8) and retrospective clinical EHR constraints?*
- **Operational Boundary**: This predictive model is an operational risk stratification tool for hospital readmission risk. It **does not diagnose diabetes** nor does it establish causal clinical etiology.

---

## Dataset Provenance, Access, & Characteristics
- **Repository**: [UCI Machine Learning Repository: Diabetes 130-US Hospitals (1999–2008)](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008) (Dataset ID: 296).
- **Landmark Reference**: Strack, B., DeShazo, J. P., Gennings, C., Olmo, J. L., Ventura, S., Cios, K. J., & Clore, J. N. (2014). *"Impact of HbA1c Measurement on Hospital Readmission Rates: Analysis of 70,000 Clinical Database Records."* BioMed Research International, 2014, Article ID 781670.
- **License**: Creative Commons Attribution 4.0 International ([CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)).
- **Verified Raw Dimensions**: **101,766 rows × 50 columns** (substantially exceeds the university requirement of $\ge 20,000$ raw samples).
- **Unique Patients**: **71,518 unique patients** (16,773 patients have multiple hospital encounters, accounting for 46.21% of all records).
- **Discrepancy Resolution**: The figure ~69,984 cited in literature represents unique patients remaining **after** excluding patients who expired in hospital (1,084 on first encounter), hospice discharges (461 on first encounter), and demographic anomalies.
- **Target Formulation**:
  - **Positive Class ($y = 1$)**: Readmission within 30 days (`readmitted == '<30'`), $N = 11,357$ (11.16% of raw dataset).
  - **Negative Class ($y = 0$)**: No readmission within 30 days (`readmitted == 'NO'` or `'>30'`), $N = 90,409$ (88.84% of raw dataset).
- **Dataset Limitations**:
  - Historical clinical care data (1999–2008); modern diabetes medications (SGLT2 inhibitors, GLP-1 receptor agonists) are absent.
  - High missingness in specific fields (`weight` > 96% missing; `medical_specialty` ~ 49% missing).
  - Lacks out-of-hospital mortality tracking and post-discharge social determinants of health (SDOH).

---

## Leakage Prevention & Patient-Grouped Splitting (Strategy B)
Because 23.45% of patients have multiple hospitalizations, standard row-level random splitting causes severe patient-level data leakage. 
We enforce **Strategy B (`GroupShuffleSplit` on `patient_nbr`)**, guaranteeing:
- **Zero Patient Overlap**: Exactly 0 patients shared across Train, Validation, and Test splits.
- **Clinical Exclusions**: Excluded 1,652 encounters where the patient expired in hospital (discharge disposition IDs 11, 19, 20, 21), because 100% were labeled `readmitted == 'NO'` (dead patients cannot physically be readmitted; retaining them distorts risk modeling).
- **Eligible Encounters**: 100,114 encounters across 70,439 unique patients.
  - **Train Split (70%)**: 70,055 encounters (49,305 unique patients), 11.29% positive.
  - **Validation Split (15%)**: 15,059 encounters (10,568 unique patients), 11.76% positive.
  - **Test Split (15%)**: 15,000 encounters (10,566 unique patients), 11.20% positive.

---

## Systematically Varied Experiments & Benchmarking
We executed **10 systematically varied experiments** (6 Traditional ML + 4 Deep Learning) tracked in `results/experiment_results.csv` and logged via Weights & Biases in offline mode:

| Experiment ID | Model Family | Algorithm / Configuration | Val ROC-AUC | Val PR-AUC | Val Recall | Val F1 | Train Time (s) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **EXP-ML-00** | Traditional ML | Dummy Baseline (Strategy='prior') | 0.5000 | 0.1176 | 0.0000 | 0.0000 | 0.01 |
| **EXP-ML-01** | Traditional ML | Logistic Regression (Unweighted, C=1.0) | 0.6752 | 0.2386 | 0.0237 | 0.0457 | 3.87 |
| **EXP-ML-02** | Traditional ML | Logistic Regression (Balanced, C=0.1) | 0.6765 | 0.2369 | 0.5618 | 0.2834 | 5.01 |
| **EXP-ML-03** | Traditional ML | Random Forest (Balanced, max_depth=12) | 0.6780 | 0.2323 | 0.5042 | 0.2880 | 7.11 |
| **EXP-ML-04** | Traditional ML | HistGradientBoosting (Default, lr=0.08) | 0.6795 | 0.2549 | 0.0141 | 0.0277 | 5.59 |
| **EXP-ML-05** | Traditional ML | **LightGBM (Cost-Sensitive, scale_pos=7.8)** | **0.6813** | **0.2565** | **0.5675** | **0.2904** | **0.84** |
| **EXP-DL-01** | Deep Learning | Baseline MLP (Sequential, 64-32, lr=1e-3) | 0.6820 | 0.2490 | 0.0169 | 0.0331 | 18.67 |
| **EXP-DL-02** | Deep Learning | Deep Regularized MLP (Functional, 128-64-32) | 0.6804 | 0.2444 | 0.0124 | 0.0244 | 44.54 |
| **EXP-DL-03** | Deep Learning | Wide MLP (128-32, lr=5e-4, batch=512) | 0.6816 | 0.2497 | 0.0158 | 0.0309 | 26.09 |
| **EXP-DL-04** | Deep Learning | **Cost-Sensitive Deep MLP (weight=7.8)** | **0.6842** | **0.2513** | **0.5551** | **0.2895** | **48.24** |

---

## Final Evaluation on Untouched Held-Out Test Set
Selected based strictly on validation ROC-AUC and evaluated on the untouched test partition ($N = 15,000$ encounters, $10,566$ independent patients):

| Metric | Traditional ML (LightGBM Cost-Sensitive) | Deep Learning (Cost-Sensitive Deep MLP) |
| :--- | :---: | :---: |
| **Test ROC-AUC** | **0.6566** | **0.6574** |
| **Test PR-AUC** | **0.2040** | **0.2108** |
| **Test Sensitivity (Recall)** | **54.17%** | **52.92%** |
| **Test Specificity** | **68.47%** | **69.21%** |
| **Test Precision (PPV)** | **17.81%** | **17.82%** |
| **Test F1-Score** | **0.2680** | **0.2666** |
| **Test Accuracy** | **66.87%** | **67.39%** |
| **Training Time** | **0.84 seconds** | **48.24 seconds** |

---

## Repository Structure
```text
diabetes-readmission-ml-dl/
├── README.md                                  # Project overview and instructions
├── requirements.txt                           # Verified package dependencies
├── .gitignore                                 # Prevents raw data and binaries from git
├── run_experiments.py                         # Master execution script for 10 experiments
├── notebooks/
│   └── diabetes_readmission_classification.ipynb  # End-to-end Colab-ready notebook
├── src/
│   ├── data_preprocessing.py                  # Loading, verification, and patient-grouped split
│   ├── feature_engineering.py                 # ICD-9 grouping and ColumnTransformer pipeline
│   ├── traditional_models.py                  # Scikit-learn and LightGBM model definitions
│   ├── deep_learning_models.py                # Keras architectures & tf.data input pipelines
│   └── evaluation.py                          # ROC/PR curves, confusion matrices, and metrics
├── results/
│   ├── experiment_results.csv                 # Master validation benchmark table (10 runs)
│   ├── final_test_evaluation.csv              # Held-out test set performance summary
│   └── figures/
│       ├── comparative_roc_pr_curves.png      # Comparative ROC & PR curves
│       ├── confusion_matrix_traditional_test.png # Test confusion matrix for LightGBM
│       ├── confusion_matrix_deep_learning_test.png # Test confusion matrix for Deep MLP
│       ├── exp-dl-01_learning_curves.png      # Baseline MLP training curves
│       ├── exp-dl-02_learning_curves.png      # Deep Regularized MLP training curves
│       ├── exp-dl-03_learning_curves.png      # Wide MLP training curves
│       └── exp-dl-04_learning_curves.png      # Cost-Sensitive Deep MLP training curves
├── models/
│   ├── best_traditional_model.joblib          # Saved LightGBM pipeline artifact
│   ├── best_deep_learning_model.keras         # Saved native Keras model artifact
│   └── model_metadata.json                    # Feature schema and performance metadata
└── report/
    └── report_outline.md                      # IEEE report structure & 11 scholarly citations
```

---

## Quickstart & Execution Instructions

### Option 1: Running in Google Colab (Recommended)
1. Open Google Colab and click **File $\rightarrow$ Open notebook $\rightarrow$ GitHub**.
2. Paste the repository URL: `https://github.com/Amazing-beep/diabetes-readmission-ml-dl`.
3. Select `notebooks/diabetes_readmission_classification.ipynb`.
4. Click **Runtime $\rightarrow$ Run all**. The notebook automatically downloads the raw dataset from the official UCI repository, runs all 10 experiments, and renders all plots top-to-bottom.

### Option 2: Running Locally
1. **Clone the repository**:
   ```bash
   git clone https://github.com/Amazing-beep/diabetes-readmission-ml-dl.git
   cd diabetes-readmission-ml-dl
   ```
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Execute the full experimental suite**:
   ```bash
   python run_experiments.py
   ```
   *All models will be trained, evaluated, and saved to `models/` and `results/`.*

---

## Model Artifacts & Google Drive Storage
Per university instructions, large binary model artifacts and the raw zipped dataset are stored in the designated Google Drive project folder:
- **Google Drive Link**: [https://drive.google.com/drive/folders/1LoVTMlsopfQ8C3oAqFJ3xGavl8eiNfFD?usp=drive_link](https://drive.google.com/drive/folders/1LoVTMlsopfQ8C3oAqFJ3xGavl8eiNfFD?usp=drive_link)
- **Included Artifacts**:
  1. `diabetes+130-us+hospitals+for+years+1999-2008.zip` (Raw dataset archive)
  2. `best_traditional_model.joblib` (Trained LightGBM model with fitted `ColumnTransformer`)
  3. `best_deep_learning_model.keras` (Trained native Keras deep neural network)
  4. `model_metadata.json` (Input dimensions, hyperparameter configurations, and metrics)
