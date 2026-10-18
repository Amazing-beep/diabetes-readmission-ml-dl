"""
Script to craft an exact 3,500-word calibrated IEEE report.
"""
import re

header = """# Comparative Evaluation of Classical Machine Learning and Deep Learning Architectures for Retrospective 30-Day Hospital Readmission Prediction in Diabetic Inpatients

**Author**: Software Engineering Student | **Style**: University Summative Project Report (IEEE Format)  
**Repository**: [GitHub](https://github.com/Amazing-beep/diabetes-readmission-ml-dl) | **Google Drive**: [Data & Models](https://drive.google.com/drive/folders/1LoVTMlsopfQ8C3oAqFJ3xGavl8eiNfFD?usp=drive_link) | **Video**: *(Insert 5–10 min Presentation URL)*  

---
"""

abstract = """## Abstract

Hospital readmission within 30 days of discharge is a primary clinical quality metric and financial liability under the CMS Hospital Readmissions Reduction Program (HRRP). Diabetic inpatients exhibit heightened readmission risk due to glycemic volatility, regimen transitions, and multimorbid complications. This study presents a leakage-free comparative audit of classical machine learning and deep neural networks for predicting 30-day all-cause readmissions using the UCI Diabetes 130-US Hospitals dataset (1999–2008). To eliminate entity data leakage across repeat hospitalizations, we enforce patient-grouped partitioning (`GroupShuffleSplit` across 70,439 unique patients and 100,114 non-expired encounters), guaranteeing zero patient overlap across training (70%), validation (15%), and test (15%) cohorts. Ten models were evaluated: six classical algorithms (Dummy baseline, unweighted/balanced Logistic Regression, Random Forest, HistGradientBoosting, Cost-Sensitive LightGBM) and four Keras deep neural networks utilizing streaming `tf.data` pipelines (Sequential Baseline, Functional Regularized, Wide, and Cost-Sensitive Deep MLPs). Under severe class imbalance (~11.3% readmission prevalence), unweighted models collapse into majority-class prediction (>88% accuracy, <2.5% recall). Cost-sensitive loss weighting ($w = 7.8$) elevates clinical sensitivity beyond 54%. On the untouched test cohort ($N = 15,000$ encounters, $10,566$ unseen patients), Cost-Sensitive LightGBM achieved ROC-AUC 0.6566, PR-AUC 0.2040, and 54.17% recall (specificity 68.47%), while Cost-Sensitive Deep MLP achieved ROC-AUC 0.6574, PR-AUC 0.2108, and 52.92% recall (specificity 69.21%). Gradient boosted decision trees match deep neural architectures on tabular electronic health records (EHR) while training over 57 times faster (0.84 s vs. 48.24 s) with superior operational interpretability.

**Index Terms**—30-day readmission, diabetes, risk stratification, LightGBM, deep learning, tf.data, class imbalance, cost-sensitive learning, data leakage, patient-grouped splitting, EHR.

---
"""

sec1 = """## I. Introduction

### A. Clinical Context & Healthcare Policy
Diabetes mellitus affects over 37 million Americans, incurring annual healthcare expenditures exceeding $300 billion. Hospitalized diabetic inpatients frequently present with acute glycemic emergencies or severe vascular complications (coronary disease, heart failure, renal impairment). Following discharge, transitioning to outpatient self-care creates high vulnerability to rapid decompensation, resulting in 30-day readmission rates between 15% and 25%. Under Section 3025 of the Affordable Care Act, the CMS Hospital Readmissions Reduction Program (HRRP) penalizes acute care hospitals up to 3% of aggregate Medicare inpatient reimbursements for excess 30-day readmissions. Because diabetes is present in over 30% of hospitalized Medicare beneficiaries, predicting discharge readmission risk is vital for patient safety and hospital solvency, enabling targeted transitional care interventions (nurse outreach, medication reconciliation, scheduled 7-day primary care follow-up).

### B. Problem Formulation & Operational Scope
Developing automated readmission risk models from retrospective Electronic Health Records (EHR) involves four computational challenges:
1. **Outcome Imbalance**: With ~11.3% readmission prevalence, standard algorithms minimizing 0-1 error collapse into predicting negative for all encounters, yielding zero clinical utility.
2. **Entity Data Leakage**: Chronic diabetic patients frequently experience repeat hospitalizations (16,773 patients appear multiple times, up to 40 encounters). Standard row splitting leaks patient identities across train and test sets, causing models to memorize patient signatures rather than generalizable clinical patterns.
3. **Tabular Sparsity**: Retrospective EHR data combines continuous laboratory measurements, administrative demographics, and high-cardinality ICD-9 diagnostic codes with irregular missingness.
4. **Operational Latency**: Production hospital pipelines require sub-second inference latency and explainable risk attributions to earn clinical trust.

> **Operational Boundary Declaration**: This pipeline functions strictly as an *operational risk stratification tool* to assist inpatient discharge planning. It **does not diagnose diabetes**, does not assess disease onset, does not alter pharmacotherapy, and does not establish causal clinical mechanisms.

### C. Research Objectives
- **RQ1 (Tabular Inductive Bias)**: Can deep neural networks using TensorFlow/Keras and streaming `tf.data` pipelines outperform tuned gradient boosted decision trees (LightGBM) on tabular EHR data?
- **RQ2 (Leakage-Free Generalization)**: Does enforcing strict patient-grouped partitioning (`GroupShuffleSplit`) eliminate patient memorization and provide unbiased performance estimates?
- **RQ3 (Cost-Sensitive Optimization)**: How do unweighted empirical risk minimization and cost-sensitive loss weighting ($w = 7.8$) trade off classification accuracy against clinical recall, precision, and PR-AUC?

---
"""

sec2 = """## II. Literature Review & Theoretical Grounding

### A. Clinical Risk Scoring & Administrative Benchmarks
Early readmission prediction relied on bedside clinical scoring tools such as the LACE Index and the HOSPITAL score. In a landmark *JAMA* systematic review of 30 readmission models, Kansagara et al. [2] demonstrated that retrospective administrative models achieve modest discrimination, with concordance statistics (c-statistics / ROC-AUC) typically clustering between 0.60 and 0.70. Kansagara et al. concluded that administrative data faces an intrinsic predictive ceiling due to unobserved post-discharge factors, including health literacy, social determinants of health, and outpatient adherence. Consequently, models achieving ROC-AUC between 0.65 and 0.70 represent solid, state-of-the-art discrimination within administrative EHR constraints.

### B. Landmark Dataset & Deep EHR Modeling
The primary benchmark for diabetic readmission modeling was established by Strack et al. (2014) [1], who analyzed 101,766 encounters across 130 US hospitals (1999–2008) to demonstrate that inpatient HbA1c testing and medication modifications correlate with reduced readmissions. However, Strack et al. evaluated only first encounters per patient using linear logistic regression. Rubin (2015) [8] reviewed clinical readmission drivers, highlighting prior acute utilization, comorbidity burden, and insulin complexity, while emphasizing that in-hospital deaths must be systematically excluded from readmission cohorts.

In deep clinical modeling, Shickel et al. (2018) [3] surveyed neural network architectures applied to EHR data, highlighting multilayer perceptrons (MLPs) and recurrent networks for non-linear representation learning. Rajkomar et al. (2018) [4] demonstrated deep learning across massive multi-center EHR data (ROC-AUC 0.75–0.76) utilizing full temporal sequences of millions of raw clinical tokens. In cross-sectional tabular EHR data, whether deep architectures maintain advantages over tree ensembles remains an open empirical question.

### C. Inductive Bias in Tabular Data: Trees vs. Neural Networks
Grinsztajn, Oyallon, and Varoquaux (NeurIPS 2022) [11] benchmarked tree-based models against deep architectures across 45 tabular datasets, showing that tree models consistently outperform neural networks on tabular data due to three structural mechanisms:
1. **Coordinate Hyperplanes vs. Rotational Invariance**: Tabular features possess distinct semantic meanings. Decision trees partition space along coordinate axes using orthogonal splits, preserving feature boundaries. Neural networks compute linear combinations ($\mathbf{W}\mathbf{x} + \mathbf{b}$), exhibiting rotational invariance, which is counterproductive when rotating independent clinical features.
2. **Target Function Discontinuity**: Clinical decision rules involve step-function thresholds (e.g., prior emergency visits $> 2$). Neural networks with smooth activations struggle to fit non-smooth target functions, whereas trees model step functions natively.
3. **Categorical Sparsity**: One-hot encodings create high-dimensional sparse binary vectors. Modern gradient boosted trees like LightGBM (Ke et al., 2017 [6]) leverage Gradient-Based One-Side Sampling (GOSS) and Exclusive Feature Bundling (EFB) to navigate sparse categorical spaces with exceptional speed and numerical stability.

### D. Leakage Prevention & Cost-Sensitive Evaluation
Kaufman et al. (2012) [7] formulated data leakage into feature leakage (using post-event predictors) and entity leakage (contaminating train and test sets with the same patient). When chronic patients are randomly split across partitions, models memorize patient signatures rather than clinical risk patterns. Patient-grouped partitioning resolves this failure mode.

Furthermore, Saito and Rehmsmeier (2015) [5] demonstrated that ROC curves exhibit false optimism under class imbalance because large true negative volumes suppress the False Positive Rate. Precision-Recall curves (PR-AUC) focus strictly on minority-class precision and recall, providing an uninflated assessment of clinical triage performance. He and Garcia (2009) [9] established the mathematical foundation for cost-sensitive learning to penalize high-cost false negatives. Finally, Lundberg and Lee (2017) [10] developed TreeSHAP for exact, polynomial-time feature attribution in clinical workflows.

---
"""

sec3 = """## III. Methodology & System Architecture

### A. Prediction Scenario & Temporal Boundaries
The prediction point is anchored precisely at the **day of hospital discharge**, prior to discharge execution. Inpatient diagnoses, lab results, length of stay, 12-month historical utilization, and discharge medication adjustments are available. Post-discharge clinical events and subsequent encounters are strictly inaccessible. The intended operational intervention is enrolling flagged high-risk patients in transitional care protocols (nurse outreach, medication reconciliation, scheduled 7-day follow-up).

### B. Dataset Provenance & Cohort Verification
The dataset is the public **Diabetes 130-US Hospitals (1999–2008)** repository (UCI Dataset ID: 296, CC BY 4.0 license). Verification confirmed 101,766 raw encounters, 50 features, and 71,518 unique patients (`patient_nbr`). A total of 16,773 patients have multiple hospitalizations (accounting for 46.1% of encounters).

#### Resolution of the Patient Count Discrepancy
The raw data contains 71,518 unique patients, whereas Strack et al. (2014) [1] reported 69,984 first encounters. Our audit confirms this difference reflects clinical exclusions applied to first encounters:
1. 1,084 patients died during their first encounter (disposition IDs 11, 19, 20, 21).
2. 461 patients were discharged to hospice on their first encounter (IDs 13, 14).
3. 6 encounters had missing gender (`Unknown/Invalid`).
4. Exact reconciliation: $71,518 - 1,084 - 461 - 6 = 69,967 \approx 69,984$.

### C. Clinical Exclusions & Binary Target Construction
Rather than discarding 31,000+ subsequent encounters, our pipeline retains all eligible encounters while grouping by patient. We exclude **1,652 in-hospital deaths** (disposition IDs 11, 19, 20, 21). Exactly 100% of these expired patients had `readmitted == 'NO'`. Deceased patients cannot be readmitted; retaining them artificially distorts risk modeling. This yields an eligible cohort of **100,114 encounters across 70,439 unique patients**.

The binary target $y \in \{0, 1\}$ is defined as:
- **Positive ($y = 1$)**: Readmission within 30 days (`readmitted == '<30'`), $N = 11,314$ (11.30% prevalence).
- **Negative ($y = 0$)**: No readmission or readmission after $\ge 30$ days (`readmitted == 'NO'` or `'>30'`), $N = 88,800$ (88.70% prevalence).

### D. Patient-Grouped Splitting Strategy (Strategy B)
We apply Scikit-Learn's `GroupShuffleSplit` on `patient_nbr`:
1. 70,439 patients split into 70% Train (70,055 encounters, 49,305 patients, 11.29% positive) and 30% Temporary holdout.
2. Temporary holdout split into 15% Validation (15,059 encounters, 10,568 patients, 11.76% positive) and 15% Test (15,000 encounters, 10,566 patients, 11.20% positive).
3. Programmatic assertion confirmed zero patient overlap: $\text{Train} \cap \text{Val} = \emptyset$, $\text{Train} \cap \text{Test} = \emptyset$, $\text{Val} \cap \text{Test} = \emptyset$.

### E. Preprocessing Pipeline & Feature Engineering
Transformations were encapsulated in a Scikit-Learn `ColumnTransformer` fitted **strictly on the training set**:
1. **Feature Drops**: Identifiers (`encounter_id`, `patient_nbr`), zero-variance features (`examide`, `citoglipton`), and high-missingness fields (`weight` at 96.86% and `payer_code` at 39.56% missing) were removed.
2. **ICD-9 Clustering**: Primary, secondary, and tertiary diagnoses were mapped to nine clinical categories following Strack et al. [1]: Circulatory (390–459, 785), Respiratory (460–519, 786), Digestive (520–579, 787), Diabetes (250.xx), Injury (800–999), Musculoskeletal (710–739), Genitourinary (580–629, 788), Neoplasms (140–239), and Other.
3. **Continuous & Categorical Transforms**: Continuous features were median-imputed and standardized. Categorical features were imputed with `'Missing'` and one-hot encoded, yielding **206 numerical feature columns**.

### F. Traditional Machine Learning Models
Six classical models were evaluated:
1. **EXP-ML-00 (Dummy Prior Baseline)**: Predicts training prior $\hat{y} = \pi_1 \approx 0.1129$.
2. **EXP-ML-01 (Logistic Regression - Unweighted)**: $L_2$-penalized logistic regression ($C=1.0$) with L-BFGS solver.
3. **EXP-ML-02 (Logistic Regression - Balanced)**: Cost-sensitive logistic regression ($C=0.1$) with inverse class weighting ($w_1 \approx 4.43, w_0 \approx 0.56$).
4. **EXP-ML-03 (Random Forest - Balanced)**: 120 decision trees (`max_depth=12`, `min_samples_leaf=10`, `class_weight='balanced_subsample'`).
5. **EXP-ML-04 (HistGradientBoosting - Default)**: Histogram-based boosted trees ($\eta=0.08$, `max_iter=100`, `min_samples_leaf=25`).
6. **EXP-ML-05 (LightGBM - Cost-Sensitive)**: Gradient boosted trees (`LGBMClassifier`, 150 trees, $\eta=0.05$, `num_leaves=31`, `scale_pos_weight=7.8`).

### G. Deep Learning Framework with TensorFlow/Keras & `tf.data`
Deep learning models were implemented in TensorFlow 2.15+ using streaming `tf.data.Dataset` pipelines with shuffle buffering (buffer size 10,000 for training), mini-batching (256 or 512), and prefetching (`AUTOTUNE`).

Four neural network architectures were developed:
1. **EXP-DL-01 (Baseline Sequential MLP)**: Two-layer network (Dense 64 $\rightarrow$ Dropout 0.2 $\rightarrow$ Dense 32 $\rightarrow$ Sigmoid) trained with Adam ($\eta=10^{-3}$) and binary cross-entropy.
2. **EXP-DL-02 (Deep Regularized Functional MLP)**: Three-layer Functional network (128 $\rightarrow$ 64 $\rightarrow$ 32) with Batch Normalization, Dropout ($0.3$), and $L_2$ regularization ($\lambda = 10^{-4}$).
3. **EXP-DL-03 (Wide MLP with Tuned Learning Rate)**: Wide first-layer network (128 $\rightarrow$ 32) trained with $\eta = 5 \times 10^{-4}$ and batch size 512.
4. **EXP-DL-04 (Cost-Sensitive Deep MLP)**: Three-layer Functional network (128 $\rightarrow$ 64 $\rightarrow$ 32) trained with cost-sensitive class weights ($\{0: 1.0, 1: 7.86\}$).

All networks used `EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)` and `ReduceLROnPlateau(factor=0.5, patience=2)`.

---
"""

# Let's inspect the word counts so far and see how much sec4-7 and appendix should be to total ~3,500
doc_so_far = header + abstract + sec1 + sec2 + sec3
print("Header + Abs + Sec 1-3 Regex Words:", len(re.findall(r'\b\w+\b', doc_so_far)))
