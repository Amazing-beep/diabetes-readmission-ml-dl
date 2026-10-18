import re

header = """# Comparative Evaluation of Classical Machine Learning and Deep Learning Architectures for Retrospective 30-Day Hospital Readmission Prediction in Diabetic Inpatients

**Author**: Software Engineering Student  
**Target Venue**: University Summative Project Report (IEEE Format)  
**Repository**: [https://github.com/Amazing-beep/diabetes-readmission-ml-dl](https://github.com/Amazing-beep/diabetes-readmission-ml-dl)  
**Deliverables**: [Google Drive Folder](https://drive.google.com/drive/folders/1LoVTMlsopfQ8C3oAqFJ3xGavl8eiNfFD?usp=drive_link)  
**Presentation Video**: *(Insert recorded 5–10 minute presentation URL here)*  

---
"""

abstract = """## Abstract

Hospital readmission within 30 days of discharge is a vital clinical quality metric and financial liability under the CMS Hospital Readmissions Reduction Program (HRRP). Diabetic inpatients face heightened readmission risks from glycemic volatility and multimorbid complications. This investigation presents an empirical comparative evaluation of classical machine learning and deep neural networks for predicting 30-day all-cause readmission using the landmark UCI Diabetes 130-US Hospitals dataset (1999–2008). To eliminate entity data leakage across repeat hospitalizations, we enforce strict patient-grouped partitioning (`GroupShuffleSplit` across 70,439 unique patients over 100,114 non-expired encounters), guaranteeing zero patient overlap across training (70%), validation (15%), and test (15%) partitions. We benchmark ten models: six classical algorithms (Dummy baseline, unweighted/balanced Logistic Regression, Random Forest, HistGradientBoosting, Cost-Sensitive LightGBM) and four deep architectures in TensorFlow/Keras with streaming `tf.data` pipelines (Sequential Baseline, Functional Regularized, Wide, and Cost-Sensitive Deep MLP). Under severe class imbalance (~11.3% readmission prevalence), unweighted models collapse into majority-class prediction (>88% accuracy, <2.5% recall). Cost-sensitive loss weighting ($w = 7.8$) recovers clinical utility, elevating sensitivity above 54%. On the held-out test cohort ($N = 15,000$ encounters, 10,566 independent patients), Cost-Sensitive LightGBM achieved test ROC-AUC 0.6566, PR-AUC 0.2040, and 54.17% recall (specificity 68.47%). Cost-Sensitive Deep MLP achieved test ROC-AUC 0.6574, PR-AUC 0.2108, and 52.92% recall (specificity 69.21%). Supporting the tabular inductive bias hypothesis, gradient boosted decision trees match deep neural networks on tabular Electronic Health Records (EHR) while training over 57 times faster (0.84 s vs. 48.24 s) and providing native TreeSHAP interpretability for point-of-care clinical trust.

**Index Terms**—30-day readmission, diabetes, clinical risk stratification, LightGBM, deep learning, tf.data, class imbalance, cost-sensitive learning, data leakage, patient-grouped splitting, EHR.

---
"""

sec1 = """## I. Introduction

### A. Clinical Context & Healthcare Policy
Diabetes mellitus impacts over 37 million Americans, incurring annual healthcare costs exceeding $300 billion. Hospitalized diabetic inpatients face elevated post-discharge vulnerabilities from glycemic instability and multimorbid complications, yielding 30-day readmission rates between 15% and 25%. Under the CMS Hospital Readmissions Reduction Program (HRRP), hospitals incur reimbursement penalties up to 3% for excess readmissions. Because diabetes affects over 30% of hospitalized Medicare beneficiaries, discharge risk stratification is an operational necessity. Flagging high-risk patients enables targeted transitional interventions, such as nurse outreach, medication reconciliation, and rapid 7-day follow-ups.

### B. Computational Challenges & Operational Scope
Predicting readmission from retrospective Electronic Health Records (EHR) involves four key challenges:
1. **Severe Class Imbalance**: Readmissions comprise ~11% of encounters, causing unweighted models to collapse into majority-class prediction.
2. **Entity Data Leakage**: Chronic diabetic patients frequently experience repeat hospitalizations (16,773 patients appear multiple times, up to 40). Random row splitting leaks patient signatures across partitions, causing models to memorize individuals rather than risk patterns.
3. **Tabular Sparsity**: EHR combines continuous vitals, administrative codes, and high-cardinality ICD-9 codes.
4. **Operational Latency**: Point-of-care deployment demands sub-second inference and transparent feature attributions.

> **Operational Boundary Declaration**: This system operates strictly as an *operational risk stratification tool* to assist discharge planning. It **does not diagnose diabetes**, alter medications, or establish causal mechanisms.

### C. Research Objectives
- **RQ1 (Tabular Inductive Bias)**: Can deep neural networks with streaming `tf.data` pipelines outperform tuned gradient boosted trees (LightGBM) on tabular EHR data?
- **RQ2 (Leakage-Free Generalization)**: Does enforcing patient-grouped partitioning (`GroupShuffleSplit` on `patient_nbr`) prevent optimistic patient memorization?
- **RQ3 (Cost-Sensitive Optimization)**: How do unweighted empirical risk minimization and cost-sensitive loss weighting ($w = 7.8$) trade off nominal accuracy against clinical recall, precision, and PR-AUC?

---
"""

sec2 = """## II. Literature Review & Theoretical Grounding

### A. Clinical Risk Scoring & Administrative Benchmarks
Early readmission forecasting relied on heuristic scoring tools such as the LACE Index and HOSPITAL score. In a systematic review in *JAMA*, Kansagara et al. [2] examined 30 readmission models, finding administrative EHR discrimination tops out between ROC-AUC 0.60–0.70. This ceiling reflects unmeasured social determinants of health, health literacy, and outpatient treatment compliance. Consequently, administrative EHR models achieving ROC-AUC 0.65–0.70 represent solid, state-of-the-art discrimination under real-world operational constraints.

### B. Benchmark Dataset & Deep Learning in EHR
Strack et al. (2014) [1] introduced the landmark diabetic readmission benchmark (101,766 encounters, 130 US hospitals, 1999–2008), demonstrating that inpatient HbA1c testing and medication adjustments correlate with lower readmission rates, though modeling only single encounters with logistic regression. Rubin (2015) [8] examined clinical readmission drivers, emphasizing comorbidity burden, insulin complexity, and the necessity of excluding in-hospital mortality. For deep clinical modeling, Shickel et al. (2018) [3] surveyed deep architectures for EHR analysis. Rajkomar et al. (2018) [4] demonstrated deep learning on multi-center EHRs (ROC-AUC 0.75–0.76) using temporal sequences of clinical tokens. Whether deep architectures outperform tree ensembles on tabular EHR remains an active empirical question.

### C. Inductive Bias in Tabular Data: Decision Trees vs. Neural Networks
Grinsztajn, Oyallon, and Varoquaux (NeurIPS 2022) [11] benchmarked tree ensembles against deep architectures across 45 tabular datasets, showing trees consistently outperform neural networks due to three structural factors:
1. **Coordinate Hyperplanes vs. Rotational Invariance**: Tabular features possess distinct semantic meanings. Decision trees partition space along orthogonal coordinate splits, whereas neural networks compute linear combinations, exhibiting rotational invariance that degrades performance on non-isotropic tabular manifolds.
2. **Target Function Discontinuity**: Clinical risk rules often involve step-function thresholds (e.g., prior emergency visits $> 2$). Neural networks with smooth activations struggle to fit discontinuous target functions, whereas trees model step functions natively.
3. **Categorical Sparsity**: One-hot encodings yield high-dimensional sparse binary vectors. Modern gradient boosted trees like LightGBM (Ke et al., 2017 [6]) leverage Gradient-Based One-Side Sampling (GOSS) and Exclusive Feature Bundling (EFB) to navigate sparse categorical spaces efficiently.

### D. Leakage Prevention & Cost-Sensitive Evaluation
Kaufman et al. (2012) [7] formulated data leakage into feature leakage and entity leakage (contaminating train and test sets with identical patient records). Patient-grouped partitioning prevents models from memorizing repeat-patient signatures. Saito and Rehmsmeier (2015) [5] showed that ROC curves exhibit false optimism under class imbalance because massive true negative volumes suppress the False Positive Rate. Precision-Recall curves (PR-AUC) focus strictly on minority-class precision and recall. He and Garcia (2009) [9] established cost-sensitive learning to penalize high-cost false negatives. Finally, Lundberg and Lee (2017) [10] emphasized feature attribution (TreeSHAP) for clinical auditability.

---
"""

sec3 = """## III. Methodology & System Architecture

### A. Prediction Scenario & Temporal Boundaries
The prediction point is anchored at hospital discharge. Inpatient diagnoses, lab tests, length of stay, 12-month historical utilization, and discharge medication adjustments are observed, while post-discharge events are strictly masked. Flagged high-risk patients are routed to transitional care protocols (nurse outreach, medication reconciliation, and scheduled 7-day follow-up).

### B. Dataset Provenance & Cohort Verification
The dataset is the public **Diabetes 130-US Hospitals (1999–2008)** repository (UCI ID: 296), comprising 101,766 raw encounters, 50 features, and 71,518 unique patients (`patient_nbr`), with 16,773 patients admitted multiple times (46.1% of encounters).  
*Discrepancy Resolution*: The 69,984 first encounters reported by Strack et al. [1] reflect excluding 1,084 first-encounter deaths (disposition IDs 11, 19, 20, 21), 461 hospice discharges (IDs 13, 14), and 6 missing gender records: $71,518 - 1,084 - 461 - 6 = 69,967 \\approx 69,984$.

### C. Clinical Exclusions & Binary Target Formulation
To retain longitudinal history without entity distortion, we include all eligible admissions while excluding **1,652 in-hospital deaths** (IDs 11, 19, 20, 21; 100% negative class), as deceased individuals cannot be readmitted. This leaves **100,114 encounters across 70,439 unique patients**. The binary target $y \\in \\{0, 1\\}$ classifies readmissions within 30 days (`<30`, $N = 11,314$, 11.30% positive) versus non-readmissions or $\\ge 30$-day readmissions (`NO` / `>30`, $N = 88,800$, 88.70% negative).

### D. Patient-Grouped Partitioning
To prevent entity leakage, Scikit-Learn's `GroupShuffleSplit` on `patient_nbr` partitions the cohort into:
- **Training Set (70%)**: 70,055 encounters, 49,305 patients (11.29% positive).
- **Validation Set (15%)**: 15,059 encounters, 10,568 patients (11.76% positive).
- **Test Set (15%)**: 15,000 encounters, 10,566 patients (11.20% positive).  
Programmatic checks verify zero patient overlap across splits ($\\text{Train} \\cap \\text{Val} = \\text{Train} \\cap \\text{Test} = \\text{Val} \\cap \\text{Test} = \\emptyset$).

### E. Preprocessing Pipeline & Feature Engineering
A `ColumnTransformer` fitted strictly on training data handles feature transforms:
1. **Removals**: IDs (`encounter_id`, `patient_nbr`), zero-variance features (`examide`, `citoglipton`), and sparse fields (`weight` 96.86%, `payer_code` 39.56% missing) were removed.
2. **ICD-9 Grouping**: Diagnoses mapped to 9 clinical categories [1]: Circulatory (390–459, 785), Respiratory (460–519, 786), Digestive (520–579, 787), Diabetes (250.xx), Injury (800–999), Musculoskeletal (710–739), Genitourinary (580–629, 788), Neoplasms (140–239), and Other.
3. **Encoding**: Continuous features were median-imputed and standardized; categoricals were imputed with `'Missing'` and one-hot encoded, yielding **206 numerical features**.

### F. Traditional Machine Learning Models
Six classical models were benchmarked:
1. **EXP-ML-00 (Dummy Prior)**: Predicting $\\hat{y} = \\pi_1 \\approx 0.1129$.
2. **EXP-ML-01 (Logistic Regression - Unweighted)**: $L_2$ regularized ($C=1.0$).
3. **EXP-ML-02 (Logistic Regression - Balanced)**: Cost-sensitive ($C=0.1$, $w_1 \\approx 4.43, w_0 \\approx 0.56$).
4. **EXP-ML-03 (Random Forest - Balanced)**: 120 trees (`max_depth=12`, `min_samples_leaf=10`).
5. **EXP-ML-04 (HistGradientBoosting - Default)**: Histogram boosted trees ($\\eta=0.08$, `max_iter=100`, `min_samples_leaf=25`).
6. **EXP-ML-05 (LightGBM - Cost-Sensitive)**: Boosted trees (150 trees, $\\eta=0.05$, `num_leaves=31`, `scale_pos_weight=7.8`).

### G. Deep Learning Framework (TensorFlow/Keras & `tf.data`)
Neural models utilized streaming `tf.data` pipelines with shuffle buffering (10,000), batching (256/512), and prefetching (`AUTOTUNE`):
1. **EXP-DL-01 (Baseline Sequential MLP)**: Dense 64 $\\rightarrow$ Dropout 0.2 $\\rightarrow$ Dense 32 $\\rightarrow$ Sigmoid ($\\eta=10^{-3}$, BCE).
2. **EXP-DL-02 (Functional Regularized MLP)**: Dense 128 $\\rightarrow$ 64 $\\rightarrow$ 32 with Batch Normalization, Dropout (0.3), and $L_2$ ($\\lambda=10^{-4}$).
3. **EXP-DL-03 (Wide MLP)**: Dense 128 $\\rightarrow$ 32 ($\\eta=5 \\times 10^{-4}$, batch size 512).
4. **EXP-DL-04 (Cost-Sensitive Deep MLP)**: Functional 128 $\\rightarrow$ 64 $\\rightarrow$ 32 with cost-sensitive class weights ($\\{0: 1.0, 1: 7.86\\}$).  
All networks employed `EarlyStopping(monitor='val_loss', patience=5)` and `ReduceLROnPlateau(factor=0.5, patience=2)`.

---
"""

sec4 = """## IV. Experimental Results & Empirical Analysis

### A. Master Validation Benchmark (Ten Experiments)
All ten models were trained on 70,055 training encounters and evaluated across 15,059 validation encounters (Table I).

#### TABLE I: Master Validation Benchmark Across Ten Experimental Configurations
| Exp ID | Model Family | Algorithm / Hyperparameters | Val ROC-AUC | Val PR-AUC | Val Recall | Val Specificity | Val Precision | Val F1 | Val Acc | Val Brier | Training Time |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **EXP-ML-00** | Traditional | Dummy Baseline (`strategy='prior'`) | 0.5000 | 0.1176 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.8824 | 0.1038 | **0.01 s** |
| **EXP-ML-01** | Traditional | Logistic Regression (Unweighted, $C=1.0$) | 0.6752 | 0.2386 | 0.0237 | 0.9980 | 0.6176 | 0.0457 | **0.8835** | 0.0984 | 3.87 s |
| **EXP-ML-02** | Traditional | Logistic Regression (Balanced, $C=0.1$) | 0.6765 | 0.2369 | 0.5618 | 0.6797 | 0.1895 | 0.2834 | 0.6658 | 0.2236 | 5.01 s |
| **EXP-ML-03** | Traditional | Random Forest (Balanced, depth=12) | 0.6780 | 0.2323 | 0.5042 | 0.7337 | 0.2015 | 0.2880 | 0.7068 | 0.2085 | 7.11 s |
| **EXP-ML-04** | Traditional | HistGradientBoosting (Default, $\\eta=0.08$) | 0.6795 | 0.2549 | 0.0141 | 0.9991 | 0.6757 | 0.0277 | 0.8833 | **0.0976** | 5.59 s |
| **EXP-ML-05** | Traditional | **LightGBM (Cost-Sensitive, $w=7.8$)** | **0.6813** | **0.2565** | **0.5675** | 0.6880 | 0.1951 | **0.2904** | 0.6738 | 0.2110 | **0.84 s** |
| **EXP-DL-01** | Deep Learning | Baseline Sequential MLP (64-32, $\\eta=10^{-3}$) | 0.6820 | 0.2490 | 0.0169 | 0.9990 | 0.6977 | 0.0331 | 0.8835 | 0.0983 | 18.67 s |
| **EXP-DL-02** | Deep Learning | Deep Regularized MLP (Functional, L2) | 0.6804 | 0.2444 | 0.0124 | 0.9994 | **0.7333** | 0.0244 | 0.8833 | 0.0981 | 44.54 s |
| **EXP-DL-03** | Deep Learning | Wide MLP (128-32, $\\eta=5 \\times 10^{-4}$) | 0.6816 | 0.2497 | 0.0158 | 0.9992 | 0.7179 | 0.0309 | 0.8835 | 0.0980 | 26.09 s |
| **EXP-DL-04** | Deep Learning | **Cost-Sensitive Deep MLP ($w=7.86$)** | **0.6842** | **0.2513** | **0.5551** | 0.6962 | 0.1958 | **0.2895** | 0.6796 | 0.1986 | 48.24 s |

### B. The Accuracy Paradox Under Severe Imbalance
The validation benchmark demonstrates the clinical Accuracy Paradox. Unweighted models (EXP-ML-01, EXP-ML-04, EXP-DL-01 to 03) achieve high nominal accuracy (88.33%–88.35%) through trivial majority-class prediction: the dummy baseline (EXP-ML-00) reaches 88.24% by predicting negative for all encounters. Clinical recall collapses to 1.24%–2.37% (EXP-DL-02 identified only 22 of 1,771 true readmissions). Conversely, cost-sensitive loss weighting ($w \\approx 7.8$) in EXP-ML-05 and EXP-DL-04 trades nominal accuracy (~67.4%) for clinical utility, surging recall to **56.75% for LightGBM** and **55.51% for the Deep MLP**, capturing over half of true readmissions.

### C. Learning Curves & Optimization Trajectories
Training trajectories across 50 epochs (`results/figures/exp-dl-01_learning_curves.png` to `exp-dl-04_learning_curves.png`) confirm effective regularization. EXP-DL-01 reached minimum validation loss (0.3472) at epoch 7 before early stopping at epoch 12. In EXP-DL-02 and EXP-DL-03, Batch Normalization and $L_2$ decay bounded generalization error, stabilizing validation ROC-AUC at 0.6804 and 0.6816. In EXP-DL-04, weighted binary cross-entropy declined from 1.05 to 0.98, reaching peak validation ROC-AUC (0.6842).

### D. Definitive Evaluation on Held-Out Test Cohort
The best traditional model (**EXP-ML-05: Cost-Sensitive LightGBM**) and deep model (**EXP-DL-04: Cost-Sensitive Deep MLP**) were evaluated on the untouched test partition ($N = 15,000$ encounters, 10,566 independent patients), summarized in Table II.

#### TABLE II: Definitive Held-Out Test Set Evaluation ($N = 15,000$ Encounters)
| Metric | Best Traditional Model (Cost-Sensitive LightGBM) | Best Deep Learning Model (Cost-Sensitive Deep MLP) | Delta ($\\Delta_{\\text{DL} - \\text{Trad}}$) |
| :--- | :---: | :---: | :---: |
| **Test ROC-AUC** | **0.6566** | **0.6574** | +0.0008 |
| **Test PR-AUC (Average Precision)** | **0.2040** | **0.2108** | +0.0068 |
| **Sensitivity (Recall)** | **54.17%** | **52.92%** | -1.25% |
| **Specificity** | **68.47%** | **69.21%** | +0.74% |
| **Positive Predictive Value (Precision)** | **17.81%** | **17.82%** | +0.01% |
| **F1-Score** | **0.2680** | **0.2666** | -0.0014 |
| **Overall Accuracy** | **66.87%** | **67.39%** | +0.52% |
| **Training Execution Latency** | **0.84 seconds** | **48.24 seconds** | **+57.4x slower** |

### E. Comparative Threshold Dynamics & Confusion Matrices
As illustrated in `results/figures/comparative_roc_pr_curves.png`, ROC trajectories track closely across thresholds (c-statistics ~0.657). In Precision-Recall space, both models achieve average precisions of 0.2040–0.2108, representing a two-fold enrichment over baseline prevalence ($\\pi_1 = 0.1120$).
Test confusion matrices (`results/figures/confusion_matrix_traditional_test.png` and `confusion_matrix_deep_learning_test.png`) over 15,000 test encounters (1,680 readmissions, 13,320 non-readmissions) record:
- **Cost-Sensitive LightGBM**: **897 true positives** (54.17% recall), 758 false negatives, 9,120 true negatives (specificity 68.47%), and 4,200 false positives.
- **Cost-Sensitive Deep MLP**: **876 true positives** (52.92% recall), 779 false negatives, 9,219 true negatives (specificity 69.21%), and 4,101 false positives.  
LightGBM detected 21 additional readmissions, while the Deep MLP produced 99 fewer false alarms.

---
"""

sec5 = """## V. Discussion, Clinical Error Analysis, & Limitations

### A. Clinical Impact of Asymmetric Classification Errors (FN vs. FP)
In clinical discharge planning, classification errors carry asymmetric consequences:
- **False Negative Harm**: Misclassifying an at-risk patient denies them transitional care, increasing the risk of acute diabetic ketoacidosis, severe hypoglycemia, or heart failure decompensation. The patient experiences avoidable morbidity, and the hospital incurs HRRP financial penalties.
- **False Positive Burden**: Flagging a stable patient triggers automated nurse phone outreach or clinic appointments, consuming modest operational resources (~$25–$50 per contact) with zero physical morbidity risk.

Unweighted models achieving 88.3% accuracy but 1.6% recall miss 98.4% of at-risk patients, offering zero operational utility. In contrast, cost-sensitive models sacrifice unhelpful majority-class accuracy (dropping to 67%) to capture over 53% of true readmissions. In an annual cohort of 10,000 inpatients, this corresponds to proactively intervening on over 500 vulnerable individuals who would otherwise deteriorate unmonitored.

### B. Algorithmic Trade-Offs: Tree Ensembles vs. Deep Neural Networks on Tabular EHR
Our experimental findings validate the empirical conclusions of Grinsztajn et al. (NeurIPS 2022) [11]: gradient boosted decision trees match deep neural networks on tabular clinical data while providing overwhelming operational advantages:
1. **Predictive Equivalence**: On the test partition, LightGBM achieved test ROC-AUC 0.6566 and PR-AUC 0.2040, while the Deep MLP achieved 0.6574 and 0.2108 (+0.0008 difference within statistical noise).
2. **Computational Efficiency**: LightGBM trained in **0.84 seconds** on a standard CPU, whereas the Deep MLP required **48.24 seconds**—a **57.4x computational penalty**—despite early stopping and `tf.data` prefetching. In production environments requiring frequent model retraining across departments, tree architectures provide substantial infrastructure savings.
3. **Engineering Overhead**: Deep architectures required extensive calibration (layer widths, dropout, batch normalization, learning rate decay schedules) to prevent divergence on sparse dummy variables. LightGBM converged reliably with default tree growth.
4. **Interpretability & Trust**: Decision tree ensembles interface directly with TreeSHAP algorithms (Lundberg & Lee [10]), computing exact Shapley values in polynomial time for point-of-care clinical explainability (e.g., attributing risk to prior emergency visits, insulin adjustments, or length of stay). Deep networks require expensive sampling approximations (KernelSHAP), introducing operational friction.

### C. Methodological Rigor: Entity Leakage and Patient-Grouped Partitioning
In our cohort of 100,114 encounters, 16,773 patients have multiple hospitalizations, with individuals admitted up to 40 times (~46% of records belong to repeat patients). A standard row split distributes earlier admissions into training and later admissions into testing. High-capacity models memorize patient biometric signatures, performing patient re-identification rather than learning clinical risk patterns. When deployed prospectively on unobserved patients, performance collapses [7]. Enforcing strict patient-level grouping with zero patient overlap across splits ensures reported test metrics (ROC-AUC 0.657, Recall 54%) reflect authentic, generalizable clinical performance.

### D. Study Limitations & Threats to Validity
Several limitations should be noted:
1. **Historical Dataset Era (1999–2008)**: Modern anti-diabetic pharmacotherapies (SGLT2 inhibitors, GLP-1 receptor agonists) were not standard of care during this period; contemporary readmission dynamics may differ.
2. **Missing Out-of-Hospital Mortality**: Patients who expire at home post-discharge are categorized as non-readmitted, creating competing risk distortion.
3. **Absence of Social Determinants of Health (SDOH)**: Socioeconomic factors (housing, income, social support) strongly influence readmission but are absent from billing records.
4. **Lack of Clinical Text**: Unstructured discharge summaries and progress notes contain nuanced functional status indicators absent from tabular ICD-9 codes.

---
"""

sec6 = """## VI. Conclusion & Future Work

### A. Summary of Findings
This study conducted a leakage-free comparative audit of classical machine learning and deep learning architectures for predicting 30-day diabetic hospital readmissions using the 101,766-record UCI 130-US Hospitals dataset. Our primary empirical conclusions are:
1. **Tabular Inductive Bias Preponderance**: Consistent with Grinsztajn et al. (2022) [11], tuned gradient boosted decision trees (LightGBM) achieved predictive discrimination identical to deep multilayer perceptrons (Test ROC-AUC: 0.6566 vs. 0.6574; PR-AUC: 0.2040 vs. 0.2108), while training over 57 times faster (0.84 s vs. 48.24 s).
2. **Imperative of Cost-Sensitive Objectives**: In clinical cohorts with 1:8 class imbalance, unweighted models achieve ~88% accuracy by predicting negative across 98% of cases. Cost-sensitive weighting ($w = 7.8$) re-aligns models with clinical utility, elevating sensitivity beyond 54% at acceptable false alarm rates.
3. **Methodological Independence via Patient Grouping**: Enforcing patient-grouped partitioning (`GroupShuffleSplit` on `patient_nbr`) guarantees zero entity data leakage across splits, ensuring reported performance metrics reflect authentic prospective generalizability.

### B. Recommendations for Enterprise Clinical Deployment
For health system engineering teams deploying operational risk stratification engines, **Cost-Sensitive LightGBM represents the superior production architecture**. It delivers state-of-the-art tabular discrimination with sub-second retraining, minimal latency, low compute cost, and native TreeSHAP interpretability for clinical point-of-care explainability.

### C. Future Research Directions
Future work will explore:
- **Sequential Longitudinal Architectures**: Modeling visit histories as temporal trajectories using RNNs/Transformers (e.g., BEHRT) to capture longitudinal disease progression.
- **Multimodal Integration**: Combining tabular diagnostic codes with unstructured clinical notes via fine-tuned ClinicalBERT models.
- **Prospective Shadow Deployment**: Integrating LightGBM into an operational EHR environment via HL7 FHIR APIs to evaluate impact on transitional nurse workflows.

---
"""

sec7 = """## VII. References (IEEE Format)

[1] B. Strack et al., "Impact of HbA1c measurement on hospital readmission rates: analysis of 70,000 clinical database records," *BioMed Res. Int.*, vol. 2014, Art. 781670, 2014. DOI: 10.1155/2014/781670.

[2] D. Kansagara et al., "Risk prediction models for hospital readmission: a systematic review," *JAMA*, vol. 306, no. 15, pp. 1688–1698, 2011. DOI: 10.1001/jama.2011.1515.

[3] B. Shickel et al., "Deep EHR: a survey of recent advances in deep learning techniques for electronic health record (EHR) analysis," *IEEE JBHI*, vol. 22, no. 5, pp. 1589–1604, 2018. DOI: 10.1109/JBHI.2017.2767063.

[4] A. Rajkomar et al., "Scalable and accurate deep learning for electronic health records," *npj Digit. Med.*, vol. 1, no. 1, Art. 18, 2018. DOI: 10.1038/s41746-018-0029-1.

[5] T. Saito and M. Rehmsmeier, "The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets," *PLOS ONE*, vol. 10, no. 3, Art. e0118432, 2015. DOI: 10.1371/journal.pone.0118432.

[6] G. Ke et al., "LightGBM: A highly efficient gradient boosting decision tree," in *Proc. NeurIPS*, vol. 30, pp. 3146–3154, 2017.

[7] S. Kaufman et al., "Leakage in data mining: Formulation, detection, and avoidance," *ACM TKDD*, vol. 6, no. 4, Art. 15, 2012. DOI: 10.1145/2382577.2382579.

[8] D. J. Rubin, "Hospital readmission of patients with diabetes," *Curr. Diab. Rep.*, vol. 15, no. 4, Art. 17, 2015. DOI: 10.1007/s11892-015-0584-7.

[9] H. He and E. A. Garcia, "Learning from imbalanced data," *IEEE TKDE*, vol. 21, no. 9, pp. 1263–1284, 2009. DOI: 10.1109/TKDE.2008.239.

[10] S. M. Lundberg and S.-I. Lee, "A unified approach to interpreting model predictions," in *Proc. NeurIPS*, vol. 30, pp. 4765–4774, 2017.

[11] L. Grinsztajn, E. Oyallon, and G. Varoquaux, "Why do tree-based models still outperform deep learning on typical tabular data?," in *Proc. NeurIPS*, vol. 35, pp. 507–520, 2022.

---
"""

appendix = """## Appendix: 5–10 Minute Oral Presentation Script & Slide Outline

| Slide # | Slide Title | Timing | Speaking Points & Visual Cues | Anticipated Technical Defense Questions |
| :---: | :--- | :---: | :--- | :--- |
| **1** | Title & Mission Alignment | 0:00 – 1:00 | Introduce problem, university assignment, and healthcare engineering mission. 30-day readmissions cost billions and harm diabetic patients. | *"Why an operational tool rather than a diagnostic one?"* $\\rightarrow$ Model estimates post-discharge operational readmission risk, not diabetes presence. |
| **2** | Dataset & Verification | 1:00 – 2:30 | Show raw stats: 101,766 rows, 71,518 patients. Explain the 69,984 discrepancy (clinical exclusions of expired/hospice patients). Emphasize 11.3% class imbalance. | *"Why exclude expired patients?"* $\\rightarrow$ 100% had label 'NO'; dead patients cannot be readmitted. Retaining them creates artificial leakage. |
| **3** | Leakage Prevention & Split | 2:30 – 4:00 | Explain Strategy B (`GroupShuffleSplit`). Show that 16,773 repeat patients cause leakage under row splits. Prove 0% patient overlap. | *"Why not random StratifiedKFold?"* $\\rightarrow$ Random splitting leaks patient idiosyncrasies into test sets, violating independence. |
| **4** | Classical ML vs. Deep Learning | 4:00 – 5:30 | Explain 10 experiments: 6 classical ML (LightGBM, Random Forest, Logistic Reg) and 4 Keras models with `tf.data`. Explain 7.8x cost-sensitive weighting. | *"Why tf.data in Keras?"* $\\rightarrow$ Enables batched streaming, autotuned prefetching, and memory scalability for large clinical cohorts. |
| **5** | Results & Learning Curves | 5:30 – 7:30 | Present Master Table. Show comparative ROC/PR curves and test evaluation table (ROC-AUC ~0.657, Recall ~54%). | *"Why is test accuracy 67% lower than baseline 88%?"* $\\rightarrow$ Cost-sensitive weighting trades unhelpful majority accuracy for 54% clinical recall. |
| **6** | Clinical Errors & Limitations | 7:30 – 9:00 | Analyze False Negatives vs. False Positives. Discuss why LightGBM matched Deep Learning (Grinsztajn et al. 2022 tabular inductive bias). Note dataset era limits. | *"Why didn't deep learning outperform LightGBM?"* $\\rightarrow$ Tabular clinical data lacks spatial invariance; decision trees excel on discrete boundaries. |
| **7** | Conclusion & Deliverables | 9:00 – 10:00 | Recap deliverables: Colab notebook, clean repository, joblib & .keras model artifacts in Google Drive. Thank faculty and open for questions. | Confident closure demonstrating full technical mastery of code and empirical results. |
"""

def apply_clean_reductions(text):
    # Perform clean prose tightenings
    replacements = [
        # Abstract
        ("Hospital readmission within 30 days of discharge is a vital clinical quality indicator and financial vulnerability under the CMS Hospital Readmissions Reduction Program (HRRP).",
         "Hospital readmission within 30 days is a critical clinical quality indicator and financial liability under the CMS Hospital Readmissions Reduction Program (HRRP)."),
        ("This investigation presents an empirical comparative evaluation of classical machine learning algorithms and deep neural networks for predicting 30-day all-cause hospital readmission using the landmark UCI Diabetes 130-US Hospitals dataset (1999–2008).",
         "This investigation presents an empirical comparative evaluation of classical machine learning and deep neural networks for predicting 30-day all-cause readmission using the UCI Diabetes 130-US Hospitals dataset (1999–2008)."),
        ("six classical algorithms (Dummy baseline, unweighted/balanced Logistic Regression, Random Forest, HistGradientBoosting, Cost-Sensitive LightGBM) and four deep architectures in TensorFlow/Keras with streaming `tf.data` pipelines (Sequential Baseline, Functional Regularized, Wide, and Cost-Sensitive Deep MLP).",
         "six classical models (Dummy baseline, unweighted/balanced Logistic Regression, Random Forest, HistGradientBoosting, Cost-Sensitive LightGBM) and four deep neural networks in TensorFlow/Keras with streaming `tf.data` pipelines (Sequential, Functional Regularized, Wide, and Cost-Sensitive Deep MLP)."),
        
        # Section I
        ("Hospitalized diabetic inpatients frequently present with acute glycemic emergencies or complex cardiovascular and renal comorbidities. Post-discharge transitions create substantial clinical vulnerability, yielding 30-day readmission rates between 15% and 25%. Under the CMS Hospital Readmissions Reduction Program (HRRP), hospitals face financial penalties up to 3% of Medicare reimbursements for excess readmissions.",
         "Hospitalized diabetic inpatients face elevated post-discharge vulnerability from acute glycemic volatility and cardiovascular comorbidities, resulting in 30-day readmission rates between 15% and 25%. Under the CMS Hospital Readmissions Reduction Program (HRRP), hospitals face financial penalties up to 3% of Medicare reimbursements for excess readmissions."),
        ("Flagging high-risk patients enables targeted transitional interventions, including nurse phone outreach, medication reconciliation, and scheduled 7-day clinic follow-ups.",
         "Flagging high-risk patients enables targeted transitional interventions: nurse phone outreach, medication reconciliation, and scheduled 7-day follow-ups."),
        
        # Section II
        ("In a systematic review in *JAMA*, Kansagara et al. [2] examined 30 readmission models, finding administrative EHR discrimination tops out between ROC-AUC 0.60–0.70. This ceiling reflects unmeasured social determinants of health, health literacy, and outpatient treatment compliance. Consequently, administrative EHR models achieving ROC-AUC 0.65–0.70 represent solid, state-of-the-art discrimination under real-world operational constraints.",
         "In a systematic *JAMA* review, Kansagara et al. [2] examined 30 readmission models, finding administrative EHR models top out between ROC-AUC 0.60–0.70 due to unmeasured social determinants, health literacy, and post-discharge compliance. Thus, models achieving ROC-AUC 0.65–0.70 represent solid discrimination under real-world operational constraints."),
        ("demonstrating that inpatient HbA1c testing and medication adjustments correlate with lower readmission rates, though modeling only single encounters with logistic regression.",
         "showing inpatient HbA1c testing and medication adjustments correlate with lower readmissions, though analyzing only single encounters via logistic regression."),
        ("For deep clinical modeling, Shickel et al. (2018) [3] surveyed deep architectures for EHR analysis. Rajkomar et al. (2018) [4] demonstrated deep learning on multi-center EHRs (ROC-AUC 0.75–0.76) using temporal sequences of clinical tokens.",
         "For deep clinical modeling, Shickel et al. (2018) [3] surveyed deep EHR architectures, and Rajkomar et al. (2018) [4] demonstrated deep learning on multi-center EHRs (ROC-AUC 0.75–0.76) using temporal token sequences."),
        
        # Section III
        ("The prediction point is anchored at hospital discharge. Inpatient diagnoses, lab tests, length of stay, 12-month historical utilization, and discharge medication adjustments are observed, while post-discharge events are strictly masked. Flagged high-risk patients are routed to transitional care protocols (nurse outreach, medication reconciliation, and scheduled 7-day follow-up).",
         "The prediction point is anchored at hospital discharge. Diagnoses, lab results, length of stay, 12-month utilization, and discharge medication changes are observed, while post-discharge events are strictly masked. High-risk patients are routed to transitional care (nurse outreach, medication reconciliation, 7-day follow-up)."),
        ("A `ColumnTransformer` fitted strictly on training data handles feature transforms:\n1. **Removals**: IDs (`encounter_id`, `patient_nbr`), zero-variance features (`examide`, `citoglipton`), and sparse fields (`weight` 96.86%, `payer_code` 39.56% missing) were removed.\n2. **ICD-9 Grouping**: Diagnoses mapped to 9 clinical categories [1]: Circulatory (390–459, 785), Respiratory (460–519, 786), Digestive (520–579, 787), Diabetes (250.xx), Injury (800–999), Musculoskeletal (710–739), Genitourinary (580–629, 788), Neoplasms (140–239), and Other.\n3. **Encoding**: Continuous features were median-imputed and standardized; categoricals were imputed with `'Missing'` and one-hot encoded, yielding **206 numerical features**.",
         "A `ColumnTransformer` fitted strictly on training data handles feature transforms:\n1. **Removals**: IDs (`encounter_id`, `patient_nbr`), constant features (`examide`, `citoglipton`), and sparse fields (`weight` 96.86%, `payer_code` 39.56% missing) were removed.\n2. **ICD-9 Grouping**: Diagnoses mapped to 9 clinical categories [1]: Circulatory (390–459, 785), Respiratory (460–519, 786), Digestive (520–579, 787), Diabetes (250.xx), Injury (800–999), Musculoskeletal (710–739), Genitourinary (580–629, 788), Neoplasms (140–239), and Other.\n3. **Encoding**: Continuous features were median-imputed and standardized; categoricals were imputed with `'Missing'` and one-hot encoded, yielding **206 numerical features**."),
        
        # Section V
        ("In clinical discharge planning, classification errors carry asymmetric consequences:\n- **False Negative Harm**: Misclassifying an at-risk patient denies them transitional care, increasing the risk of acute diabetic ketoacidosis, severe hypoglycemia, or heart failure decompensation. The patient experiences avoidable morbidity, and the hospital incurs HRRP financial penalties.\n- **False Positive Burden**: Flagging a stable patient triggers automated nurse phone outreach or clinic appointments, consuming modest operational resources (~$25–$50 per contact) with zero physical morbidity risk.",
         "In clinical discharge planning, classification errors carry asymmetric costs:\n- **False Negative Harm**: Misclassifying an at-risk patient denies them transitional care, risking acute diabetic ketoacidosis, severe hypoglycemia, or heart failure decompensation. The patient suffers avoidable morbidity, and the hospital incurs HRRP penalties.\n- **False Positive Burden**: Flagging a stable patient triggers automated nurse phone outreach or clinic check-ins, consuming modest operational resources (~$25–$50 per contact) with zero physical harm."),
        ("Unweighted models achieving 88.3% accuracy but 1.6% recall miss 98.4% of at-risk patients, offering zero operational utility. In contrast, cost-sensitive models sacrifice unhelpful majority-class accuracy (dropping to 67%) to capture over 53% of true readmissions. In an annual cohort of 10,000 inpatients, this corresponds to proactively intervening on over 500 vulnerable individuals who would otherwise deteriorate unmonitored.",
         "Unweighted models achieving 88.3% accuracy but 1.6% recall miss 98.4% of at-risk patients, offering zero operational utility. In contrast, cost-sensitive models sacrifice unhelpful majority-class accuracy (dropping to 67%) to capture over 53% of true readmissions. In a cohort of 10,000 inpatients, this corresponds to proactively intervening on over 500 vulnerable individuals who would otherwise deteriorate unmonitored."),
        ("Our experimental findings validate the empirical conclusions of Grinsztajn et al. (NeurIPS 2022) [11]: gradient boosted decision trees match deep neural networks on tabular clinical data while providing overwhelming operational advantages:",
         "Our findings validate the empirical conclusions of Grinsztajn et al. (NeurIPS 2022) [11]: gradient boosted trees match deep neural networks on tabular clinical data while providing decisive operational advantages:"),
        ("In our cohort of 100,114 encounters, 16,773 patients have multiple hospitalizations, with individuals admitted up to 40 times (~46% of records belong to repeat patients). A standard row split distributes earlier admissions into training and later admissions into testing. High-capacity models memorize patient biometric signatures, performing patient re-identification rather than learning clinical risk patterns. When deployed prospectively on unobserved patients, performance collapses [7]. Enforcing strict patient-level grouping with zero patient overlap across splits ensures reported test metrics (ROC-AUC 0.657, Recall 54%) reflect authentic, generalizable clinical performance.",
         "In our cohort of 100,114 encounters, 16,773 patients have multiple hospitalizations, up to 40 times (~46% of records belong to repeat patients). A standard row split distributes earlier admissions into training and later admissions into testing. High-capacity models memorize patient biometric signatures, performing re-identification rather than learning general risk patterns. When deployed prospectively, performance collapses [7]. Enforcing patient-grouped partitioning with zero patient overlap across splits ensures reported test metrics (ROC-AUC 0.657, Recall 54%) reflect authentic, generalizable clinical performance."),
         
        # Appendix
        ("Introduce problem, university assignment, and healthcare engineering mission. 30-day readmissions cost billions and harm diabetic patients.",
         "Introduce problem, assignment, and healthcare engineering mission. 30-day readmissions cost billions and harm diabetic patients."),
        ("Show raw stats: 101,766 rows, 71,518 patients. Explain the 69,984 discrepancy (clinical exclusions of expired/hospice patients). Emphasize 11.3% class imbalance.",
         "Show raw stats: 101,766 rows, 71,518 patients. Explain the 69,984 discrepancy (clinical exclusions of expired/hospice). Emphasize 11.3% class imbalance."),
        ("Explain Strategy B (`GroupShuffleSplit`). Show that 16,773 repeat patients cause leakage under row splits. Prove 0% patient overlap.",
         "Explain Strategy B (`GroupShuffleSplit`). Show that 16,773 repeat patients cause leakage under row splits. Confirm 0% patient overlap."),
        ("Explain 10 experiments: 6 classical ML (LightGBM, Random Forest, Logistic Reg) and 4 Keras models with `tf.data`. Explain 7.8x cost-sensitive weighting.",
         "Explain 10 experiments: 6 classical ML (LightGBM, Random Forest, Logistic Reg) and 4 Keras models with `tf.data`. Explain 7.8x loss weighting."),
        ("Present Master Table. Show comparative ROC/PR curves and test evaluation table (ROC-AUC ~0.657, Recall ~54%).",
         "Present Master Table. Show comparative ROC/PR curves and test table (ROC-AUC ~0.657, Recall ~54%)."),
        ("Analyze False Negatives vs. False Positives. Discuss why LightGBM matched Deep Learning (Grinsztajn et al. 2022 tabular inductive bias). Note dataset era limits.",
         "Analyze False Negatives vs. False Positives. Discuss why LightGBM matched Deep Learning (Grinsztajn et al. 2022 tabular inductive bias). Note limitations."),
        ("Recap deliverables: Colab notebook, clean repository, joblib & .keras model artifacts in Google Drive. Thank faculty and open for questions.",
         "Recap deliverables: Colab notebook, repository, joblib & .keras model artifacts in Google Drive. Thank faculty and open for questions.")
    ]
    for old, new in replacements:
        text = text.replace(old, new)
    return text

sections = [
    ("header", header),
    ("abstract", abstract),
    ("sec1", sec1),
    ("sec2", sec2),
    ("sec3", sec3),
    ("sec4", sec4),
    ("sec5", sec5),
    ("sec6", sec6),
    ("sec7", sec7),
    ("appendix", appendix)
]

raw_full = "".join([s for _, s in sections])
final_text = apply_clean_reductions(raw_full)

words = len(re.findall(r'\b[A-Za-z0-9_-]+\b', final_text))
print(f"Final total words: {words}")

with open("report/report_outline.md", "w", encoding="utf-8") as f:
    f.write(final_text)

print("Saved to report/report_outline.md")
