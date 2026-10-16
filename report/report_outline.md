# Academic Report & Presentation Guide: IEEE Structure & Evidence Base

**Project Title**: Comparing Traditional Machine Learning and Deep Learning for Predicting 30-Day Hospital Readmission Among Patients with Diabetes  
**Author**: Software Engineering Student  
**Target Venue / Style**: University Summative Project Report in IEEE Conference / Transactions Format  
**Word Count Target**: 3,500 – 5,000 words (excluding references, figure captions, and appendices)  
**Submission Links Required on First Page Header**:
- **GitHub Repository**: [https://github.com/Amazing-beep/diabetes-readmission-ml-dl](https://github.com/Amazing-beep/diabetes-readmission-ml-dl)
- **Google Drive Submission Folder**: [https://drive.google.com/drive/folders/1LoVTMlsopfQ8C3oAqFJ3xGavl8eiNfFD?usp=drive_link](https://drive.google.com/drive/folders/1LoVTMlsopfQ8C3oAqFJ3xGavl8eiNfFD?usp=drive_link)
- **Presentation Video Link**: *(Insert your recorded video link here)*

---

## Academic Writing Guidelines & Integrity Notice
> [!IMPORTANT]
> **University Rubric Constraint (<20% AI Detection)**:
> This outline provides the **exact empirical tables, verified mathematical formulations, genuine literature citations, and structural section headers**. 
> To satisfy the rubric's requirement that your report reflects an authentic student voice with less than 20% AI-generated text, **do not submit boilerplate prose verbatim**. Use the guided reflection prompts under each section to draft your own narrative, synthesizing your personal understanding of software engineering and healthcare.

---

## Section-by-Section IEEE Report Structure

### Title & Header Metadata
- **Title**: *Comparative Evaluation of Classical Machine Learning and Deep Learning Architectures for Retrospective 30-Day Hospital Readmission Prediction in Diabetic Inpatients*
- **Author Affiliation**: Department of Software Engineering / Computer Science.
- **Header Links**: Include working links to your GitHub repository, Google Drive folder, and presentation video prominently beneath the abstract.

---

### Abstract (150–250 words)
- **Context**: 30-day readmissions in diabetes impose substantial clinical and economic strain, incentivizing CMS Hospital Readmissions Reduction Program (HRRP) penalty avoidance.
- **Problem**: Severe class imbalance (~1:8), patient encounter multiplicity, and high-cardinality EHR data hinder accurate post-discharge risk stratification.
- **Methodology**: Evaluated across 100,114 eligible inpatient encounters from the UCI 130-US Hospitals dataset (1999–2008). Enforced patient-grouped splitting (`GroupShuffleSplit` on 70,439 unique patients) with 0% patient leakage. Implemented and benchmarked 6 traditional ML models (Scikit-Learn, LightGBM) and 4 deep neural network architectures using TensorFlow/Keras with `tf.data` input pipelines and cost-sensitive loss weighting.
- **Results**: Best traditional model (LightGBM Cost-Sensitive) achieved test ROC-AUC of 0.6566, PR-AUC of 0.2040, and 54.17% recall. Best deep learning model (Cost-Sensitive MLP) achieved test ROC-AUC of 0.6574, PR-AUC of 0.2108, and 52.92% recall.
- **Conclusion**: Tabular gradient boosted decision trees achieve equivalent predictive discrimination to deep architectures with >40x training efficiency and superior clinical interpretability.

---

### I. Introduction (approx. 600 words)
#### Core Argument & Mission Alignment
1. **Clinical Background**: Diabetes mellitus affects over 37 million Americans. Inpatients with diabetes face readmission rates exceeding 20%, driven by glycemic instability, microvascular/macrovascular complications, and care fragmentation during hospital discharge.
2. **Economic & Policy Imperatives**: The Patient Protection and Affordable Care Act established the HRRP, placing up to 3% of total Medicare reimbursements at risk for excess readmissions.
3. **Software Engineering & ML Problem Statement**: Retrospective Electronic Health Records (EHR) present unique challenges: categorical sparsity, high missingness, encounter clustering (multiple hospitalizations per patient), and severe outcome imbalance (~11.3% 30-day readmissions).
4. **Research Objectives**:
   - Formulate a leakage-free, patient-grouped risk stratification pipeline.
   - Benchmark traditional ML algorithms against deep neural networks leveraging `tf.data`.
   - Quantify the clinical trade-offs between unweighted accuracy and cost-sensitive recall.

*Guided Writing Prompts for Your Voice*:
- *Reflect on your motivation as a software engineering student: why is reproducibility in clinical AI more crucial than in traditional consumer software?*
- *Explicitly declare the operational boundary: explain that this pipeline assists clinical discharge planning for transitional care follow-up, rather than diagnosing diabetes or claiming causal etiology.*

---

### II. Literature Review & Theoretical Grounding (approx. 800 words)
#### Critical Synthesis of 11 Verified Scholarly Sources

| Citation Key | Formal IEEE Citation | Critical Synthesis & Relevance to Project |
| :--- | :--- | :--- |
| **[1] Strack2014** | B. Strack, J. P. DeShazo, C. Gennings, J. L. Olmo, S. Ventura, K. J. Cios, and J. N. Clore, "Impact of HbA1c measurement on hospital readmission rates: analysis of 70,000 clinical database records," *BioMed Res. Int.*, vol. 2014, Art. no. 781670, 2014. DOI: 10.1155/2014/781670. | Benchmark dataset paper. Established the 130-US hospitals cohort and showed that HbA1c testing correlates with lower readmissions, but relied on basic odds ratios without modern gradient boosting or deep learning. |
| **[2] Kansagara2011** | D. Kansagara *et al.*, "Risk prediction models for hospital readmission: a systematic review," *JAMA*, vol. 306, no. 15, pp. 1688–1698, 2011. DOI: 10.1001/jama.2011.1515. | Systematically reviewed 30 readmission models (LACE, HOSPITAL). Found most retrospective administrative models achieve modest discrimination (c-statistic 0.60–0.70), framing our empirical baseline expectations. |
| **[3] Shickel2018** | B. Shickel, P. J. Tighe, A. Bihorac, and P. Rashidi, "Deep EHR: a survey of recent advances in deep learning techniques for electronic health record (EHR) analysis," *IEEE J. Biomed. Health Inform.*, vol. 22, no. 5, pp. 1589–1604, 2018. DOI: 10.1109/JBHI.2017.2767063. | Comprehensive survey establishing the theoretical foundation for applying feedforward and recurrent neural networks to heterogeneous EHR features. |
| **[4] Rajkomar2018** | A. Rajkomar *et al.*, "Scalable and accurate deep learning for electronic health records," *npj Digit. Med.*, vol. 1, no. 1, Art. no. 18, 2018. DOI: 10.1038/s41746-018-0029-1. | Demonstrated deep learning across massive EHR systems for 30-day readmissions (AUC 0.75–0.76), highlighting the importance of end-to-end representations over manual feature curation. |
| **[5] Saito2015** | T. Saito and M. Rehmsmeier, "The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets," *PLOS ONE*, vol. 10, no. 3, Art. no. e0118432, 2015. DOI: 10.1371/journal.pone.0118432. | Mathematical justification for prioritizing Precision-Recall curves (PR-AUC) over ROC-AUC when class prevalence is ~11%, preventing false optimism from large true negative volumes. |
| **[6] Ke2017** | G. Ke *et al.*, "LightGBM: A highly efficient gradient boosting decision tree," in *Proc. Adv. Neural Inf. Process. Syst. (NeurIPS)*, vol. 30, pp. 3146–3154, 2017. | Introduced Gradient-Based One-Side Sampling (GOSS) and Exclusive Feature Bundling (EFB), explaining why LightGBM trains in <1 second while maintaining state-of-the-art tabular accuracy. |
| **[7] Kaufman2012** | S. Kaufman, S. Rosset, C. Perlich, and O. Stitelman, "Leakage in data mining: Formulation, detection, and avoidance," *ACM Trans. Knowl. Discov. Data*, vol. 6, no. 4, pp. 1–21, 2012. DOI: 10.1145/2382577.2382579. | Foundational paper on data leakage. Forms the basis for our patient-grouped splitting strategy to prevent repeated encounter contamination across train and test sets. |
| **[8] Rubin2015** | D. J. Rubin, "Hospital readmission of patients with diabetes," *Curr. Diab. Rep.*, vol. 15, no. 4, Art. no. 17, 2015. DOI: 10.1007/s11892-015-0584-7. | Reviews clinical predictors of diabetes readmission: prior emergency visits, insulin complexity, comorbidity count, and discharge support. |
| **[9] He2009** | H. He and E. A. Garcia, "Learning from imbalanced data," *IEEE Trans. Knowl. Data Eng.*, vol. 21, no. 9, pp. 1263–1284, 2009. DOI: 10.1109/TKDE.2008.239. | Theoretical justification for cost-sensitive learning and class weighting over naive accuracy metrics in skewed data regimes. |
| **[10] Lundberg2017** | S. M. Lundberg and S.-I. Lee, "A unified approach to interpreting model predictions," in *Proc. Adv. Neural Inf. Process. Syst. (NeurIPS)*, vol. 30, pp. 4765–4774, 2017. | Explains the necessity of feature attribution (SHAP values) for clinical transparency and trust in healthcare deployments. |
| **[11] Grinsztajn2022** | L. Grinsztajn, E. Oyallon, and G. Varoquaux, "Why do tree-based models still outperform deep learning on typical tabular data?," in *Proc. Adv. Neural Inf. Process. Syst. (NeurIPS)*, vol. 35, pp. 507–520, 2022. | Benchmark study proving tree-based models (LightGBM, XGBoost) consistently outperform deep architectures on tabular tabular data with unnormalized distributions and discrete features. |

---

### III. Methodology (approx. 1,000 words)

#### A. Prediction Scenario & Temporal Boundaries
- **Time of Prediction**: Inpatient discharge planning (day of anticipated discharge).
- **Intended Intervention**: Enrolling high-risk patients in transitional nurse phone calls, post-discharge medication reconciliation, and scheduled 7-day primary care follow-up.
- **Available Predictors**: Demographics, admission source/type, ICD-9 primary/secondary diagnoses, lab order flags (HbA1c, glucose serum), inpatient length of stay, medication adjustments, and 12-month historical utilization (outpatient, emergency, inpatient visits).

#### B. Clinical Exclusions & Leakage Prevention
- **In-Hospital Deaths**: Patients with discharge disposition IDs 11, 19, 20, 21 ($N = 1,652$) were excluded. 100% of these patients had `readmitted == 'NO'`. Keeping them creates artificial bias since deceased patients cannot be readmitted.
- **Feature Exclusion Table**:

| Feature Name | Action | Methodological Justification |
| :--- | :--- | :--- |
| `encounter_id`, `patient_nbr` | Dropped | Administrative identifiers; non-generalizable patient memorization leakage. |
| `weight` | Dropped | 96.86% missingness; imputation would introduce synthetic noise. |
| `payer_code` | Dropped | Administrative billing code; 39.56% missingness; lacks clinical mechanism. |
| `examide`, `citoglipton` | Dropped | Constant zero-variance features (100% 'No' across all encounters). |
| `readmitted`, `target` | Separated | Target labels. |
| `diag_1`, `diag_2`, `diag_3` | Transformed | Grouped into 9 broad ICD-9 diagnostic clusters following Strack et al. (2014). |
| `discharge_disposition_id` | Cleaned / Encoded | Expired patients excluded; remaining valid categories encoded as one-hot tokens. |

#### C. Patient-Grouped Splitting Strategy
- **GroupShuffleSplit**: 70% Train, 15% Validation, 15% Test grouped strictly by `patient_nbr`.
- **Validation Audit**:
  - Train: 70,055 encounters (49,305 unique patients), 11.29% positive.
  - Validation: 15,059 encounters (10,568 unique patients), 11.76% positive.
  - Test: 15,000 encounters (10,566 unique patients), 11.20% positive.
  - Patient Overlap: **Exactly 0 patients shared across splits**.

#### D. Modeling Frameworks
1. **Classical Machine Learning**: DummyClassifier, Logistic Regression (unweighted & balanced), Random Forest, HistGradientBoosting, LightGBM (cost-sensitive).
2. **Deep Learning with `tf.data`**:
   - `make_tf_dataset`: Prefetched, batched pipelines with training shuffle buffer ($10,000$).
   - Baseline Sequential MLP (64 $\rightarrow$ 32, Dropout 0.2).
   - Functional Deep Regularized MLP (128 $\rightarrow$ 64 $\rightarrow$ 32, BatchNorm, Dropout 0.3, L2 $10^{-4}$).
   - Wide MLP (128 $\rightarrow$ 32, lr=$5 \times 10^{-4}$, batch size 512).
   - Cost-Sensitive Deep MLP (balanced class weight penalty of 7.8x on positive class).
   - Training callbacks: `EarlyStopping` (patience=5, restore best weights) and `ReduceLROnPlateau`.

---

### IV. Results (approx. 1,000 words)

#### A. Master Experimental Validation Benchmarking (10 Runs)
All 10 experimental configurations logged to `results/experiment_results.csv` and W&B offline are summarized below:

| Experiment ID | Model Family | Algorithm / Configuration | Val ROC-AUC | Val PR-AUC | Val Recall | Val F1 | Training Time (s) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **EXP-ML-00** | Traditional | Dummy Baseline (Strategy='prior') | 0.5000 | 0.1176 | 0.0000 | 0.0000 | 0.01 |
| **EXP-ML-01** | Traditional | Logistic Regression (Unweighted, C=1.0) | 0.6752 | 0.2386 | 0.0237 | 0.0457 | 3.87 |
| **EXP-ML-02** | Traditional | Logistic Regression (Balanced, C=0.1) | 0.6765 | 0.2369 | 0.5618 | 0.2834 | 5.01 |
| **EXP-ML-03** | Traditional | Random Forest (Balanced, max_depth=12) | 0.6780 | 0.2323 | 0.5042 | 0.2880 | 7.11 |
| **EXP-ML-04** | Traditional | HistGradientBoosting (Default, lr=0.08) | 0.6795 | 0.2549 | 0.0141 | 0.0277 | 5.59 |
| **EXP-ML-05** | Traditional | **LightGBM (Cost-Sensitive, scale_pos=7.8)** | **0.6813** | **0.2565** | **0.5675** | **0.2904** | **0.84** |
| **EXP-DL-01** | Deep Learning | Baseline MLP (Sequential, 64-32, lr=1e-3) | 0.6820 | 0.2490 | 0.0169 | 0.0331 | 18.67 |
| **EXP-DL-02** | Deep Learning | Deep Regularized MLP (Functional, 128-64-32) | 0.6804 | 0.2444 | 0.0124 | 0.0244 | 44.54 |
| **EXP-DL-03** | Deep Learning | Wide MLP (128-32, lr=5e-4, batch=512) | 0.6816 | 0.2497 | 0.0158 | 0.0309 | 26.09 |
| **EXP-DL-04** | Deep Learning | **Cost-Sensitive Deep MLP (weight=7.8)** | **0.6842** | **0.2513** | **0.5551** | **0.2895** | **48.24** |

#### B. Final Evaluation on Untouched Held-Out Test Set
Selected based strictly on validation ROC-AUC / PR-AUC and evaluated on the untouched test partition ($N = 15,000$ encounters, $10,566$ independent patients):

| Metric | Traditional ML (LightGBM Cost-Sensitive) | Deep Learning (Cost-Sensitive Deep MLP) | Clinical & Statistical Meaning |
| :--- | :---: | :---: | :--- |
| **ROC-AUC** | **0.6566** | **0.6574** | Overall ranking discrimination across all classification thresholds. |
| **PR-AUC (Avg Prec)** | **0.2040** | **0.2108** | Precision-recall area (baseline prevalence = 0.1120). |
| **Sensitivity (Recall)**| **54.17%** | **52.92%** | Proportion of true 30-day readmissions correctly flagged for intervention. |
| **Specificity** | **68.47%** | **69.21%** | Proportion of stable patients correctly identified as low risk. |
| **Precision (PPV)** | **17.81%** | **17.82%** | Positive predictive value in clinical triage. |
| **F1-Score** | **0.2680** | **0.2666** | Harmonic mean of precision and recall under cost-sensitive adjustment. |
| **Accuracy** | **66.87%** | **67.39%** | Overall correct rate under cost-sensitive operational threshold. |
| **Training Time** | **0.84 seconds** | **48.24 seconds** | LightGBM trained >57x faster than the deep neural network. |

#### C. Figure Captions & Visual Integration
- **Fig. 1**: `results/figures/comparative_roc_pr_curves.png` — Comparative ROC and Precision-Recall curves across validation experiments. Shows cost-sensitive models tracking closely with moderate discriminative power significantly above chance.
- **Fig. 2**: `results/figures/confusion_matrix_traditional_test.png` & `results/figures/confusion_matrix_deep_learning_test.png` — Test confusion matrices demonstrating ~54% true positive capture with controlled false alarm rates.
- **Fig. 3**: `results/figures/exp-dl-01_learning_curves.png` to `exp-dl-04_learning_curves.png` — Training vs. validation loss and AUC trajectories, demonstrating that EarlyStopping and ReduceLROnPlateau successfully halted training before validation degradation.

---

### V. Discussion & Error Analysis (approx. 1,000 words)

#### A. Clinical Impact of Misclassifications (FN vs. FP)
- **False Negative (FN) Harm**: When a high-risk diabetic patient is misclassified as low-risk, they receive standard post-discharge care without follow-up coordination. For patients with complex medication regimens or prior inpatient admissions, this directly increases the risk of diabetic ketoacidosis (DKA), severe hypoglycemia, or acute heart failure decompensation.
- **False Positive (FP) Burden**: Classifying a stable patient as high-risk triggers automated nurse phone calls or expedited primary care clinic appointments. While consuming administrative hours, it poses **zero patient morbidity risk**.
- **Conclusion**: A cost-sensitive threshold capturing >53% of readmissions is vastly superior in clinical practice to an unweighted model with 88% accuracy but 1.6% recall.

#### B. Algorithmic Trade-Offs: Tree Ensembles vs. Deep Architectures
- Supporting the findings of Grinsztajn et al. (2022) [11], LightGBM matched the predictive capacity of deep neural networks on tabular clinical EHR data (Test ROC-AUC: 0.6566 vs. 0.6574) while training in 0.84 seconds versus 48.24 seconds.
- Deep neural networks require extensive hyperparameter tuning (batch normalization, learning rate decay, L2 regularization) to prevent rapid overfitting on sparse one-hot dummy variables, whereas gradient boosted trees natively handle tabular thresholds with minimal parameter sensitivity.

#### C. Methodological Rigor & Data Leakage Prevention
- Explaining the necessity of `GroupShuffleSplit`: without patient grouping, repeated admissions from the 16,773 repeat patients would leak across splits, inflating apparent performance by allowing models to memorize patient IDs.
- Demonstrating the clinical justification for excluding expired patients: eliminating the artificial predictive distortion caused by deceased patients who cannot physically be readmitted.

#### D. Limitations
- Retrospective observational EHR data from 1999–2008 lacks modern pharmacotherapy (SGLT2 inhibitors, GLP-1 receptor agonists).
- Absence of post-discharge out-of-hospital mortality, social determinants of health (SDOH), and community support resources.

---

### VI. Conclusion & Future Work (approx. 400 words)
- Summary of findings: Tabular gradient boosting and deep learning provide identical moderate predictive power (~0.66 ROC-AUC) for 30-day diabetic readmission.
- Final takeaway: Cost-sensitive learning is indispensable for operational clinical utility under severe class imbalance.
- Future directions: Integrating sequential longitudinal EHR trajectories via Transformers/RNNs, incorporating census-tract socioeconomic indices, and deploying prospective shadow evaluations in real-world EHR systems.

---

## 5–10 Minute Oral Presentation Script & Slide Outline

| Slide # | Slide Title | Recommended Timing | Speaking Points & Visual Cues | Anticipated Technical Defense Questions |
| :---: | :--- | :---: | :--- | :--- |
| **1** | Title & Mission Alignment | 0:00 – 1:00 (1 min) | Introduce problem, university assignment, and your healthcare engineering mission. Explain that 30-day readmissions cost billions and harm diabetic patients. | *"Why is this an operational tool rather than a diagnostic one?"* $\rightarrow$ Model estimates post-discharge operational readmission risk, not diabetes presence. |
| **2** | Dataset & Empirical Verification | 1:00 – 2:30 (1.5 min) | Show raw stats: 101,766 rows, 71,518 patients. Address the 69,984 discrepancy (clinical exclusions of expired/hospice patients). Emphasize 11.3% class imbalance. | *"Why exclude expired patients?"* $\rightarrow$ 100% had label 'NO'; dead patients cannot be readmitted. Retaining them creates artificial leakage. |
| **3** | Leakage Prevention & Grouped Split | 2:30 – 4:00 (1.5 min) | Explain Strategy B (`GroupShuffleSplit`). Show that 16,773 repeat patients could cause massive leakage if split by rows. Prove 0% patient overlap. | *"Why not use random StratifiedKFold?"* $\rightarrow$ Random splitting leaks patient idiosyncrasies into test sets, violating independence. |
| **4** | Modeling: Classical ML vs. Deep Learning | 4:00 – 5:30 (1.5 min) | Explain the 10 experiments: 6 traditional ML (LightGBM, Random Forest, Logistic Reg) and 4 Keras models with `tf.data`. Explain cost-sensitive weighting (7.8x penalty). | *"Why tf.data instead of numpy arrays in Keras?"* $\rightarrow$ Enables batched streaming, autotuned prefetching, and memory scalability for large clinical cohorts. |
| **5** | Results & Comparative Learning Curves | 5:30 – 7:30 (2 min) | Present Master Experiment Table. Show comparative ROC/PR curves and test evaluation table (ROC-AUC ~0.657, Recall ~54%). | *"Why is test accuracy 67% lower than baseline accuracy 88%?"* $\rightarrow$ Cost-sensitive weighting trades unhelpful majority accuracy for 54% clinical recall! |
| **6** | Clinical Error Analysis & Limitations | 7:30 – 9:00 (1.5 min) | Analyze False Negatives vs. False Positives. Discuss why LightGBM matched Deep Learning (Grinsztajn et al. 2022 tabular inductive bias). Note dataset era limitations. | *"Why didn't deep learning outperform LightGBM?"* $\rightarrow$ Tabular clinical data lacks spatial/temporal invariance; decision trees excel on discrete boundaries. |
| **7** | Conclusion & Deliverables | 9:00 – 10:00 (1 min) | Recap deliverables: Colab notebook, clean repository, joblib & .keras model artifacts in Google Drive. Thank faculty and open for questions. | Confident closure demonstrating full technical mastery of code and empirical results. |
