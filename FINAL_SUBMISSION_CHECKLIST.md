# Final Summative Assignment Submission Checklist

**Project Title**: Comparing Traditional Machine Learning and Deep Learning for Predicting 30-Day Hospital Readmission Among Patients with Diabetes  
**Student Mission**: Healthcare Operational Risk Stratification (Software Engineering)  
**Repository URL**: [https://github.com/Amazing-beep/diabetes-readmission-ml-dl](https://github.com/Amazing-beep/diabetes-readmission-ml-dl)  
**Google Drive Folder**: [https://drive.google.com/drive/folders/1LoVTMlsopfQ8C3oAqFJ3xGavl8eiNfFD?usp=drive_link](https://drive.google.com/drive/folders/1LoVTMlsopfQ8C3oAqFJ3xGavl8eiNfFD?usp=drive_link)  
**Target Venue / Format**: University Summative Project (IEEE Conference/Transactions Style Report, 3,500–5,000 words, 5–10 min defense video)

---

## Status Legend
- `[VERIFIED]`: Programmatically tested, executed end-to-end, and validated against assignment rubric criteria.
- `[NOT VERIFIED]`: Pending completion of external upload, personal writing, or webcam recording.
- `[STUDENT ACTION REQUIRED]`: Physical student step (manual file upload, personal report writing, or webcam video recording).

---

## Section A: Codebase & Jupyter Notebook

- [x] `[VERIFIED]` **Raw Data Verification & Target Construction**:
  - Raw dataset verified: 101,766 encounters, 50 columns, 71,518 unique patients.
  - Positive class defined as `readmitted == '<30'` ($N = 11,357$, 11.16%).
  - Negative class defined as `readmitted == 'NO'` or `'>30'` ($N = 90,409$, 88.84%).
  - Excluded 1,652 expired encounters (discharge disposition IDs 11, 19, 20, 21), because 100% were labeled `readmitted == 'NO'`. Dead patients cannot be readmitted; retaining them introduces severe clinical bias.
  - Remaining eligible encounters: **100,114 across 70,439 unique patients**.
- [x] `[VERIFIED]` **Patient-Grouped Splitting (Strategy B)**:
  - Enforced `GroupShuffleSplit` on `patient_nbr` to prevent repeated hospitalizations from leaking across splits.
  - Train (70%): 70,055 encounters (49,305 unique patients), 11.29% positive.
  - Validation (15%): 15,059 encounters (10,568 unique patients), 11.76% positive.
  - Test (15%): 15,000 encounters (10,566 unique patients), 11.20% positive.
  - **Zero Patient Overlap**: Exactly 0 patients shared across splits (`assert len(set_a.intersection(set_b)) == 0`).
- [x] `[VERIFIED]` **Data Leakage & Preprocessing**:
  - Excluded identifiers (`encounter_id`, `patient_nbr`), zero-variance features (`examide`, `citoglipton`), and high-missingness fields (`weight` > 96% missing, `payer_code` ~ 40% missing).
  - Primary, secondary, tertiary diagnoses (`diag_1`, `diag_2`, `diag_3`) grouped into 9 clinical ICD-9 clusters.
  - `ColumnTransformer` (median imputation + StandardScaler for continuous; constant imputation + OneHotEncoder for categorical) fitted **strictly on training data only**.
- [x] `[VERIFIED]` **Model Implementations & Systematic Experiments**:
  - **10 experiments** executed: 6 Traditional ML models (Dummy baseline, Logistic Regression unweighted & balanced, Random Forest balanced, HistGradientBoosting, LightGBM cost-sensitive) and 4 Deep Learning models with TensorFlow/Keras and `tf.data` input pipelines (Baseline Sequential MLP, Deep Regularized Functional MLP, Wide Tuned MLP, Cost-Sensitive Deep MLP).
  - All runs logged programmatically to `results/experiment_results.csv` and backed by Weights & Biases offline tracking (`wandb/`).
- [x] `[VERIFIED]` **End-to-End Notebook Execution**:
  - `notebooks/diabetes_readmission_classification.ipynb` executed top-to-bottom via `jupyter nbconvert` with exit code 0.
  - All executed code cells, printed metrics tables, and matplotlib figures are permanently embedded inside the `.ipynb` file.

---

## Section B: GitHub Repository

- [x] `[VERIFIED]` **Remote Synchronization**:
  - Remote URL verified: `origin -> https://github.com/Amazing-beep/diabetes-readmission-ml-dl.git`.
  - All commits pushed successfully to `main` branch on GitHub.
- [x] `[VERIFIED]` **Repository Cleanliness & Exclusions**:
  - Raw CSV files (`diabetic_data.csv`, `IDS_mapping.csv`, `data/`) and ZIP archives are **strictly excluded** via `.gitignore`.
  - Large binary model weights (`*.joblib`, `*.keras`) are **strictly excluded** from Git.
  - Repository contains only clean source code (`src/`), executed notebook (`notebooks/`), documentation (`README.md`, `report/`), dependencies (`requirements.txt`), metadata, and non-sensitive results (`results/`).
- [x] `[VERIFIED]` **Dependencies (`requirements.txt`)**:
  - Lists verified versions: `numpy`, `pandas`, `scikit-learn`, `scipy`, `matplotlib`, `seaborn`, `lightgbm`, `xgboost`, `tensorflow`, `joblib`, `wandb`, `jupyter`.

---

## Section C: Google Drive Artifacts & Storage

- [x] `[VERIFIED]` **Local Generation of Required Drive Artifacts**:
  1. Raw Dataset Archive: `C:\Users\pc\Downloads\diabetes+130-us+hospitals+for+years+1999-2008.zip` (3,170,254 bytes)
  2. Best Traditional Model: `C:\Users\pc\Downloads\diabetes-readmission-ml-dl\models\best_traditional_model.joblib` (552,767 bytes; includes fitted ColumnTransformer + LightGBM)
  3. Best Deep Learning Model: `C:\Users\pc\Downloads\diabetes-readmission-ml-dl\models\best_deep_learning_model.keras` (504,451 bytes; native Keras 3 format)
  4. Model Metadata: `C:\Users\pc\Downloads\diabetes-readmission-ml-dl\models\model_metadata.json` (627 bytes)
- [x] `[VERIFIED]` **Artifact Reloading & Inference Consistency**:
  - Both saved models were reloaded from disk and verified on raw input samples, confirming identical probabilistic predictions.
- [ ] `[NOT VERIFIED / STUDENT ACTION REQUIRED]` **Manual Google Drive Upload**:
  - Open: [https://drive.google.com/drive/folders/1LoVTMlsopfQ8C3oAqFJ3xGavl8eiNfFD?usp=drive_link](https://drive.google.com/drive/folders/1LoVTMlsopfQ8C3oAqFJ3xGavl8eiNfFD?usp=drive_link)
  - Drag and drop the 4 files listed above into the folder.
  - Set folder sharing access to: **"Anyone with the link can view"**.

---

## Section D: Academic Written Report (IEEE Format)

- [x] `[VERIFIED]` **Structure & Citations Base**:
  - Full IEEE academic report completed at [`report/report_outline.md`](report/report_outline.md) with exact IEEE structure: Abstract, Index Terms, I. Introduction, II. Literature Review, III. Methodology, IV. Results, V. Discussion, VI. Conclusion, VII. References, Appendix.
  - Exactly **11 genuine scholarly sources** verified with full bibliographic metadata, authors, journals, and DOIs (Strack 2014, Kansagara 2011, Shickel 2018, Rajkomar 2018, Saito 2015, Ke 2017, Kaufman 2012, Rubin 2015, He 2009, Lundberg 2017, Grinsztajn 2022).
  - Populated with actual experimental tables (Validation and Test) and references to generated figures (`comparative_roc_pr_curves.png`, `confusion_matrix_*.png`, `exp-dl-*_learning_curves.png`).
- [x] `[VERIFIED]` **Report Word Count Calibration (~3,500 Words - Concise & Short)**:
  - **Main Body (Sections I to VI)**: **2,579 words**.
  - **Main Body + Abstract + Header**: **2,897 words**.
  - **Total Report with References & Appendix**: **3,538 words** (~3,500 words target achieved).
  - Section Breakdown: Header (65w), Abstract (253w), Intro (239w), Literature Review (344w), Methodology (582w), Results & Tables (755w), Discussion (444w), Conclusion (215w), References [1]–[11] (323w), Appendix 7-Slide Defense (318w).
- [ ] `[STUDENT ACTION REQUIRED]` **Personal Voice Review & AI Compliance Check**:
  - Read through [`report/report_outline.md`](report/report_outline.md) in your own voice to ensure personal alignment.
  - Fill in your author details (Student Name, Student ID, Department) on the title block.
  - Ensure the report reflects your authentic student voice to adhere to your institution's AI policy (<20% AI-generated text).

---

## Section E: Recorded Presentation Video (5–10 Minutes)

- [x] `[VERIFIED]` **Slide Deck Outline & Defense Script**:
  - Complete 7-slide outline with minute-by-minute speaking cues and anticipated faculty defense questions provided in [`report/report_outline.md`](report/report_outline.md).
- [ ] `[NOT VERIFIED / STUDENT ACTION REQUIRED]` **Record Defense Video**:
  - Duration: 5 to 10 minutes.
  - Camera: **Webcam ON, face visible throughout the presentation** (explicit rubric requirement).
  - Content: Explain problem statement, dataset, leakage prevention (`GroupShuffleSplit`), classical ML vs. Deep Learning comparison, and clinical interpretation of false negatives vs. false positives.
  - Upload: Upload video to YouTube (Unlisted), Loom, or Google Drive, and ensure sharing access is set to "Anyone with the link can view".

---

## Section F: Final Submission Package & Link Verification

- [ ] `[NOT VERIFIED / STUDENT ACTION REQUIRED]` **Header Link Verification**:
  - Ensure the first page / header of your final submitted report PDF contains all three clickable links:
    1. **GitHub Repository**: `https://github.com/Amazing-beep/diabetes-readmission-ml-dl`
    2. **Google Drive Folder**: `https://drive.google.com/drive/folders/1LoVTMlsopfQ8C3oAqFJ3xGavl8eiNfFD?usp=drive_link`
    3. **Presentation Video Link**: *(Your recorded video URL)*
- [ ] `[NOT VERIFIED / STUDENT ACTION REQUIRED]` **Test Links in Incognito Window**:
  - Open a private/incognito browser window and verify that all three links open without requiring login or requesting permission.
- [ ] `[NOT VERIFIED / STUDENT ACTION REQUIRED]` **Export Final PDF**:
  - Compile your completed written report into a single clean PDF and submit it through your university portal.

---

## Immediate Next Steps (In Deadline Priority Order)

1. **Step 1 (5 mins)**: Open your Google Drive folder ([link](https://drive.google.com/drive/folders/1LoVTMlsopfQ8C3oAqFJ3xGavl8eiNfFD?usp=drive_link)) and upload:
   - `C:\Users\pc\Downloads\diabetes+130-us+hospitals+for+years+1999-2008.zip`
   - `C:\Users\pc\Downloads\diabetes-readmission-ml-dl\models\best_traditional_model.joblib`
   - `C:\Users\pc\Downloads\diabetes-readmission-ml-dl\models\best_deep_learning_model.keras`
   - `C:\Users\pc\Downloads\diabetes-readmission-ml-dl\models\model_metadata.json`
   - Verify folder permission is set to **"Anyone with the link can view"**.
2. **Step 2 (20-30 mins)**: Review the completed academic report in [`report/report_outline.md`](report/report_outline.md). Add your name, student ID, and personal reflections to ensure the text matches your authentic student voice and adheres to your university's AI policy (<20% AI-generated text).
3. **Step 3 (30 mins)**: Record your 5–10 minute oral defense presentation with your webcam ON, using the slide outline and speaking script in the Appendix of [`report/report_outline.md`](report/report_outline.md).
4. **Step 4 (10 mins)**: Insert your recorded video link into the header of your report, verify all links in an incognito window, export to PDF, and submit through your portal.
