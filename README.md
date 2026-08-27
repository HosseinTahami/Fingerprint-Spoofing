# Fingerprint Spoofing Detection

A comparative study of generative and discriminative classifiers for detecting spoofed
(fake) fingerprint images from genuine ones, using a 6-dimensional feature representation.
Built as a semester-long machine learning project: every stage — exploratory analysis,
dimensionality reduction, density estimation, generative and discriminative classifiers,
decision-theoretic evaluation, calibration, and fusion — is implemented from scratch in
NumPy/SciPy, without relying on `scikit-learn` model implementations.

Full write-up with all figures, result tables, and discussion: **[`report.pdf`](report.pdf)**
(LaTeX source in [`report.tex`](report.tex)).

## Task

Binary classification: given a 6-dimensional feature vector extracted from a fingerprint
image, predict whether it is **genuine** (class 1) or **spoofed/fake** (class 0). The
training set has 6000 labeled samples (2990 fake, 3010 genuine); a separate 6000-sample
evaluation set is held out for final, unbiased testing of the delivered system.

## Repository structure

```
── NoteBooks
│   ├── P1 - feature analysis.ipynb
│   ├── P2 - Dim Reduction (PCA, LDA).ipynb
│   ├── P3 - Density Probability.ipynb
│   ├── P4 - Generative Gaussian.ipynb
│   ├── P5 - Bayes Decision.ipynb
│   ├── P6 - Logistic Regression.ipynb
│   ├── P7 - SVM.ipynb
│   ├── P8 - GMM.ipynb
│   ├── P9 -  Calibration Fusion.ipynb
│   └── __pycache__
│       └── utils.cpython-312.pyc
├── Project.md
├── README.md
├── Report
│   ├── report.pdf
│   └── report.tex
├── Results
│   ├── Histogram Plots.png
│   └── Scatter Plots.png
└── Utils
    └── utils.py
```

Each notebook corresponds to one stage of the pipeline and can be run independently
against `data/trainData.txt` (the later notebooks additionally use `data/evalData.txt`
for the final, held-out evaluation).

## Method overview

| Stage | What's done |
|---|---|
| **Exploratory analysis** | Per-class histograms and pairwise scatter plots to characterize overlap, mean/variance structure, and modality of each feature |
| **Dimensionality reduction** | PCA (all 6 directions) and 1D LDA; LDA used directly as a linear classifier, with and without PCA pre-processing |
| **Density estimation** | Maximum-likelihood univariate Gaussian fit per class, per feature |
| **Generative models** | Full-covariance MVG, tied-covariance MVG, and diagonal-covariance Naive Bayes; covariance/correlation analysis |
| **Decision theory** | Optimal Bayes decisions, minimum and actual DCF, effective priors, Bayes error plots across 5 application scenarios |
| **Logistic regression** | Standard, prior-weighted, and quadratic (degree-2 feature expansion) formulations; regularization sweep |
| **SVM** | Linear, degree-2/4 polynomial kernel, and RBF kernel, with C/γ grid search |
| **GMM** | Full-covariance mixtures per class, component count selected by validation minDCF |
| **Calibration & fusion** | K-fold score calibration (prior-weighted logistic regression) and score-level fusion of the best model per family; final delivered system confirmed on the held-out evaluation set |

## Key results

All figures below use minimum DCF (minDCF) at the primary application
(π_T = 0.1, C_fn = C_fp = 1), i.e. the best achievable cost with an ideal threshold —
the standard way to compare classifiers independently of calibration quality.

| Model | Best minDCF |
|---|---|
| **GMM (full covariance, asymmetric components)** | **0.150** |
| SVM (degree-4 polynomial kernel) | 0.177 |
| SVM (RBF kernel) | ≈0.18 |
| Logistic Regression (quadratic) | 0.244 |
| Naive Bayes Gaussian | 0.257 |
| MVG (full covariance) | 0.263 |
| Logistic Regression / SVM (linear) / Tied Gaussian | ≈0.36 |

The dataset's two most informative features (5 and 6) have a multi-modal, clustered
distribution that no linear or single-Gaussian model can represent well; every model
capable of representing non-linear or multi-component structure (quadratic LR,
polynomial/RBF SVM, GMM) substantially outperforms the linear/single-Gaussian baselines,
with the GMM — the most flexible density model tested — performing best overall.

After K-fold calibration and fusion, the **calibrated GMM** was selected as the final
delivered system based on actual DCF, and confirmed on the independent evaluation set
(minDCF = 0.154, actDCF = 0.168), with good calibration maintained across a wide range
of operating points. Full details, all intermediate results, and Bayes error plots are
in [`report.pdf`](report.pdf).

## Requirements

```
numpy
scipy
matplotlib
pandas
```

All models (Gaussian classifiers, logistic regression, SVM, GMM, DCF/calibration
utilities) are implemented from scratch; no `scikit-learn` estimators are used.

## Reproducing the results

```bash
git clone <repo-url>
cd fingerprint-spoofing-detection
pip install -r requirements.txt
jupyter notebook notebooks/
```

Run the notebooks in order (`P1` → `P9`); each is self-contained given
`data/trainData.txt` (and `data/evalData.txt` for `P9`).

## Author

Seyed Hossein Tahami — MSc Computer Engineering (AI and Data Analysis), Politecnico di Torino
