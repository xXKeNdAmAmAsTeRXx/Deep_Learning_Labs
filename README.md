# Deep Learning Labs

A collection of Jupyter notebooks and Python scripts implementing core deep-learning concepts with **PyTorch** and the standard scientific Python stack.

## Repository Layout

```
Lab1/            – Perceptrons, logistic regression, XOR problem
Lab2/            – PyTorch tensors, gradient descent, nn.Module
Lab3/            – MLP classification on MNIST, optimisers, Optuna, dropout
Lab4/            – MLP regression & multi-label classification, Optuna
Lab5/            – TensorBoard integration and Optuna visualisation
Lab6_7-project/  – Milestone project: air-quality multiclass classification
Lab8/            – Audio classification pipeline, Optuna, transfer learning, augmentation
Lab9/            – Probability calibration and sklearn wrapper integration for PyTorch models
lca_courses_projects/ – LangChain Foundation course projects
```

---

## Additional LLM Projects

The `lca_courses_projects/` directory contains projects from the LangChain Foundation track.

- **LangChain**: a framework for building LLM-powered applications (chains, prompts, tools, and retrieval).
- **LangGraph**: an orchestration layer for stateful, multi-step, and agent-like workflows built on top of LangChain concepts.
- **LangSmith**: a platform for tracing, debugging, evaluating, and monitoring LLM applications.

---

## Lab Descriptions

### Lab 1 – Perceptrons, Logistic Regression, and the XOR Problem

**Files:** `Lab1/1_XOR_and_OR_problem.ipynb` · `Lab1/2_MLP_for_XOR.ipynb` · `Lab1/3_Decysion_Boundry.ipynb` · `Lab1/4_XOR_Logistic_dim.ipynb`

Explores the classical XOR/OR binary-classification problem as a gateway to understanding linear separability.

- **`1_XOR_and_OR_problem.ipynb`** — NumPy hand-coded perceptron for OR; demonstrates why a single linear classifier fails on XOR; confirmed with `sklearn.linear_model.LogisticRegression` and confusion-matrix / classification-report metrics.
- **`2_MLP_for_XOR.ipynb`** — Two-layer PyTorch MLP (`nn.Module`) that correctly learns XOR, showing why non-linearity is necessary.
- **`3_Decysion_Boundry.ipynb`** — Decision-boundary visualisation using an sklearn `Pipeline` + `FunctionTransformer` for polynomial feature expansion; matplotlib/seaborn plots.
- **`4_XOR_Logistic_dim.ipynb`** — Further study of dimension-extending feature maps applied to logistic regression for non-linear boundaries.

**Topics:** perceptron, linear separability, logistic regression, feature engineering, decision boundaries, MLP.  
**Dataset:** Synthetic XOR/OR binary data (NumPy arrays).

---

### Lab 2 – PyTorch Fundamentals

**Files:** `Lab2/00_tensors.ipynb` · `Lab2/01_Gradient_decent.ipynb` · `Lab2/02_Torch_Modules.ipynb`

Hands-on introduction to PyTorch as a numerical computing and deep-learning framework.

- **`00_tensors.ipynb`** — Tensor creation, arithmetic, broadcasting, CPU ↔ GPU transfer (`to("cuda")`), and visualisation with seaborn/matplotlib.
- **`01_Gradient_decent.ipynb`** — Gradient descent on a 1-D regression problem: first with manually computed gradients, then with PyTorch autograd.
- **`02_Torch_Modules.ipynb`** — Building custom `nn.Module` subclasses with `nn.Linear`; forward pass; full training loop on synthetic data; saving/loading weights (`Lab2/model.pth`).

**Topics:** tensors, autograd, gradient descent, `nn.Module`, `nn.Linear`, training loop.  
**Dataset:** Synthetic regression data (generated in-notebook).

---

### Lab 3 – MLP on MNIST: Optimisers, Cross-Validation, Ensembling, Dropout

**Files:** `Lab3/MNIST_MLP.ipynb` · `Lab3/MNIST_SGD.ipynb` · `Lab3/MNIST_ADAM.ipynb` · `Lab3/MNIST_LR.ipynb` · `Lab3/Dropout_regularization.ipynb`

Applies MLPs to the MNIST handwritten-digit dataset (70 000 samples, 784 features, 10 classes), loaded via `sklearn.datasets.fetch_openml('mnist_784', version=1)`.

- **`MNIST_MLP.ipynb`** — Baseline MLP with 5-fold cross-validation; majority-voting ensemble; model uncertainty measured via Shannon entropy (`scipy.stats.entropy`).
- **`MNIST_SGD.ipynb`** — SGD-optimised MLP; Optuna study to tune the learning rate; retrain with the best trial.
- **`MNIST_ADAM.ipynb`** — Same pipeline with the Adam optimiser and Optuna hyperparameter search.
- **`MNIST_LR.ipynb`** — Learning-rate sweep with Optuna on an SGD base optimiser; final model evaluation.
- **`Dropout_regularization.ipynb`** — Dropout layers for regularisation; mean-response ensembling (predictions averaged across multiple stochastic forward passes in `train` mode).

**Topics:** MNIST, MLP, SGD, Adam, K-fold CV, Optuna, dropout, ensembling, model uncertainty.  
**Dataset:** MNIST (`mnist_784`, OpenML).

---

### Lab 4 – MLP Regression and Multi-Label Classification

**Files:** `Lab4/1_Regression-part1.ipynb` · `Lab4/2_Regression_optuna.ipynb` · `Lab4/3_Multilabel_classification.ipynb` · `Lab4/4_classification_architectures.ipynb` · `Lab4/5_shapes.ipynb` · `Lab4/MLPRegressor.py`

Extends MLPs beyond classification into regression and multi-label scenarios.

- **`1_Regression-part1.ipynb`** — `MLPRegressor` (`nn.Module`) on the **California Housing** dataset (OpenML `data_id=43939`, ≈20 640 samples, 9 features including `ocean_proximity`); compares MLP depth; evaluates with MSE and R².
- **`2_Regression_optuna.ipynb`** — Optuna-driven hyperparameter tuning (hidden size, depth, learning rate) for `MLPRegressor` on California Housing; K-fold cross-validation.
- **`3_Multilabel_classification.ipynb`** — Multi-label classification on the OpenML **Segment** dataset (`id=36`, image-segment features); `StratifiedKFold`; custom `ModelTrainer` class; Optuna; Hamming loss and F1 evaluation.
- **`4_classification_architectures.ipynb`** — Architecture comparison (various MLP widths/depths) for multi-label classification on the OpenML **Scene** dataset (`id=312`, 294 image attributes, 6 labels: Beach, Sunset, FallFoliage, Field, Mountain, Urban).
- **`5_shapes.ipynb`** — Investigates tensor/layer shape arithmetic for the Scene dataset; builds on architectures from notebook 4.
- **`MLPRegressor.py`** — Reusable `nn.Module`; constructor parameters: `input_size`, `hidden_size`, `n_hidden`, `output_size`; ReLU activations; linear output (no activation, for regression).

**Topics:** regression (MSE, R²), multi-label classification (Hamming loss, F1), Optuna, K-fold CV, MLP architecture search.  
**Datasets:** California Housing (OpenML 43939), Segment (OpenML 36), Scene (OpenML 312).

---

### Lab 5 – TensorBoard Monitoring and Optuna Visualisation

**Files:** `Lab5/1_Tensorboard.ipynb` · `Lab5/2_Training_TensorBoard.ipynb` · `Lab5/3_optuna_tensorboard.ipynb` · `Lab5/4_Other_functions_Tenosrboard.ipynb`

Covers experiment tracking and visualisation with TensorBoard, integrated into PyTorch training loops.

- **`1_Tensorboard.ipynb`** — Introduction to `torch.utils.tensorboard.SummaryWriter`; writing scalar logs; launching TensorBoard.
- **`2_Training_TensorBoard.ipynb`** — Full MLP training on MNIST with per-epoch loss and accuracy logged via `SummaryWriter`; train/val split; uses `shutil` for log-directory management.
- **`3_optuna_tensorboard.ipynb`** — Combines Optuna hyperparameter search with TensorBoard logging; each Optuna trial writes its own run for side-by-side comparison.
- **`4_Other_functions_Tenosrboard.ipynb`** — Additional `SummaryWriter` APIs: `add_image`, `add_figure`, histogram logging.

**Topics:** TensorBoard, `SummaryWriter`, experiment tracking, Optuna integration, MNIST.  
**Dataset:** MNIST (`mnist_784`, OpenML) in notebooks 2 and 3.

---

### Lab 6 / 7 – Milestone Project: Air Quality Classification

**Files:** `Lab6_7-project/1_Dataset_data_cleaning_split.ipynb` · `Lab6_7-project/2_EDA.ipynb` · `Lab6_7-project/3_Optuna.ipynb` · `Lab6_7-project/4_Training.ipynb` · `Lab6_7-project/5_Evaluation_TOP1.ipynb` · `Lab6_7-project/5_Evaluation_TOP2.ipynb` · `Lab6_7-project/5_Evaluation_TOP3.ipynb` · `Lab6_7-project/6_Model_Testing.ipynb`

An end-to-end multiclass classification project on the **Air Quality and Pollution Assessment** dataset (5 000 samples, 9 environmental/demographic features, 4 target classes: Good, Moderate, Poor, Hazardous).

- **`1_Dataset_data_cleaning_split.ipynb`** — Loads the dataset from OpenML; checks for missing values; performs stratified train/validation and test split; saves splits to CSV.
- **`2_EDA.ipynb`** — Exploratory data analysis: pair plots, normality tests, Pearson correlation heatmap, Variance Inflation Factor (VIF), mutual information scores; feature selection rationale; applies `RobustScaler` to all features.
- **`3_Optuna.ipynb`** — Optuna hyperparameter study (100 trials, 5-fold cross-validation) for MLP learning rate, batch size, and Adam β₁/β₂; results saved to `optuna_results/`; integrated with TensorBoard logging.
- **`4_Training.ipynb`** — Trains the top-3 Optuna configurations using a custom `SummaryWriter`-backed training loop; saves each fold's model weights and parameters to `Models/`; visualises the best model architecture with `torchview`.
- **`5_Evaluation_TOP1/2/3.ipynb`** — Per-model evaluation: classification report, confusion matrix, ROC-AUC curves (one-vs-rest), ensemble predictions averaged across folds.
- **`6_Model_Testing.ipynb`** — Final statistical comparison of the three ensembles on held-out test data using the Friedman test with post-hoc Nemenyi correction (`utils/stat_testing.py`); decision-coverage analysis.

**Utility modules (`Lab6_7-project/utils/`):**
- `MLPClassifier.py` — Configurable `nn.Module` (ReLU or Tanh activation, variable depth/width).
- `training.py` — `create_model`, `create_training_dict`, `train_from_dict` helpers with K-fold, gradient clipping, `ReduceLROnPlateau` scheduler.
- `Predictor.py` — Loads a saved ensemble from disk and provides predict / evaluate methods.
- `EDA.py` — Helper functions: normality check, VIF calculation, mutual information plot, correlation heatmap, Robust scaling.
- `stat_testing.py` — `compare_ensembles`: Friedman test + Nemenyi post-hoc critical difference.

**Topics:** multiclass classification, EDA, VIF, feature selection, Optuna, K-fold CV, TensorBoard, ensemble evaluation, Friedman/Nemenyi statistical testing.  
**Dataset:** Air Quality and Pollution Assessment (OpenML / [Kaggle](https://www.kaggle.com/datasets/mujtabamatin/air-quality-and-pollution-assessment), 5 000 samples).

---

### Lab 8 – Audio Classification: Preprocessing, Optimisation, Transfer Learning, and Augmentation

**Files:** `Lab8/1_AudioData_Preprocessing.ipynb` · `Lab8/2_Training.ipynb` · `Lab8/3_Optuna.ipynb` · `Lab8/4_Training_optimized.ipynb` · `Lab8/5_dataset_size.ipynb` · `Lab8/6_Transfer_Learning.ipynb` · `Lab8/7_data_augmentation.ipynb` · `Lab8/8_training_aug.ipynb`

Builds an end-to-end audio-classification workflow, from waveform preprocessing to model optimisation and robustness checks.

- **`1_AudioData_Preprocessing.ipynb`** — Audio preprocessing pipeline with `librosa`/`torchaudio`: resampling to 8 kHz, waveform and mel-spectrogram visualisation, and dataset directory preparation.
- **`2_Training.ipynb`** — Baseline training and evaluation of a custom 1D CNN classifier (`utils8/AudioCNN.py`) on processed audio data.
- **`3_Optuna.ipynb`** — Optuna-based hyperparameter search for audio-CNN training with experiment tracking.
- **`4_Training_optimized.ipynb`** — Retraining and validating models with the best Optuna configuration.
- **`5_dataset_size.ipynb`** — Sensitivity study of model performance versus training-dataset size.
- **`6_Transfer_Learning.ipynb`** — Transfer-learning experiments using previously trained audio models and cross-dataset fine-tuning.
- **`7_data_augmentation.ipynb`** — Audio augmentation design and testing (e.g., noise injection, pitch shift, reverb, bass boost) using custom transforms from `utils8/augmentations.py`.
- **`8_training_aug.ipynb`** — Training/evaluation with augmented data and comparison against non-augmented baselines.

**Topics:** audio preprocessing, mel-spectrograms, 1D CNNs, Optuna, transfer learning, data augmentation, model comparison.  
**Dataset:** Audio command recordings organized in class-based directories (processed to 1-second, 8 kHz samples in notebook workflows).

---

### Lab 9 – Model Calibration and sklearn-Compatible Wrappers

**Files:** `Lab9/1_toy_set.ipynb` · `Lab9/2_ensemble_testing.ipynb` · `Lab9/3_Wrapper.ipynb` · `Lab9/4_real_data_calibration.ipynb`

Focuses on probability calibration for neural classifiers and interoperability between PyTorch models and sklearn tooling.

- **`1_toy_set.ipynb`** — Binary toy-dataset experiments with temperature scaling; compares pre- vs post-calibration probabilities and Brier score.
- **`2_ensemble_testing.ipynb`** — Ensemble-level calibration analysis with mean-response aggregation and statistical comparison of calibrated vs uncalibrated predictions.
- **`3_Wrapper.ipynb`** — Implements an sklearn-compatible wrapper (`BaseEstimator`/`ClassifierMixin`) around a PyTorch model and applies `CalibratedClassifierCV` (isotonic calibration).
- **`4_real_data_calibration.ipynb`** — Extends wrapper + calibration workflow to a real multiclass OpenML dataset (`openml.datasets.get_dataset(46880)`), with one-vs-rest calibration curves.

**Topics:** temperature scaling, isotonic calibration, Brier score, calibration curves, ensemble probability aggregation, sklearn wrapper design.  
**Dataset:** Synthetic toy data (notebooks 1–3) and OpenML dataset 46880 (notebook 4).

---

## Installation

### 1. Create a Python environment

Python **3.10** or **3.11** is recommended.

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
```

### 2. Install PyTorch

PyTorch must be installed separately so you can choose the right compute backend:

```bash
# CPU-only (any OS)
pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu

# CUDA 11.8
pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121
```

### 3. Install remaining dependencies

```bash
pip install -r requirements.txt
```

### 4. Launch Jupyter

```bash
jupyter notebook
# or
jupyter lab
```

---

## Dependencies

| Package | Version | Used in |
|---|---|---|
| `torch` | 2.3.1 | All labs (install separately – see above) |
| `numpy` | 1.26.4 | All labs |
| `scipy` | 1.13.1 | Lab 3, Lab 6/7 |
| `pandas` | 2.2.2 | Lab 4, Lab 5, Lab 6/7 |
| `scikit-learn` | 1.5.0 | Lab 1–5, Lab 6/7 |
| `matplotlib` | 3.9.0 | Lab 1–5, Lab 6/7 |
| `seaborn` | 0.13.2 | Lab 1–2, Lab 6/7 |
| `optuna` | 3.6.1 | Lab 3–5, Lab 6/7 |
| `openml` | 0.14.2 | Lab 4, Lab 6/7 |
| `tensorboard` | 2.17.0 | Lab 5, Lab 6/7 |
| `statsmodels` | 0.14.4 | Lab 6/7 |
| `torchview` | 0.2.7 | Lab 6/7 |

---

## Notes

- All notebooks auto-detect CUDA via `torch.device("cuda" if torch.cuda.is_available() else "cpu")` — no code changes are needed when switching between CPU and GPU.
- OpenML datasets (MNIST, California Housing, Segment, Scene, Air Quality) are downloaded and cached automatically on first run; internet access is required.
- `Lab4/MLPRegressor.py` must be in the same directory as the Lab 4 notebooks that import it (`from MLPRegressor import MLPRegressor`).
- The Lab 6/7 project notebooks must be run from the `Lab6_7-project/` directory so that the relative `utils/` imports and `Data/`, `Models/`, `optuna_results/` paths resolve correctly.
