# ES335-25-fall-assignment3







#Question 1
C++ Code Generation Project
For category II : Next-Token Predictor (C++ Code)

* **Download `model_cpp_low.pth`:** https://drive.google.com/file/d/1f917U2an9pKLlEfLEjh59n-NqAK7TMcO/view?usp=sharing
* **Download `model_cpp_medium.pth`:** https://drive.google.com/file/d/1r954Rf09DwE9mmTnuuvMGKPL1lTnk9tF/view?usp=sharing
* **Download `model_cpp_high.pth`:** https://drive.google.com/file/d/1qupmuOIr8o2CzP4HXMeKaJEdlHYqfRt3/view?usp=sharing

For category I : Sherlock Holmes (

* **Download `model_cpp_low.pth`:** https://drive.google.com/file/d/1IXzqHBSZUfYfd68SipxCdyT5MpMRdl_2/view?usp=sharing
* **Download `model_cpp_medium.pth`:** https://drive.google.com/file/d/1HzR1FMEMj7Kg-gU2ABGmBs9c_Yt2PC1b/view?usp=sharing
* **Download `model_cpp_high.pth`:** https://drive.google.com/file/d/1M0AT5eipWW7BPlH3hoRAUk8AM6wilosG/view?usp=sharing

Once the files are in the directory, all cells (including t-SNE and the Streamlit app) will run in just a few minutes.

For Category 1 : Next-Token Predictor (C++ Code):
https://c-codegeneerator-avrhahevm8ar9m2tusxhas.streamlit.app/ 

For Category 2 : Sherlock Holmes
https://c-codegeneerator-zrddtfz8psxwgbxyhvgcwl.streamlit.app/



# ES335-25 — Assignment 3

Overview
--------
This repository contains the deliverables for Assignment 3 of ES335-25 (Fall). The assignment has three main parts:

1. Next-Word Prediction using an MLP (text generation + embedding visualizations + Streamlit app) — Question 1  
2. Make-Moons dataset experiments and regularization comparisons — Question 2  
3. MNIST experiments with MLP and CNN (and comparisons to baseline models and pretrained networks) — Question 3

Each question has a self-contained notebook named `question1.ipynb`, `question2.ipynb`, and `question3.ipynb`. The notebooks include code, experimental results, plots, commentary and answers to the report prompts. A Streamlit app for interactive next-word generation is also included.

Repository structure
--------------------
- `question1.ipynb` — Next-word prediction (MLP), embedding visualization and Streamlit app link / usage notes. Contains preprocessing, model training, t-SNE plots, comparisons (Category I vs Category II), sample generations, and instructions to run the Streamlit app.
- `question2.ipynb` — Make-Moons experiments: dataset generation, models (MLP variants, L1/L2 regularization, logistic regression with polynomial features), evaluation, decision-boundary plots and imbalance experiments.
- `question3.ipynb` — MNIST experiments with MLP & CNN: training, baselines (Random Forest, Logistic Regression), t-SNE visualizations, cross-domain tests on Fashion-MNIST, pretrained CNN inference and comparisons.
- `streamlit_app/` — Streamlit app code and assets:
  - `app.py` — main Streamlit app for interactive next-word prediction
  - `models/` — trained model checkpoints used by the app (if included)
  - `vocab/` — vocabulary files (if included)
- `data/` — dataset download / extraction scripts and small example data files (NOTE: large raw datasets may be downloaded on demand; notebooks include the download links)
- `requirements.txt` — Python package dependencies
- `README.md` — this file
- `results/` — saved plots, trained model logs, and evaluation tables (CSV/PNG)
- `utils/` — helper scripts for preprocessing, training loops, metrics, and plotting

What to look for in each notebook: 
----------------------------------------------------
Question 1:
- Preprocessing steps and vocabulary construction (vocab size, 10 most frequent & 10 least frequent words).
- Training vs validation loss plots, final validation loss/accuracy.
- Example predictions (samples) and discussion of learning behavior.
- Embedding visualization (t-SNE or scatter) with observations and interpretation.
- Streamlit app with controls for context length, embedding dim, activation, temperature, random seed, and a strategy for handling OOV words (documented).
- Comparative analysis between Category I (natural language) and Category II (structured) datasets.

Question 2:
- Custom Make-Moons generator (no sklearn.make_moons).
- Train/evaluate: MLP (early stopping), MLP with L1 grid, MLP with L2, logistic regression with polynomial features.
- Plots: validation AUROC vs lambda for L1 grid, decision boundaries for all models (side-by-side), table of accuracies across noise levels (0.1, 0.2, 0.3) with parameter counts.
- Experiments with class imbalance (70:30) and discussion.

Question 3:
- MLP on MNIST (layer sizes 30 → 20 → 10), baselines (Random Forest, Logistic Regression), metrics (accuracy, F1, confusion matrix).
- t-SNE visualizations of the 20-neuron layer (trained vs untrained).
- Cross-domain evaluation on Fashion-MNIST and t-SNE comparisons.
- CNN implementation: Conv(32×3×3) → MaxPool → FC(128) → FC(10), pretrained CNN inferences, comparisons for accuracy, F1, confusion matrix, model size (# params), and inference time.


