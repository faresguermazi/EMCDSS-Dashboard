# 🧠 EMCDSS-Dashboard: Explainable Multimodal Clinical Decision Support System

## Overview

**EMCDSS-Dashboard** is an interactive dashboard for Explainable Artificial Intelligence (XAI) in multimodal clinical decision support. It provides structured, interpretable visualizations and evaluation tools for patient-level predictions, leveraging both structured (time-series, static) and unstructured (text) data. The dashboard is the main deliverable of a master's thesis focused on advancing explainability in multimodal healthcare AI.

The dashboard enables clinicians, researchers, and developers to:
- Explore patient data and model predictions.
- Visualize and compare attributions from multiple XAI methods across modalities.
- Investigate example-based explanations (similar patients).
- Evaluate the quality and consistency of explanations using both quantitative metrics and human-in-the-loop feedback (CLIX-M checklist).

## Key Features

- **Multimodal XAI**: Supports structured (time-series, static) and unstructured (clinical notes) data.
- **Multiple XAI Methods**: Integrated Gradients (IG), Layer Integrated Gradients (LIG), Layer DeepLift, and SHAP.
- **Example-Based Explanations**: Find and compare similar patients using embedding-based nearest neighbor search.
- **Evaluation Suite**: Quantitative evaluation of attribution consistency, confidence-alignment, and human-centered feedback via the CLIX-M checklist.
- **Interactive Visualizations**: Built with Streamlit for real-time, user-friendly exploration.

## Project Structure

```
EMCDSS-Dashboard-main/
  dashboard.py                # Main Streamlit dashboard application
  utils/
    attribution_consistency.py    # Quantitative evaluation of XAI methods
    clix_m.py                    # CLIX-M human-in-the-loop evaluation logic
    dataset_overview.py          # Dataset statistics and visualization
    example_based_explainer.py   # Example-based (nearest neighbor) explanations
    global_explainer.py          # Global XAI visualizations
    load_patient_data.py         # Data loading and preprocessing
    local_explainer.py           # Local (patient-level) XAI visualizations
    useful_example_extractor.py  # Extraction of useful/misclassified examples
  utils_XAI_files/
    script_Attribution_Methods.py    # Scripts to generate attributions (all methods)
    script_IntegratedGradients.py    # IG-specific attribution generation
    script_LayerDeepLift.py          # LayerDeepLift-specific attribution generation
    script_LIG.py                    # LIG-specific attribution generation
    script_SHAP.py                   # SHAP-specific attribution generation
    script_Example_Based.py          # Embedding/neighbor generation for example-based XAI
    script_TestData_Example_Based.py # Test data for example-based XAI
  README.md
```

## How It Works

### 1. Data and Preprocessing

- The dashboard expects preprocessed patient data, including time-series, static features (e.g., age, gender, ethnicity), and clinical notes.
- Data is loaded from HDF5 and JSON files, with configuration and encoding handled in `utils/load_patient_data.py`.

### 2. Attribution and Embeddings Files

- **Attribution Files**: Scripts in `utils_XAI_files/` (e.g., `script_SHAP.py`, `script_IntegratedGradients.py`, etc.) generate JSONL files containing attributions for each patient and modality, for each XAI method. These files are typically named like:
  - `LIG_attribution_output_*.jsonl`
  - `IG_attribution_output_*.jsonl`
  - `DeepLift_attribution_output_*.jsonl`
  - `SHAP_attribution_output_*.jsonl`
- **Embedding Files**: Example-based scripts generate embedding files (e.g., `train_data.json`, `test_data.json`) used for nearest neighbor search.

These files must be generated before running the dashboard and placed in the expected locations (see code comments for paths). Alternatively, you can download the pre-generated files from the provided cloud storage link (see the section 'Download Data Files: Attribution and Embeddings' above) to avoid the generation process.

### 3. Dashboard Functionality (`dashboard.py`)

- **Patient Overview**: Enter a patient index to view ground truth, prediction, probability, and top-3 similar patients.
- **Dataset Insights**: Visualize dataset statistics (label, age, gender, ethnicity distributions) and time-series feature overviews.
- **Explainability**:
  - **Global**: Compare mean attributions per modality, top features, temporal evolution, and word clouds across XAI methods.
  - **Local**: Inspect patient-level attributions for static, time-series, and text features, with support for method comparison.
  - **Example-Based**: Explore similar patients using embedding-based nearest neighbor search.
- **Evaluation**:
  - **CLIX-M**: Human-in-the-loop evaluation using the CLIX-M checklist, with role-based feedback for clinicians and developers.
  - **Attribution Consistency**: Quantitative plots for internal consistency, confidence-alignment, and outcome-based attribution strength.

### 4. XAI Methods Supported

- **Integrated Gradients (IG)**
- **Layer Integrated Gradients (LIG)**
- **Layer DeepLift**
- **SHAP (DeepLiftShap variant)**

All methods are implemented using the [Captum](https://captum.ai/) library and are adapted for multimodal (structured + text) models.

### 5. Example-Based Explanations

- Embeddings for each patient are generated and stored.
- Nearest neighbor search is performed to find similar patients in the test set, aiding case-based reasoning and interpretability.

### 6. Evaluation and Human Feedback

- **Attribution Consistency**: Correlation and rank-based metrics to compare XAI methods across modalities.
- **Confidence Alignment**: Plots relating model confidence to attribution strength.
- **CLIX-M Checklist**: Collects structured feedback from clinicians and developers on explanation quality, actionability, and fairness.

## Getting Started

### Prerequisites

- Python 3.10.12 (recommended; other 3.10.x versions may work)
- [Streamlit](https://streamlit.io/)
- [PyTorch](https://pytorch.org/)
- [Captum](https://captum.ai/)
- [transformers](https://huggingface.co/transformers/)
- [scikit-learn](https://scikit-learn.org/)
- [pandas](https://pandas.pydata.org/)
- [matplotlib](https://matplotlib.org/)
- [seaborn](https://seaborn.pydata.org/)
- [h5py](https://www.h5py.org/)
- [nltk](https://www.nltk.org/)

Install dependencies:
```bash
pip install -r requirements.txt
```

### Download Data Files: Attribution and Embeddings

The dashboard requires pre-generated attribution and embedding files (JSON/JSONL) for XAI visualizations and example-based explanations. Due to their size, these files are **not included in the repository**.

**To use the dashboard:**
1. Download the required files from the provided cloud storage link:  
https://tubcloud.tu-berlin.de/f/4409671816
2. Place them in the main project folder 

- **Paths**: Update file paths in the scripts and `dashboard.py` as needed for your environment.


### Important: Data and Path Configuration

- The attribution and embedding files required to run the dashboard are already provided for download (see the link above), and the code paths for these files have been updated accordingly. **You do not need to change any paths for these pre-generated files.**

- Some required data files—such as dataset splits, configuration files, and encoder files (e.g., `splits.hdf5`, `discretizer_config.json`, `onehotencoder.pkl`)—are **not included** due to size/privacy. If you have your own versions, update the relevant paths in the code to point to them.

- If you want to generate new attribution or embedding files yourself, you must update any `/workspace/` paths in the data generation scripts (`utils_XAI_files/`) to match your own environment.

This setup allows you to use the dashboard immediately with the provided files, or to generate new data if you have access to the necessary raw files and update the paths accordingly.

### Running the Dashboard


```bash
streamlit run dashboard.py
```



## Acknowledgements

- This project was developed as part of the Master’s thesis entitled "Explainable Multimodal Clinical Decision Support Systems" by Fares Guermazi.
- Built on top of open-source libraries: Streamlit, PyTorch, Captum, HuggingFace Transformers, and more.
- CLIX-M checklist adapted from [Suresh et al., 2021](https://arxiv.org/abs/2107.07511).

## License

For academic use only. Contact the author for other use cases.
