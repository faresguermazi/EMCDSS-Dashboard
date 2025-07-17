# utils/attribution_consistency.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from utils.local_explainer import (
    IG_all_attributions,
    LIG_all_attributions,
    SHAP_all_attributions,
    LayerDeepLift_all_attributions,
    method_map
)
from utils.useful_example_extractor import load_misclassification_data

def flatten_vector(v):
    return np.ravel(v)

# ----------- Consistency Evaluation Per Modality -----------
def evaluate_internal_consistency():
    methods = list(method_map.keys())
    num_patients = min(len(method_map[methods[0]]), 100)  # use first 100 patients for performance

    modalities = ['static', 'text', 'time_series']
    correlations = {mod: pd.DataFrame(index=methods, columns=methods, dtype=float) for mod in modalities}

    for modality in modalities:
        for method1 in methods:
            for method2 in methods:
                all_corrs = []
                for i in range(num_patients):
                    try:
                        if modality == 'static':
                            # get full decoded static attribution vectors
                            attr1 = np.array(method_map[method1][i]['attri_s'])
                            attr2 = np.array(method_map[method2][i]['attri_s'])

                            # make sure vectors match length after decoding
                            if attr1.shape != attr2.shape:
                                continue
                            vec1 = flatten_vector(attr1)
                            vec2 = flatten_vector(attr2)

                        elif modality == 'text':
                            attr1 = np.array(method_map[method1][i]['attri_txt_mean'])
                            attr2 = np.array(method_map[method2][i]['attri_txt_mean'])

                            # Truncate to shortest length to align token counts
                            min_len = min(len(attr1), len(attr2))
                            vec1 = flatten_vector(attr1[:min_len])
                            vec2 = flatten_vector(attr2[:min_len])

                        elif modality == 'time_series':
                            attr1 = np.array(method_map[method1][i]['attri_X']).flatten()
                            attr2 = np.array(method_map[method2][i]['attri_X']).flatten()
                            vec1, vec2 = attr1, attr2

                        if vec1.shape != vec2.shape:
                            continue

                        corr = np.corrcoef(vec1, vec2)[0, 1]
                        if not np.isnan(corr):
                            all_corrs.append(corr)
                    except Exception:
                        continue

                if all_corrs:
                    correlations[modality].loc[method1, method2] = np.mean(all_corrs)

    return correlations

from scipy.stats import spearmanr

def evaluate_spearman_rank_correlation():
    """
    Computes Spearman rank correlation (global consistency) between attribution methods
    for each modality over the first 100 patients.
    """
    methods = list(method_map.keys())
    num_patients = min(len(method_map[methods[0]]), 100)  # limit for performance
    modalities = ['static', 'text', 'time_series']
    
    correlations = {
        mod: pd.DataFrame(index=methods, columns=methods, dtype=float)
        for mod in modalities
    }

    for modality in modalities:
        for method1 in methods:
            for method2 in methods:
                spearman_scores = []
                for i in range(num_patients):
                    try:
                        if modality == 'static':
                            vec1 = flatten_vector(np.array(method_map[method1][i]['attri_s']))
                            vec2 = flatten_vector(np.array(method_map[method2][i]['attri_s']))
                        elif modality == 'text':
                            attr1 = np.array(method_map[method1][i]['attri_txt_mean'])
                            attr2 = np.array(method_map[method2][i]['attri_txt_mean'])
                            min_len = min(len(attr1), len(attr2))
                            vec1 = flatten_vector(attr1[:min_len])
                            vec2 = flatten_vector(attr2[:min_len])
                        elif modality == 'time_series':
                            vec1 = np.array(method_map[method1][i]['attri_X']).flatten()
                            vec2 = np.array(method_map[method2][i]['attri_X']).flatten()
                        else:
                            continue

                        if vec1.shape != vec2.shape or len(vec1) == 0:
                            continue

                        rho, _ = spearmanr(vec1, vec2)
                        if not np.isnan(rho):
                            spearman_scores.append(rho)

                    except Exception:
                        continue

                if spearman_scores:
                    correlations[modality].loc[method1, method2] = np.mean(spearman_scores)

    return correlations

# ----------- Plotting -----------
def plot_correlation_heatmaps(correlations_dict):
    plots = {}
    for modality, df in correlations_dict.items():
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(df.astype(float), annot=True, fmt=".2f", cmap="Blues", ax=ax)
        ax.set_title(f"Cross-method Attribution Correlation: {modality.title()} Modality")
        fig.tight_layout()
        plots[modality] = fig
    return plots

def plot_confidence_vs_attr_strength(prediction_file, method="LIG", modality="time_series"):
    """
    Scatter plot of model confidence vs attribution strength for a given method and modality.
    Useful for evaluating if confident predictions are supported by strong attributions.
    """
    # Load predictions
    df = pd.read_csv(prediction_file)
    confidences = df["prob"].values
    y_pred = df["y_pred"].values
    y_true = df["y_true"].values

    # Get attributions
    attributions = method_map[method]
    attr_strengths = []

    for i in range(len(df)):
        try:
            if modality == "text":
                vec = np.array(attributions[i]["attri_txt_mean"])
            elif modality == "static":
                vec = np.array(attributions[i]["attri_s"])
            elif modality == "time_series":
                vec = np.array(attributions[i]["attri_X"]).flatten()
            else:
                raise ValueError("Unknown modality")

            score = np.mean(np.abs(vec))
            attr_strengths.append(score)
        except:
            attr_strengths.append(np.nan)

    # Filter NaNs
    valid_idx = ~np.isnan(attr_strengths)
    confidences = confidences[valid_idx]
    attr_strengths = np.array(attr_strengths)[valid_idx]
    y_pred = y_pred[valid_idx]
    y_true = y_true[valid_idx]

    # Plot
    fig, ax = plt.subplots(figsize=(6, 4))
    scatter = ax.scatter(confidences, attr_strengths, c=(y_pred == y_true), cmap="coolwarm", alpha=0.7)
    ax.set_xlabel("Prediction Confidence")
    ax.set_ylabel("Attribution Strength (|mean|)")
    ax.set_title(f"Confidence vs Attribution Strength\n(Method: {method}, Modality: {modality})")

    legend1 = ax.legend(*scatter.legend_elements(),
                        loc="upper left", title="Correct")
    ax.add_artist(legend1)

    fig.tight_layout()
    return fig

def load_probabilities(file_path="/netscratch/fguermazi/XAI/evaluation_predictions.csv"):
    df = pd.read_csv(file_path)
    return df["prob"].values

def compute_confidence_vs_attr(method_attributions, probs):
    attr_strengths = []
    confidences = []
    for i in range(min(len(probs), len(method_attributions))):
        try:
            attr = method_attributions[i]["attri_X"]  # time-series only
            strength = np.mean(np.abs(attr))
            confidence = abs(probs[i] - 0.5)
            attr_strengths.append(strength)
            confidences.append(confidence)
        except Exception:
            continue
    return pd.DataFrame({
        "Attribution Strength": attr_strengths,
        "Confidence": confidences
    })

def get_confidence_vs_attr_plots():
    probs = load_probabilities()

    methods = {
        "IG": IG_all_attributions,
        "LIG": LIG_all_attributions,
        "SHAP": SHAP_all_attributions,
        "LayerDeepLift": LayerDeepLift_all_attributions
    }

    plots = {}
    for method_name, attributions in methods.items():
        df = compute_confidence_vs_attr(attributions, probs)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.scatterplot(data=df, x="Confidence", y="Attribution Strength", ax=ax)
        ax.set_title(f"{method_name}: Confidence vs Attribution Strength")
        ax.set_xlabel("Confidence (|prob - 0.5|)")
        ax.set_ylabel("Mean Attribution Strength")
        fig.tight_layout()
        plots[method_name] = fig

    return plots


# ----------- Attribution Strength by Prediction Outcome -----------
def plot_attr_strength_by_outcome():
    """
    For each method, compute mean attribution strength across prediction outcomes (TP, FP, FN, TN).
    Returns one boxplot per method.
    """
    # Get patient groupings
    outcome_dict = load_misclassification_data()
    prediction_types = ["TP", "FP", "FN", "TN"]
    method_plots = {}

    for method_name, attr_list in method_map.items():
        data = []
        for label in prediction_types:
            for idx in outcome_dict[label]["indexes"]:
                try:
                    # Use time series modality for attribution strength
                    attr = np.array(attr_list[idx]["attri_X"])
                    strength = np.mean(np.abs(attr))
                    data.append({"group": label, "strength": strength})
                except Exception:
                    continue

        if not data:
            continue

        # Create DataFrame for boxplot
        import pandas as pd
        df = pd.DataFrame(data)

        # Plot
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(data=df, x="group", y="strength", palette="pastel", ax=ax)
        ax.set_title(f"{method_name}: Attribution Strength by Prediction Type")
        ax.set_ylabel("Mean Attribution Strength")
        ax.set_xlabel("Prediction Outcome")
        fig.tight_layout()

        method_plots[method_name] = fig

    return method_plots

#-----------


