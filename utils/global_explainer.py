# global_explainer.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils.local_explainer import (
    IG_all_attributions,
    LIG_all_attributions,
    SHAP_all_attributions,
    LayerDeepLift_all_attributions,
    config,
    method_map
)
from wordcloud import WordCloud
from collections import defaultdict
from utils.load_patient_data import tokenizer,HDF5_PATH,GROUP
import h5py

with h5py.File(HDF5_PATH, "r") as hf:
    test_data = hf[GROUP]["test"]
    all_input_ids = np.array(test_data["input_ids"])  # Shape: (N, Seq)


def generate_mean_attribution_per_modality_plot():
    try:
        def extract_modality_means(attributions, method_name):
            values = []
            for patient in attributions:
                val_txt = np.mean(np.abs(patient['attri_txt_mean']))
                val_ts = np.mean(np.abs(patient['attri_X']))
                val_s = np.mean(np.abs(patient['attri_s']))
                values.append({'Method': method_name, 'Modality': 'Textual data', 'Attribution': val_txt})
                values.append({'Method': method_name, 'Modality': 'Time-Series data', 'Attribution': val_ts})
                values.append({'Method': method_name, 'Modality': 'Static data', 'Attribution': val_s})
            return values

        # Collect all data
        all_data = []
        all_data.extend(extract_modality_means(LIG_all_attributions, "LIG"))
        all_data.extend(extract_modality_means(IG_all_attributions, "IG"))
        all_data.extend(extract_modality_means(SHAP_all_attributions, "SHAP"))
        all_data.extend(extract_modality_means(LayerDeepLift_all_attributions, "LayerDeepLift"))

        df_modalities = pd.DataFrame(all_data)

        # Create summary table
        summary_table = df_modalities.groupby(['Method', 'Modality'])['Attribution'].mean().unstack()
        formatted_values = summary_table.applymap(lambda x: f"{x:.6e}")

        # === Plot: Table + Barplot ===
        fig = plt.figure(figsize=(16, 6))
        gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 2.2])
        fig.suptitle("Mean Attribution per Modality across XAI Methods (Table & Plot)", fontsize=16, weight='bold', y=1.05)

        # Left: Table
        ax1 = fig.add_subplot(gs[0])
        ax1.axis('off')

        table = ax1.table(
            cellText=formatted_values.values,
            colLabels=formatted_values.columns,
            rowLabels=formatted_values.index,
            loc='center',
            cellLoc='center',
            rowLoc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.2, 1.2)

        for (row, col), cell in table.get_celld().items():
            if row == 0 or col == -1:
                cell.set_text_props(weight='bold', color='black')
                cell.set_facecolor('#f0f0f0')
            elif row % 2 == 0:
                cell.set_facecolor('#f9f9f9')
            else:
                cell.set_facecolor('white')

        # Right: Barplot
        ax2 = fig.add_subplot(gs[1])
        #sns.barplot(data=df_modalities, x="Modality", y="Attribution", hue="Method", ci="sd", ax=ax2)
        sns.barplot(data=df_modalities, x="Modality", y="Attribution", hue="Method", errorbar='sd', ax=ax2)

        ax2.set_ylabel("Mean Absolute Attribution")
        ax2.set_xlabel("Modality")
        ax2.grid(True, axis='y')
        ax2.legend(title='Method')

        plt.tight_layout()

        return fig, formatted_values  # Return figure and table

    except Exception as e:
        print(f"Error generating global attribution plot: {e}")
        return None, None

def generate_top10_features_all_methods():

    """
    Generates a bar plot comparing the top 10 time-series features across all XAI methods.
    Returns: (figure, summary_table)
    """
    try:
        def aggregate_feature_attributions(all_attributions, config):
            all_ts_attr = np.array([
                np.abs(sample['attri_X']).mean(axis=0) 
                for sample in all_attributions
            ])
            feature_scores = {}
            idx = 0
            for feat in config["id_to_channel"]:
                if feat in config["is_categorical_channel"] and config["is_categorical_channel"][feat]:
                    n_cats = len(config["possible_values"][feat])
                    mean_attr = np.mean(all_ts_attr[:, idx:idx + n_cats], axis=1)
                    score = np.mean(mean_attr)
                    idx += n_cats
                else:
                    score = np.mean(all_ts_attr[:, idx])
                    idx += 1
                feature_scores[feat] = score
            return feature_scores

        # Step 1: Compute per-method feature attributions
        lig_feats = aggregate_feature_attributions(LIG_all_attributions, config)
        ig_feats = aggregate_feature_attributions(IG_all_attributions, config)
        shap_feats = aggregate_feature_attributions(SHAP_all_attributions, config)
        dl_feats = aggregate_feature_attributions(LayerDeepLift_all_attributions, config)

        # Step 2: Combine into DataFrame
        df_feat_importance = pd.DataFrame({
            "IG": ig_feats,
            "LIG": lig_feats,
            "SHAP": shap_feats,
            "LayerDeepLift": dl_feats
        }).T

        df_feat_importance = df_feat_importance.T
        df_feat_importance = df_feat_importance.sort_values(by="LIG", ascending=False)

        # Step 3: Plot Top 10 Features
        top_k = 10
        top_features = df_feat_importance.head(top_k)
        top_features_T = top_features.T

        fig, ax = plt.subplots(figsize=(14, 6))
        top_features_T.plot(kind='bar', ax=ax, width=0.8)

        ax.set_title(f"Top {top_k} Time-Series Features Attributions from All XAI Methods", fontsize=14, weight='bold')
        ax.set_ylabel("Mean Absolute Attribution", fontsize=12)
        ax.set_xlabel("XAI Method", fontsize=12)
        ax.grid(axis='y')
        ax.legend(title="Feature", bbox_to_anchor=(1.01, 1), loc='upper left')
        ax.tick_params(axis='x', rotation=0)

        plt.tight_layout()

        return fig, top_features_T

    except Exception as e:
        print(f"Error in generate_top10_features_all_methods: {e}")
        return None, pd.DataFrame()
    
def compute_temporal_profile(attributions, method_name, normalize=True):
    """Computes mean attribution per timestep, optionally normalized per method.

    - attributions: List of patient-level attributions.
    - method_name: Name of the method (string).
    - normalize: If True, normalize the time-series attribution profile per method.

    Returns:
        DataFrame with columns: Timestep, Attribution, Method.
    """
    all_ts_attr = np.array([
        np.abs(sample['attri_X']) for sample in attributions
    ])  # shape: (N_patients, T_timesteps, F_features)

    per_timestep = np.mean(all_ts_attr, axis=2)  # shape: (N, T)
    avg_profile = np.mean(per_timestep, axis=0)  # shape: (T,)

    if normalize:
        max_val = np.max(avg_profile)
        if max_val > 0:
            avg_profile /= max_val  # Normalize to [0, 1]

    return pd.DataFrame({
        "Timestep": list(range(len(avg_profile))),
        "Attribution": avg_profile,
        "Method": method_name
    })

def generate_temporal_evolution_plot(normalize=True):

    """
    Generates a line plot showing temporal evolution of time-series attributions.

    - normalize: If True, normalizes attribution profiles per method (emphasizing structure over magnitude).

    Returns:
        (fig, df_temporal)
    """
    try:
        profiles = []
        profiles.append(compute_temporal_profile(IG_all_attributions, "IG", normalize=normalize))
        profiles.append(compute_temporal_profile(LIG_all_attributions, "LIG", normalize=normalize))
        profiles.append(compute_temporal_profile(SHAP_all_attributions, "SHAP", normalize=normalize))
        profiles.append(compute_temporal_profile(LayerDeepLift_all_attributions, "LayerDeepLift", normalize=normalize))

        df_temporal = pd.concat(profiles)

        fig, ax = plt.subplots(figsize=(12, 6))
        sns.lineplot(data=df_temporal, x="Timestep", y="Attribution", hue="Method", marker="o", ax=ax)

        plot_title = ("Normalized Temporal Evolution of Time-Series Attribution "
                      )
        ax.set_title(plot_title, fontsize=14, weight='bold')
        ax.set_xlabel("Time Step (Hour)", fontsize=12)
        ax.set_ylabel("Normalized Mean Attribution", fontsize=12)
        ax.grid(True)
        ax.legend(title="XAI Method")
        plt.tight_layout()

        return fig, df_temporal

    except Exception as e:
        print(f"Error in generate_temporal_evolution_plot: {e}")
        return None, pd.DataFrame()

def plot_temporal_attribution_heatmap(method_name, normalize=True):
    """
    Plots a heatmap showing temporal attribution evolution per feature for the selected method.
    
    Args:
        method_name (str): e.g. 'SHAP', 'IG', etc.
        normalize (bool): whether to normalize values across each feature
    
    Returns:
        matplotlib.figure.Figure: The generated heatmap plot
    """
    try:
        # Extract patient attributions
        attributions_list = method_map[method_name]
        attri_X_all = np.array([np.abs(p["attri_X"]) for p in attributions_list])  # Shape: (N, T, F)
        
        feature_matrix = []
        feature_labels = []
        idx = 0
        
        for feature_name in config["id_to_channel"]:
            if config["is_categorical_channel"].get(feature_name, False):
                num_cat = len(config["possible_values"][feature_name])
                attr = attri_X_all[:, :, idx:idx+num_cat].mean(axis=2)
                idx += num_cat
            else:
                attr = attri_X_all[:, :, idx]
                idx += 1
            
            avg_curve = attr.mean(axis=0)  # Average over patients → shape: (T,)
            
            if normalize:
                max_val = np.max(avg_curve)
                if max_val > 0:
                    avg_curve /= max_val
            
            feature_matrix.append(avg_curve)
            feature_labels.append(feature_name)
        
        feature_matrix = np.array(feature_matrix)  # shape: (F, T)
        
        # Plot
        fig, ax = plt.subplots(figsize=(16, 10))
        sns.heatmap(
            feature_matrix, 
            xticklabels=4, 
            yticklabels=feature_labels, 
            cmap="YlGnBu", 
            linewidths=0.3, 
            ax=ax
        )
        
        note = "(Normalized)" if normalize else ""
        ax.set_title(f"Temporal Attribution Heatmap per Feature {note} ({method_name})", fontsize=14, weight='bold')
        ax.set_xlabel("Time Step (Hour)")
        ax.set_ylabel("Feature")
        plt.tight_layout()
        
        return fig
    
    except Exception as e:
        print(f"Error generating heatmap: {e}")
        return None  

def generate_wordcloud_for_method(method_name):
    """
    Generates a word cloud figure for text attributions from a given method.
    """
    try:
        attributions = method_map[method_name]
        word_attr = defaultdict(list)

        for idx, p in enumerate(attributions):
            token_ids = all_input_ids[idx]  # Now uses preloaded input_ids
            token_attributions = np.abs(p['attri_txt_mean'])

            tokens = tokenizer.convert_ids_to_tokens(token_ids)
            for tok, score in zip(tokens, token_attributions):
                if tok in ["[CLS]", "[SEP]", "[PAD]", "[UNK]"]:
                    continue
                if tok.startswith("##") and len(word_attr) > 0:
                    last_key = list(word_attr.keys())[-1]
                    word_attr[last_key][-1] += score
                else:
                    word_attr[tok].append(score)

        if not word_attr:
            print(f"⚠️ No tokens found for word cloud generation.")
            return None

        avg_word_attr = {word: np.mean(scores) for word, scores in word_attr.items() if word.isalpha()}

        wc = WordCloud(width=1000, height=600, background_color="white", colormap='viridis')
        wc.generate_from_frequencies(avg_word_attr)

        fig, ax = plt.subplots(figsize=(14, 8))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(f"Most Influential Words in Clinical Notes ({method_name})", fontsize=16)
        plt.tight_layout()

        return fig

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error generating word cloud for {method_name}: {e}")
        return None




