# local_explainer.py

import json
import pandas as pd
import numpy as np
import h5py
from transformers import AutoTokenizer
from utils.load_patient_data import get_patient_info_by_index, one_hotencoder, config, HDF5_PATH, GROUP
import string
from collections import Counter
from utils.load_patient_data import get_patient_info_by_index,decode_static_features
from nltk.corpus import stopwords
import re
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter



# Define paths
LIG_attribution_file = 'utils_generated_files/LIG_attribution_output_MBertLstm.jsonl' 
IG_attribution_file = 'utils_generated_files/IG_attribution_output_MBertLstm.jsonl' 
LayerDeepLift_attribution_file = 'utils_generated_files/DeepLift_attribution_output_MBertLstm.jsonl' 
SHAP_attribution_file = 'utils_generated_files/SHAP_attribution_output_MBertLstm.jsonl' 
hdf5_path = "/netscratch/fguermazi/XAI/data/mortality/splits.hdf5" #CHANGE PATH IF NEEDED

# Load attribution files
def load_jsonl(file_path):
    attributions = []
    with open(file_path, 'r') as f:
        for line in f:
            attributions.append(json.loads(line))
    return attributions

LIG_all_attributions = load_jsonl(LIG_attribution_file)
LayerDeepLift_all_attributions = load_jsonl(LayerDeepLift_attribution_file)
SHAP_all_attributions = load_jsonl(SHAP_attribution_file)
IG_all_attributions = load_jsonl(IG_attribution_file)

# Mapping for method selection
method_map = {
    "IG": IG_all_attributions,
    "LIG": LIG_all_attributions,
    "LayerDeepLift": LayerDeepLift_all_attributions,
    "SHAP": SHAP_all_attributions
}

def get_mean_static_attributions(patient_index, method_name):
    try:
        # Get patient info
        patient_info = get_patient_info_by_index(patient_index)
        static_features = patient_info['static_features']
        gender = patient_info['gender']
        ethnicity = patient_info['ethnicity']

        # Fetch attributions for the selected method
        attributions = method_map[method_name]
        attri_s = np.array(attributions[patient_index]['attri_s'])

        # Decode static features using one-hot encoder
        gender_categories = one_hotencoder.categories_[0]
        ethnicity_categories = one_hotencoder.categories_[1]
        num_gender = len(gender_categories)
        num_ethnicity = len(ethnicity_categories)

        # Find the index of the actual value within the one-hot categories
        gender_active_index = np.where(gender_categories == gender)[0][0]
        ethnicity_active_index = np.where(ethnicity_categories == ethnicity)[0][0]

        # Calculate mean attribution for each category in one-hot encoded features
        gender_attribution = np.mean(attri_s[:num_gender])
        ethnicity_attribution = np.mean(attri_s[num_gender:num_gender + num_ethnicity])
        age_attribution = np.mean(attri_s[-1])

        # Extract active feature attributions based on actual value
        gender_active_attr = attri_s[gender_active_index]
        ethnicity_active_attr = attri_s[num_gender + ethnicity_active_index]

        # Return the mean attributions and active attributions as a tuple
        return (gender_attribution, ethnicity_attribution, age_attribution), (gender_active_attr, ethnicity_active_attr, age_attribution)
    except Exception as e:
        print(f"Error in get_mean_static_attributions: {e}")
        return (None, None, None), (None, None, None)

def get_mean_time_series_attributions(patient_index, method_name):
    try:
        # Get patient info
        patient_info = get_patient_info_by_index(patient_index)
        time_series_data = np.array(patient_info['time_series'])

        # Fetch attributions for the selected method
        attributions = method_map[method_name]
        attri_X = np.array(attributions[patient_index]['attri_X'])

        # Extract time-series data and attributions
        feature_data = []
        idx = 0

        # Loop through features based on the configuration
        for feature_name in config["id_to_channel"]:
            if feature_name in config["is_categorical_channel"] and config["is_categorical_channel"][feature_name]:
                # Handle categorical features
                num_categories = len(config["possible_values"][feature_name])
                category_values = time_series_data[:, idx:idx + num_categories]
                decoded_values = [config["possible_values"][feature_name][np.argmax(timestep)] for timestep in category_values]
                feature_attributions = attri_X[:, idx:idx + num_categories]
                avg_attr = np.mean(feature_attributions)
                idx += num_categories
            else:
                # Handle continuous features
                decoded_values = time_series_data[:, idx]
                feature_attributions = attri_X[:, idx]
                avg_attr = np.mean(feature_attributions)
                idx += 1

            # Format the mean value
            formatted_value = f"{np.mean(decoded_values):.3f}" if isinstance(np.mean(decoded_values), (int, float)) else decoded_values[-1]

            # Append the feature name, mean attribution, and mean value
            feature_data.append([feature_name, formatted_value, f"{avg_attr:.4e}"])

        # Convert to DataFrame without sorting or ranking
        feature_df = pd.DataFrame(feature_data, columns=["Feature", "Mean value", "Mean Attribution"])

        return feature_df
    except Exception as e:
        print(f"Error in get_mean_time_series_attributions: {e}")
        return pd.DataFrame()

def get_textual_attributions(patient_index, method_name, percentile):
    try:
        attributions = method_map[method_name]
        attri_txt = np.array(attributions[patient_index]['attri_txt_mean'])
        patient_info = get_patient_info_by_index(patient_index)

        tokens = patient_info['text'].split()
        threshold = calculate_threshold(attri_txt, percentile)

        highlighted_text = ""
        for word, score in zip(tokens, attri_txt):
            color = "lightgray" if abs(score) < threshold else "red" if score < 0 else "green"
            highlighted_text += f"<span style='background-color:{color}; padding:3px;'>{word}</span> "

        return highlighted_text
    except Exception as e:
        print(f"Error in get_textual_attributions: {e}")
        return None

def get_combined_time_series_attributions(patient_index, method_name=None):
    try:
        with h5py.File(HDF5_PATH, "r") as hf:
            data = hf[GROUP]["test"]
            X = np.array(data["X"][patient_index])

        # Define attribution dictionary based on method_name
        all_attr_dict = {
            "IG": np.array(IG_all_attributions[patient_index]["attri_X"]),
            "LIG": np.array(LIG_all_attributions[patient_index]["attri_X"]),
            "LayerDeepLift": np.array(LayerDeepLift_all_attributions[patient_index]["attri_X"]),
            "SHAP": np.array(SHAP_all_attributions[patient_index]["attri_X"])
        }

        # If a specific method is selected, only include that one
        if method_name and method_name in all_attr_dict:
            attr_dict = {method_name: all_attr_dict[method_name]}
        else:
            attr_dict = all_attr_dict

        feature_data = []
        idx = 0
        for feature_name in config["id_to_channel"]:
            if config["is_categorical_channel"].get(feature_name, False):
                num_categories = len(config["possible_values"][feature_name])
                values = X[:, idx:idx + num_categories]
                decoded_values = [config["possible_values"][feature_name][np.argmax(t)] for t in values]
                mean_val = decoded_values[-1]
                attrs = {k: np.mean(v[:, idx:idx + num_categories]) for k, v in attr_dict.items()}
                idx += num_categories
            else:
                values = X[:, idx]
                mean_val = np.mean(values)
                attrs = {k: np.mean(v[:, idx]) for k, v in attr_dict.items()}
                idx += 1

            row = [feature_name, f"{mean_val:.3f}" if isinstance(mean_val, (int, float)) else mean_val]
            for method in attr_dict:
                row.append(f"{attrs[method]:.4e}")

            feature_data.append(row)

        columns = ["Feature", "Mean Value"] + list(attr_dict.keys())
        return pd.DataFrame(feature_data, columns=columns)

    except Exception as e:
        print(f"Error in get_combined_time_series_attributions: {e}")
        return pd.DataFrame()

stop_words = set(stopwords.words('english'))
punct_table = str.maketrans('', '', string.punctuation)

def get_top_contributing_words_table(patient_index, method_name, top_n=10):
    try:
        attributions = method_map[method_name]
        attri_txt = np.array(attributions[patient_index]['attri_txt_mean'])

        # Get raw text and  with regex to remove punctuation properly
        patient_info = get_patient_info_by_index(patient_index)
        raw_text = patient_info["text"].lower()
        tokens = re.findall(r'\b\w+\b', raw_text)  # Extract word tokens only

        # Filter tokens: remove stop words and empty strings
        filtered = [(w, s) for w, s in zip(tokens, attri_txt)
                    if w not in stop_words and w.strip() != ""]

        # Sort by absolute attribution value (descending)
        filtered.sort(key=lambda x: abs(x[1]), reverse=True)

        # Select top-N words
        top_filtered = filtered[:top_n]
        words = [w for w, _ in top_filtered]
        scores = [f"{s:.2e}" for _, s in top_filtered]

        return pd.DataFrame({
            "Word": words,
            "Attribution": scores
        })

    except Exception as e:
        return pd.DataFrame({
            "Word": ["Error"],
            "Attribution": [str(e)]
        })

def get_detailed_static_attributions(patient_index, method_name=None):
    try:
        patient_info = get_patient_info_by_index(patient_index)
        static_features = np.array(patient_info['static_features'])

        # Decode gender, ethnicity, and age using existing function
        gender, ethnicity, age = decode_static_features(static_features)

        # One-hot categories
        gender_categories = one_hotencoder.categories_[0]
        ethnicity_categories = one_hotencoder.categories_[1]
        num_gender = len(gender_categories)
        num_ethnicity = len(ethnicity_categories)

        feature_names = []
        feature_values = []

        # Gender
        gender_one_hot = static_features[:num_gender]
        for i, cat in enumerate(gender_categories):
            feature_names.append(f"Gender: {cat}")
            feature_values.append("✓" if np.argmax(gender_one_hot) == i else "✗")

        # Ethnicity
        ethnicity_one_hot = static_features[num_gender:num_gender + num_ethnicity]
        for i, cat in enumerate(ethnicity_categories):
            feature_names.append(f"Ethnicity: {cat}")
            feature_values.append("✓" if np.argmax(ethnicity_one_hot) == i else "✗")

        # Age
        feature_names.append("Age")
        feature_values.append(age)

        data = {
            "Feature": feature_names,
            "Value": feature_values
        }

        if method_name is None or method_name == "Compare all methods":
            for method in ["IG", "LIG", "LayerDeepLift", "SHAP"]:
                attributions = method_map[method]
                attri_s = np.array(attributions[patient_index]['attri_s'])

                gender_attr = list(attri_s[:num_gender])
                ethnicity_attr = list(attri_s[num_gender:num_gender + num_ethnicity])
                age_attr = [attri_s[-1]]

                all_attrs = gender_attr + ethnicity_attr + age_attr
                data[method] = [f"{val:.2e}" for val in all_attrs]
        else:
            attributions = method_map[method_name]
            attri_s = np.array(attributions[patient_index]['attri_s'])

            gender_attr = list(attri_s[:num_gender])
            ethnicity_attr = list(attri_s[num_gender:num_gender + num_ethnicity])
            age_attr = [attri_s[-1]]

            all_attrs = gender_attr + ethnicity_attr + age_attr
            data["Attribution"] = [f"{val:.2e}" for val in all_attrs]

        return pd.DataFrame(data)

    except Exception as e:
        return pd.DataFrame({
            "Feature": ["Error"],
            "Value": [""],
            "Attribution": [str(e)]
        })
    
def plot_patient_time_series_attributions(patient_index, method_name="SHAP", top_k=6):
    """
    Plots the top K time-series feature attributions over time for a given patient and method.
    Returns a matplotlib figure object (for Streamlit integration).
    """
    try:
        # Fetch the attributions for the selected method
        attributions = method_map.get(method_name)
        if attributions is None:
            print(f" Unknown method: {method_name}. Choose from {list(method_map.keys())}")
            return None

        attri_X = np.array(attributions[patient_index]['attri_X'])  # Shape: (timesteps, features)

        # Load patient time-series data
        with h5py.File(hdf5_path, "r") as hf:
            data = hf[GROUP]["test"]
            time_series_data = np.array(data["X"][patient_index])
            icu_id = data["icu"][patient_index]
            label = data["label"][patient_index]

        # Decode feature attributions over time
        feature_attr = {}
        idx = 0
        for feature in config["id_to_channel"]:
            if config["is_categorical_channel"].get(feature, False):
                n_cat = len(config["possible_values"][feature])
                attr = attri_X[:, idx:idx+n_cat].mean(axis=1)  # mean across categories
                idx += n_cat
            else:
                attr = attri_X[:, idx]
                idx += 1
            feature_attr[feature] = attr

        # Sort features by mean attribution magnitude
        sorted_features = sorted(feature_attr.items(), key=lambda x: np.mean(np.abs(x[1])), reverse=True)
        top_features = sorted_features[:top_k]

        # Plot
        fig, ax = plt.subplots(figsize=(8, 4))
        for feature, attr in top_features:
            ax.plot(attr, label=feature)

        ax.set_title(f"Top {top_k} Feature Attributions Over Time\n(Patient {patient_index}, Method: {method_name})")
        ax.set_xlabel("Time Step (Hour)")
        ax.set_ylabel("Attribution Score")
        ax.legend()
        ax.grid(True)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2e'))  
        fig.tight_layout()

        return fig

    except Exception as e:
        print(f"Error in plot_patient_time_series_attributions: {e}")
        return None

def plot_feature_attribution_over_time(patient_index, feature_name, method_name):
    """
    Plots the time-series attribution for a specific feature and a single XAI method.
    Returns a matplotlib figure object for Streamlit integration.
    """
    try:
        # Fetch the attributions for the selected method
        attributions = method_map.get(method_name)
        if attributions is None:
            print(f"Unknown method: {method_name}. Choose from {list(method_map.keys())}")
            return None

        attri_X = np.array(attributions[patient_index]['attri_X'])  # Shape (timesteps, features)

        # Locate feature index
        idx = 0
        feature_pos = None
        for name in config["id_to_channel"]:
            if name == feature_name:
                feature_pos = idx
                break
            if config["is_categorical_channel"].get(name, False):
                idx += len(config["possible_values"][name])
            else:
                idx += 1

        if feature_pos is None:
            print(f"Feature '{feature_name}' not found.")
            return None

        # Extract attributions
        if config["is_categorical_channel"].get(feature_name, False):
            n_cat = len(config["possible_values"][feature_name])
            attr_vals = attri_X[:, feature_pos:feature_pos + n_cat].mean(axis=1)
        else:
            attr_vals = attri_X[:, feature_pos]

        # Plot
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(attr_vals, label=method_name)
        ax.set_title(f"Attribution of '{feature_name}' Over Time (Patient {patient_index})")
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Attribution Score")
        ax.legend()
        ax.grid(True)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2e'))  
        fig.tight_layout()

        return fig

    except Exception as e:
        print(f"Error in plot_feature_attribution_over_time: {e}")
        return None

def plot_feature_over_time_all_methods(patient_index, feature_name):
    """
    Plots the time-series attribution of a selected feature for all methods.
    Returns a matplotlib figure object for Streamlit integration.
    """
    try:
        # Create dictionary of method name -> attributions
        method_dict = {
            "IG": IG_all_attributions,
            "LIG": LIG_all_attributions,
            "LayerDeepLift": LayerDeepLift_all_attributions,
            "SHAP": SHAP_all_attributions
        }

        # Locate feature index
        idx = 0
        feature_pos = None
        for name in config["id_to_channel"]:
            if name == feature_name:
                feature_pos = idx
                break
            if config["is_categorical_channel"].get(name, False):
                idx += len(config["possible_values"][name])
            else:
                idx += 1

        if feature_pos is None:
            print(f"Feature '{feature_name}' not found.")
            return None

        # Plot
        fig, ax = plt.subplots(figsize=(8, 4))
        for method, attributions in method_dict.items():
            attr = np.array(attributions[patient_index]['attri_X'])  # shape (48, num_features)
            if config["is_categorical_channel"].get(feature_name, False):
                n_cat = len(config["possible_values"][feature_name])
                attr_vals = attr[:, feature_pos:feature_pos + n_cat].mean(axis=1)
            else:
                attr_vals = attr[:, feature_pos]
            ax.plot(attr_vals, label=method)

        ax.set_title(f"Attribution of '{feature_name}' Over Time (Patient {patient_index})")
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Attribution Score")
        ax.legend()
        ax.grid(True)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2e'))  

        fig.tight_layout()

        return fig

    except Exception as e:
        print(f"Error in plot_feature_over_time_all_methods: {e}")
        return None

def plot_time_window_attributions(patient_index, method_name, num_windows=12):
    try:
        # Fetch attributions
        attributions = method_map[method_name]
        attri_X = np.array(attributions[patient_index]['attri_X'])  # shape (48, num_features)
        num_timesteps, num_features = attri_X.shape

        # Calculate window size
        window_size = num_timesteps // num_windows

        # Initialize array to store mean absolute attributions per window
        window_means = []

        for w in range(num_windows):
            start = w * window_size
            end = (w + 1) * window_size if w < num_windows - 1 else num_timesteps
            window_attr = attri_X[start:end, :]
            mean_attr = np.mean(np.abs(window_attr))  # Average over features and time steps
            window_means.append(mean_attr)

        # Plot
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(range(1, num_windows + 1), window_means, color='skyblue', edgecolor='black')
        ax.set_xlabel("Time Window (4 hours each)")
        ax.set_ylabel("Mean Absolute Attribution")
        ax.set_title(f"Mean Attribution per Time Window\n(Patient {patient_index}, Method: {method_name})")
        ax.set_xticks(range(1, num_windows + 1))
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()

        return fig

    except Exception as e:
        print(f"Error in plot_time_window_attributions: {e}")
        return None

def calculate_threshold(attributions, percentile):
    abs_attributions = np.abs(np.array(attributions).flatten())
    threshold = np.percentile(abs_attributions, percentile)
    threshold = max(threshold, np.max(abs_attributions) * 0.005)
    return threshold

def visualize_text_attributions_with_auto_threshold(patient_index, method_name, percentile):
    """
    Highlights words in the patient's clinical text based on their attributions using dynamic thresholding.
    Words with absolute attributions below the threshold are yellow (neutral),
    stronger positive/negative attributions are green/red.
    """

    try:
        # Fetch attributions
        attributions = method_map[method_name]
        attri_txt = np.array(attributions[patient_index]['attri_txt_mean'])

        # Fetch tokens using patient index
        patient_info = get_patient_info_by_index(patient_index)
        text = patient_info["text"]
        tokens = text.split()

        # Ensure the number of tokens matches attributions
        if len(tokens) != len(attri_txt):
            print(f"⚠️ Warning: Number of tokens ({len(tokens)}) does not match attributions length ({len(attri_txt)}).")
            attri_txt = attri_txt[:len(tokens)]  # Truncate if needed

        # Normalize attributions
        max_abs = np.max(np.abs(attri_txt))
        normalized_attributions = attri_txt / max_abs if max_abs > 0 else attri_txt

        # Compute threshold
        threshold = calculate_threshold(normalized_attributions, percentile)

        # Generate colored text
        def get_color(score):
            base_brightness = 80
            intensity = int(abs(score) * 175) + base_brightness
            intensity = min(255, intensity)
            if abs(score) < threshold:
                return "rgb(255, 255, 128)"  # Yellow
            elif score > 0:
                return f"rgb(0, {intensity}, 0)"  # Green
            else:
                return f"rgb({intensity}, 0, 0)"  # Red

        colored_text = ""
        for word, score in zip(tokens, normalized_attributions):
            color = get_color(float(score))
            colored_text += f'<span style="background-color: {color}; color: black; padding:4px; margin:2px; border-radius:3px; display: inline-block;"> {word} </span> '

        explanation = f"""
                    Intensity of colors reflects the magnitude of the word's contribution. 
                    Words with |attribution| < {threshold:.5f} (calculated threshold) are considered neutral.
                """

        # Return as HTML
        html_output = f"""
        <div style="background-color: white; padding: 15px; border: 1px solid #ddd; border-radius: 8px; line-height: 1.6;">
            {colored_text}
            {explanation}
        </div>
        """
        return html_output, threshold

    except Exception as e:
        print(f"Error in visualize_text_attributions_with_auto_threshold: {e}")
        return "<div>Error displaying text attributions.</div>", None
