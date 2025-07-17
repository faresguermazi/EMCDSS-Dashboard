# load_patient_data.py

import h5py
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import pandas as pd
import io

from transformers import AutoTokenizer

# Paths
HDF5_PATH = "/netscratch/fguermazi/XAI/data/mortality/splits.hdf5" #CHANGE PATH IF NEEDED
CONFIG_PATH = "/netscratch/fguermazi/XAI/data/mortality/discretizer_config.json" #CHANGE PATH IF NEEDED
ENCODER_PATH = "/netscratch/fguermazi/XAI/data/mortality/onehotencoder.pkl" #CHANGE PATH IF NEEDED
GROUP = "with_notes" #CHANGE PATH IF NEEDED

# Load once
one_hotencoder = joblib.load(ENCODER_PATH)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

def detokenize(tokens):
    sentence = []
    for token in tokens:
        if token in ["[CLS]", "[SEP]", "[PAD]", "[UNK]"]:
            continue
        if token.startswith("##"):
            if sentence:
                sentence[-1] += token[2:]
        else:
            sentence.append(token)
    return " ".join(sentence)

def decode_static_features(static_features):
    gender_categories = one_hotencoder.categories_[0]
    ethnicity_categories = one_hotencoder.categories_[1]
    num_gender = len(gender_categories)
    num_ethnicity = len(ethnicity_categories)

    gender_one_hot = static_features[:num_gender]
    ethnicity_one_hot = static_features[num_gender:num_gender + num_ethnicity]
    age = int(static_features[-1])

    gender = gender_categories[np.argmax(gender_one_hot)]
    ethnicity = ethnicity_categories[np.argmax(ethnicity_one_hot)]
    return gender, ethnicity, age

def decode_time_series(time_series_data,config):
    rows = []
    idx = 0
    for feature_name in config["id_to_channel"]:
        if feature_name in config["is_categorical_channel"] and config["is_categorical_channel"][feature_name]:
            num_categories = len(config["possible_values"][feature_name])
            category_values = time_series_data[:, idx:idx + num_categories]
            decoded_values = [config["possible_values"][feature_name][np.argmax(timestep)] for timestep in category_values]
            idx += num_categories
        else:
            decoded_values = time_series_data[:, idx]
            idx += 1

        formatted_values = [f"{v:.2f}" if isinstance(v, (int, float)) else v for v in decoded_values]
        display_value = formatted_values[0] if len(set(formatted_values)) == 1 else ", ".join(formatted_values)
        
        rows.append([feature_name, display_value])
    return rows

def get_patient_info_by_index(index):
    with h5py.File(HDF5_PATH, "r") as hf:
        test_data = hf[GROUP]["test"]

        icu_id = test_data["icu"][index]
        input_ids = test_data["input_ids"][index]
        static_features = test_data["s"][index]
        time_series_data = test_data["X"][index]
        true_label = test_data["label"][index]

    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    text = detokenize(tokens)
    gender, ethnicity, age = decode_static_features(static_features)
    decoded_time_series = decode_time_series(time_series_data,config)

    return {
        "icu_id": int(icu_id),
        "true_label": int(true_label),
        "text": text,
        "gender": gender,
        "ethnicity": ethnicity,
        "age": age,
        "static_features": static_features.tolist(),
        "time_series_shape": time_series_data.shape,
        "time_series": decoded_time_series
    }


def plot_time_series_evolution(patient_index, feature_name):
    try:
        info = get_patient_info_by_index(patient_index)
        time_series_data = info["time_series"]

        matching_row = [row for row in time_series_data if row[0] == feature_name]
        if not matching_row:
            raise ValueError(f"Feature '{feature_name}' not found.")

        values_str = matching_row[0][1]
        raw_values = values_str.split(", ")

        is_categorical = config["is_categorical_channel"].get(feature_name, False)
        fig, ax = plt.subplots(figsize=(10, 5))

        if not is_categorical:
            values = [float(v) for v in raw_values]
            if len(set(values)) == 1:
                values = [values[0]] * 48  
            ax.plot(values, marker='o', label=feature_name)
            ax.set_ylabel("Value")
        else:
            possible_vals = config["possible_values"].get(feature_name, [])
            val_to_num = {val: idx for idx, val in enumerate(possible_vals)}
            numeric_values = [val_to_num.get(v, np.nan) for v in raw_values]
            if len(set(numeric_values)) == 1:
                numeric_values = [numeric_values[0]] * 48
            if any(np.isnan(numeric_values)):
                raise ValueError(f"Some categorical values could not be mapped using config for '{feature_name}'.")

            ax.step(range(len(numeric_values)), numeric_values, where='post', label=feature_name)
            ax.set_yticks(range(len(possible_vals)))
            ax.set_yticklabels(possible_vals)
            prev_label = None
            for i, (num, label) in enumerate(zip(numeric_values, raw_values)):
                if label != prev_label:
                    ax.text(i, num + 0.2, label, ha='center', va='bottom', fontsize=7, rotation=30)
                prev_label = label

            ax.set_ylabel("Category Index")

        ax.set_title(f"Evolution of '{feature_name}' Over 48 Hours")
        ax.set_xlabel("Time Step (Hour)")
        ax.grid(True)
        ax.legend()

        return fig  # Return the actual figure object

    
    except Exception as e:
        return f"<div>Error generating plot: {e}</div>"

def plot_compare_time_series_evolution(test_info, train_info, feature_name):
    try:
        test_ts = test_info["time_series"]
        train_ts = train_info["time_series"]

        def extract_values(data):
            row = [r for r in data if r[0] == feature_name]
            if not row:
                return None
            return row[0][1].split(", ")

        test_vals_raw = extract_values(test_ts)
        train_vals_raw = extract_values(train_ts)
        if test_vals_raw is None or train_vals_raw is None:
            raise ValueError(f"Feature '{feature_name}' not found.")

        is_categorical = config["is_categorical_channel"].get(feature_name, False)

        fig, ax = plt.subplots(figsize=(10, 5))

        if not is_categorical:
            test_vals = [float(v) for v in test_vals_raw]
            train_vals = [float(v) for v in train_vals_raw]

            if len(set(train_vals)) == 1:
                train_vals = [train_vals[0]] * 48
            if len(set(test_vals)) == 1:
                test_vals = [test_vals[0]] * 48

            ax.plot(test_vals, label="Test", color="blue", marker="o")
            ax.plot(train_vals, label="Train", color="orange", linestyle="--", marker="x")
            ax.set_ylabel("Value")
        else:
            possible_vals = config["possible_values"].get(feature_name, [])
            val_to_num = {val: idx for idx, val in enumerate(possible_vals)}

            test_numeric = [val_to_num.get(v, np.nan) for v in test_vals_raw]
            train_numeric = [val_to_num.get(v, np.nan) for v in train_vals_raw]

            if len(set(train_numeric)) == 1:
                train_numeric = [train_numeric[0]] * 48
            if len(set(test_numeric)) == 1:
                test_numeric = [test_numeric[0]] * 48

            ax.step(range(len(test_numeric)), test_numeric, where='post', label="Test", color="blue")
            ax.step(range(len(train_numeric)), train_numeric, where='post', label="Train", color="orange", linestyle="--")
            ax.set_yticks(range(len(possible_vals)))
            ax.set_yticklabels(possible_vals)
            ax.set_ylabel("Category Index")

        ax.set_title(f"Compare '{feature_name}' (Test vs. Train)")
        ax.set_xlabel("Time Step (Hour)")
        ax.grid(True)
        ax.legend()
        plt.tight_layout()

        buf = BytesIO()
        fig.savefig(buf, format="png")
        plt.close(fig)
        img_str = base64.b64encode(buf.getvalue()).decode()
        return f"<img src='data:image/png;base64,{img_str}'/>"

    except Exception as e:
        return f"<div>Error: {e}</div>"

def plot_time_series_base64(patient_index, feature_name):
    fig = plot_time_series_evolution(patient_index, feature_name)
    if fig is None:
        return None

    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_b64
