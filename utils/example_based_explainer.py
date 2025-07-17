# example_based_explainer.py

import json
import numpy as np
import torch
import h5py
from sklearn.neighbors import NearestNeighbors
from transformers import AutoTokenizer
from utils.load_patient_data import decode_static_features, detokenize,decode_time_series, config, GROUP, HDF5_PATH
import re
from nltk.corpus import stopwords
import string

#GROUP_NAME = "with_notes"
TOKENIZER = AutoTokenizer.from_pretrained("bert-base-uncased")
# Load train embeddings
with open("utils_generated_files/train_data.json", "r") as f: #CHANGE PATH FOR CLUSTER
    train_data = json.load(f)
# Load test embeddings
with open("utils_generated_files/test_data.json", "r") as f: #CHANGE PATH FOR CLUSTER
    test_data = json.load(f)


train_icu_ids = np.array([entry["icu"] for entry in train_data])
train_labels = np.array([entry["label"] for entry in train_data])
train_embeds = np.array([entry["embedding"] for entry in train_data])

test_icu_ids = np.array([entry["icu"] for entry in test_data])
test_labels = np.array([entry["label"] for entry in test_data])
test_embeds = np.array([entry["embedding"] for entry in test_data])
# KNN search
knn = NearestNeighbors(n_neighbors=5, metric='euclidean')
knn.fit(train_embeds)
test_knn = NearestNeighbors(n_neighbors=4, metric='euclidean')  # +1 for self
test_knn.fit(test_embeds)

stop_words = set(stopwords.words('english'))

def clean_tokens(tokens):
    # Remove punctuation and lowercase
    tokens = [re.sub(rf"[{string.punctuation}]", "", t.lower()) for t in tokens]
    # Remove empty strings and stopwords
    return [t for t in tokens if t and t not in stop_words]


def find_train_neighbors(test_embedding: np.ndarray):
    """
    Find top 5 most similar train samples to given test embedding.
    """
    distances, indices = knn.kneighbors(test_embedding.reshape(1, -1))
    return [{
        "rank": rank + 1,
        "train_index": int(train_idx),
        "icu_id": int(train_icu_ids[train_idx]),
        "label": int(train_labels[train_idx]),
        "distance": float(dist)
    } for rank, (train_idx, dist) in enumerate(zip(indices[0], distances[0]))]


def get_train_patient_info(train_index: int):
    """
    Fetch decoded patient info from HDF5 for a train sample.
    """
    with h5py.File(HDF5_PATH, "r") as hf:
        train_group = hf[GROUP]["train"]
        icu_id = train_group["icu"][train_index]
        input_ids = train_group["input_ids"][train_index]
        static_features = train_group["s"][train_index]
        time_series = train_group["X"][train_index]
        label = train_group["label"][train_index]

    tokens = TOKENIZER.convert_ids_to_tokens(input_ids)
    text = detokenize(tokens)
    gender, ethnicity, age = decode_static_features(static_features)
    decoded_ts = decode_time_series(np.array(time_series),config)

    return {
        "icu_id": int(icu_id),
        "label": int(label),
        "text": text,
        "gender": gender,
        "ethnicity": ethnicity,
        "age": age,
        "time_series": decoded_ts
    }

def compare_patients(test_index: int, train_index: int):
    """
    Compare test vs. train patient info (token overlap, static features, TS summary).
    """
    with h5py.File(HDF5_PATH, "r") as hf:
        test = hf[GROUP]["test"]
        train = hf[GROUP]["train"]

        # Text token comparison
        
        test_ids = test["input_ids"][test_index]
        train_ids = train["input_ids"][train_index]
        tokens_test = TOKENIZER.convert_ids_to_tokens(test_ids)
        tokens_train = TOKENIZER.convert_ids_to_tokens(train_ids)

        words_test = clean_tokens(detokenize(tokens_test).split())
        words_train = clean_tokens(detokenize(tokens_train).split())

        common_tokens = set(words_test).intersection(words_train)
        
        # Static feature comparison
        static_test = test["s"][test_index]
        static_train = train["s"][train_index]
        g_t, e_t, a_t = decode_static_features(static_test)
        g_r, e_r, a_r = decode_static_features(static_train)

        # Time-series comparison
        ts_test = test["X"][test_index]
        ts_train = train["X"][train_index]

        ts_summary = []
        idx = 0
        for feat in config["id_to_channel"]:
            if config["is_categorical_channel"].get(feat, False):
                n_cat = len(config["possible_values"][feat])
                t_vals = ts_test[:, idx:idx + n_cat]
                r_vals = ts_train[:, idx:idx + n_cat]
                t_dec = [config["possible_values"][feat][np.argmax(x)] for x in t_vals]
                r_dec = [config["possible_values"][feat][np.argmax(x)] for x in r_vals]
                match = sum(t == r for t, r in zip(t_dec, r_dec)) / len(t_dec)
                ts_summary.append((feat, f"{match * 100:.1f}% match"))
                idx += n_cat
            else:
                t_mean = np.nanmean(ts_test[:, idx])
                r_mean = np.nanmean(ts_train[:, idx])
                ts_summary.append((feat, f"{t_mean:.2f} vs {r_mean:.2f}"))
                idx += 1

    return {
        
        "token_overlap": {
            "score": len(common_tokens) / max(len(set(words_test)), 1),
            "words": list(common_tokens)[:10]
        },
        "static_comparison": {
            "gender": (g_t, g_r),
            "ethnicity": (e_t, e_r),
            "age_diff": abs(a_t - a_r)
        },
        "ts_comparison": ts_summary
    }



def find_test_neighbors(test_index: int, k=5):
    distances, indices = test_knn.kneighbors(test_embeds[test_index].reshape(1, -1))
    return [{
        "rank": rank + 1,
        "test_index": int(idx),
        "icu_id": int(test_icu_ids[idx]),
        "label": int(test_labels[idx]),
        "distance": float(dist)
    } for rank, (idx, dist) in enumerate(zip(indices[0], distances[0])) if idx != test_index][:k]

def get_patient_group_distribution(test_index: int):
    """
    Returns formatted summary stats showing mortality rate by group,
    with patient's actual group value labeled clearly.
    """
    import pandas as pd

    with h5py.File(HDF5_PATH, "r") as hf:
        test_group = hf[GROUP]["test"]
        train_group = hf[GROUP]["train"]

        # Get static features of the test patient
        static_test = test_group["s"][test_index]
        g_t, e_t, a_t = decode_static_features(static_test)
        test_groups = {
            "gender": g_t,
            "ethnicity": e_t,
            "age_bin": f"{(a_t // 10) * 10}-{(a_t // 10) * 10 + 9}"
        }

        # Extract static features and labels from training set
        train_static = train_group["s"][:]
        train_labels = train_group["label"][:]

        train_records = []
        for i in range(train_static.shape[0]):
            g, e, a = decode_static_features(train_static[i])
            train_records.append({
                "gender": g,
                "ethnicity": e,
                "age_bin": f"{(a // 10) * 10}-{(a // 10) * 10 + 9}",
                "label": int(train_labels[i])
            })

    df = pd.DataFrame(train_records)

    # Pretty names for display
    display_names = {
        "ethnicity": "Patient's Ethnicity",
        "age_bin": "Patient's Age Group",
        "gender": "Patient's Gender"
    }

    output_lines = []
    for feature in ["ethnicity", "age_bin", "gender"]:
        val = test_groups[feature]
        group_df = df[df[feature] == val]
        if len(group_df) == 0:
            line = f"No data for {feature} group '{val}'"
        else:
            rate = group_df["label"].mean() * 100
            if feature == "gender" and val.lower() in ["male", "female"]:
                label_text = "men" if val.lower() == "male" else "women"
                line = f"{rate:.1f}% {label_text} died ({display_names[feature]}: {val})"
            else:
                line = f"{rate:.1f}% in {val} group died ({display_names[feature]}: {val})"
        output_lines.append(line)

    return output_lines
