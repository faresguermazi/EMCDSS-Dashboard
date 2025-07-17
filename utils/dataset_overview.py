import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from wordcloud import WordCloud
from io import BytesIO
import base64
import pandas as pd
import json


# These should be imported from `load_patient_data.py`
from utils.load_patient_data import GROUP, HDF5_PATH, one_hotencoder

def get_dataset_statistics(group):
    with h5py.File(HDF5_PATH, "r") as hf:
        data = hf[GROUP][group]

        # Number of samples
        num_samples = len(data["label"])

        # Label distribution
        labels = np.array(data["label"])
        label_counts = Counter(labels)
        fig, ax = plt.subplots()
        ax.bar(["Survived", "Died"], [label_counts.get(0, 0), label_counts.get(1, 0)], color=["green", "red"])
        ax.set_title(f"Label Distribution ({group.capitalize()})")
        ax.set_ylabel("Count")
        label_plot = fig
        return {
            "num_samples": num_samples,
            "label_plot": label_plot,
        }

def plot_age_distribution_categorized(group):
    with h5py.File(HDF5_PATH, "r") as hf:
        data = hf[GROUP][group]
        static_data = data["s"][()]

    age_values = [row[-1] for row in static_data]

    bins = {
        "Infant (0–1)": 0,
        "Child (1–12)": 0,
        "Adolescent (13–17)": 0,
        "Adult (18–64)": 0,
        "Elderly (65–110)": 0,
        "Unknown (>110)": 0
    }

    for age in age_values:
        if age <= 1:
            bins["Infant (0–1)"] += 1
        elif age <= 12:
            bins["Child (1–12)"] += 1
        elif age <= 17:
            bins["Adolescent (13–17)"] += 1
        elif age <= 64:
            bins["Adult (18–64)"] += 1
        elif age <= 110:
            bins["Elderly (65–110)"] += 1
        else:
            bins["Unknown (>110)"] += 1

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(bins.keys(), bins.values(), color='skyblue', edgecolor='black')
    ax.set_title(f"Age Distribution (Categorized) - {group.capitalize()}")
    ax.set_ylabel("Number of Patients")
    ax.set_xticklabels(bins.keys(), rotation=30, ha='right')
    plt.tight_layout()


    return fig

def plot_gender_pie(group):
    with h5py.File(HDF5_PATH, 'r') as hf:
        data = hf[GROUP][group]
        genders = data['s'][:, 0]  # Assuming index 0 is gender
    gender_counts = dict(Counter(genders))

    labels = ["Male", "Female"] if 0 in gender_counts and 1 in gender_counts else list(gender_counts.keys())
    sizes = [gender_counts.get(0, 0), gender_counts.get(1, 0)]

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    ax.set_title(f"Gender Distribution - {group.capitalize()}")

    return fig

def plot_ethnicity_distribution(group):
    with h5py.File(HDF5_PATH, "r") as hf:
        data = hf[GROUP][group]
        static_data = data["s"][()]

    # Extract one-hot categories from your trained encoder
    gender_categories = one_hotencoder.categories_[0]
    ethnicity_categories = one_hotencoder.categories_[1]
    num_gender = len(gender_categories)
    num_ethnicity = len(ethnicity_categories)

    ethnicity_counts = Counter()

    for row in static_data:
        ethnicity_one_hot = row[num_gender:num_gender + num_ethnicity]
        idx = np.argmax(ethnicity_one_hot)
        ethnicity = ethnicity_categories[idx]
        ethnicity_counts[ethnicity] += 1

    # Sort by count
    sorted_ethnicities = sorted(ethnicity_counts.items(), key=lambda x: x[1], reverse=True)
    labels, values = zip(*sorted_ethnicities)

    fig, ax = plt.subplots(figsize=(10, max(6, len(labels) * 0.3)))  # Dynamically scale height
    ax.barh(labels, values, color='lightgreen', edgecolor='black')
    ax.set_title(f"Ethnicity Distribution - {group.capitalize()}")
    ax.set_xlabel("Number of Patients")
    ax.invert_yaxis()  # Highest value on top
    plt.tight_layout()

    return fig

def generate_timeseries_overview_table(config):
    rows = []
    for feature in config["id_to_channel"]:
        is_cat = config["is_categorical_channel"].get(feature, False)
        possible_vals = config["possible_values"].get(feature, [])
        normal_val = config["normal_values"].get(feature, "N/A")

        type_str = "Categorical" if is_cat else "Continuous"
        if is_cat:
            type_str += f" [{len(possible_vals)} categories]"

        row = {
            "Feature Name": feature,
            "Type": type_str,
            "Normal Value": normal_val
        }

        rows.append(row)

    df = pd.DataFrame(rows)
    return df
