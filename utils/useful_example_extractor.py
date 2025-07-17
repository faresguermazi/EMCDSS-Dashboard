# utils/useful_example_extractor.py

import pandas as pd

def load_misclassification_data(file_path="/netscratch/fguermazi/XAI/evaluation_predictions.csv"):
    """
    Load the prediction data and categorize patients as TP, FP, FN, TN.
    Returns a dictionary with counts and test data indexes for each category.
    """
    try:
        # Load the predictions file
        df_preds = pd.read_csv(file_path)

        # Identify each type of classification
        tp = df_preds[(df_preds['y_true'] == 1) & (df_preds['y_pred'] == 1)]
        fp = df_preds[(df_preds['y_true'] == 0) & (df_preds['y_pred'] == 1)]
        fn = df_preds[(df_preds['y_true'] == 1) & (df_preds['y_pred'] == 0)]
        tn = df_preds[(df_preds['y_true'] == 0) & (df_preds['y_pred'] == 0)]

        # Return the test data index instead of ICU ID
        return {
            "TP": {"count": len(tp), "indexes": tp.index.tolist()},
            "FP": {"count": len(fp), "indexes": fp.index.tolist()},
            "FN": {"count": len(fn), "indexes": fn.index.tolist()},
            "TN": {"count": len(tn), "indexes": tn.index.tolist()}
        }

    except Exception as e:
        return {"error": str(e)}

def load_confidence_based_data(file_path="/netscratch/fguermazi/XAI/evaluation_predictions.csv"):
    """
    Load the prediction data and categorize patients into high and low confidence predictions.
    Returns a dictionary with counts and test data indexes for high and low confidence.
    """
    try:
        # Load the predictions file
        df_preds = pd.read_csv(file_path)

        # Check if "prob" column exists, otherwise raise an error
        if "prob" not in df_preds.columns:
            return {"error": "Probability column not found in the predictions file."}

        # Compute confidence as distance from 0.5
        df_preds["confidence"] = abs(df_preds["prob"] - 0.5)

        # Thresholds for analysis
        high_conf_threshold = 0.9  # Very close to 1 or 0
        low_conf_threshold = 0.55  # Close to 0.5

        # Filter high and low confidence predictions
        high_conf = df_preds[df_preds["confidence"] >= (high_conf_threshold - 0.5)]
        low_conf = df_preds[df_preds["confidence"] <= (low_conf_threshold - 0.5)]

        # Prepare the result as a dictionary
        result = {
            "High Confidence": {
                "count": len(high_conf),
                "indexes": high_conf.index.tolist()
            },
            "Low Confidence": {
                "count": len(low_conf),
                "indexes": low_conf.index.tolist()
            }
        }

        return result

    except Exception as e:
        return {"error": str(e)}
