import streamlit as st
from PIL import Image
import os
from utils.load_patient_data import get_patient_info_by_index,plot_time_series_evolution,plot_compare_time_series_evolution,config, plot_time_series_base64
import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt


from utils.local_explainer import (
    plot_feature_attribution_over_time,
    plot_feature_over_time_all_methods,
    plot_time_window_attributions,
    plot_patient_time_series_attributions,
    get_detailed_static_attributions,
    visualize_text_attributions_with_auto_threshold,
    get_mean_static_attributions, 
    get_combined_time_series_attributions,
    get_top_contributing_words_table
)


from utils.global_explainer import (generate_mean_attribution_per_modality_plot,
                                    generate_top10_features_all_methods,
                                    generate_temporal_evolution_plot,
                                    plot_temporal_attribution_heatmap,
                                    generate_wordcloud_for_method)


from utils.dataset_overview import (
    get_dataset_statistics,
    plot_age_distribution_categorized,
    plot_gender_pie,
    generate_timeseries_overview_table,
    plot_ethnicity_distribution
)
from utils.attribution_consistency import (evaluate_internal_consistency,
                                            plot_correlation_heatmaps,
                                            get_confidence_vs_attr_plots,
                                            plot_attr_strength_by_outcome,
                                            evaluate_spearman_rank_correlation
                                            
)
from utils.example_based_explainer import find_test_neighbors
from utils.clix_m import load_clixm_items, save_clixm_feedback
import datetime


prediction_df = pd.read_csv("/netscratch/fguermazi/XAI/utils_generated_files/evaluation_predictions.csv")

# Setup
st.set_page_config(page_title="Exp-MCDSS", layout="wide")

st.markdown(
    "<h1 style='text-align: center;'>🧠 EMCDSS - Explainable Multimodal Clinical " \
    "Decision Support System</h1>",
    unsafe_allow_html=True
)
# Layout: Left = Patient Info | Right = Tabs
left_col, right_col = st.columns([1, 4])  

# ===================== LEFT COLUMN =====================
with left_col:
    st.markdown("### 🧾 Patient Overview")
    #st.text_input("Index from Test_Dataset", value="N/A")
    index_input = st.text_input("Index from Test_Dataset", value="")
    # Validate input
    if index_input == "":
        st.warning("Please enter an index.")
        valid_index = False
    elif not index_input.isdigit() :
        st.error("Index must be an integer.")
        valid_index = False
    else:
        valid_index = True
        patient_index = int(index_input)
        
       
        if valid_index and patient_index < len(prediction_df):
            row = prediction_df.iloc[patient_index]
            icuid = int(row['ICUSTAY_ID'])
            true_label = int(row['y_true'])
            pred_label = int(row['y_pred'])
            prob = float(row['prob'])

            true_str = "Died" if true_label == 1 else "Survived"
            pred_str = "Died" if pred_label == 1 else "Survived"

            correct = true_label == pred_label
            result_icon = "✅" if correct else "❌"

            st.markdown(f"""
            **ICU_ID**: {icuid}  
            **Ground truth**: {true_str}  
            **Prediction**: {pred_str} {result_icon}  
            **Probability of Death**: {prob:.3f}
            """)
            
            
            st.markdown("**Top-3 Similar Test Patients:**")
            neighbors = find_test_neighbors(patient_index)
            for n in neighbors:
                neighbor_row = prediction_df.iloc[n['test_index']]
                icu = int(neighbor_row['ICUSTAY_ID'])
                gt_label = int(neighbor_row['y_true'])
                pred_label = int(neighbor_row['y_pred'])
                prob = float(neighbor_row['prob'])

                gt_str = "Died" if gt_label == 1 else "Survived"
                pred_str = "Died" if pred_label == 1 else "Survived"
                result_icon = "✅" if gt_label == pred_label else "❌"

                with st.expander(f"Index: {n['test_index']} | Dist: {n['distance']:.3f}"):
                    st.markdown(f"""
                    - **ICU_ID**: {icu}  
                    - **Ground truth**: {gt_str}  
                    - **Prediction**: {pred_str} {result_icon}  
                    - **Probability of Death**: {prob:.3f}
                    """)


        else:
            st.warning("Invalid index or prediction not available.")

# ===================== RIGHT COLUMN =====================
with right_col:

    main_tabs = st.tabs(["Dataset Insights","Potential useful samples",
                         "Explainability","Patient complete information",
                         "Evaluation"])

    # ---- EXPLAINABILITY TAB ----
    with main_tabs[0]:  # Dataset Insights
        # Load stats for both sets
        train_stats = get_dataset_statistics("train")
        test_stats = get_dataset_statistics("test")
        train_age_fig = plot_age_distribution_categorized("train")
        test_age_fig = plot_age_distribution_categorized("test")
        train_gender_fig = plot_gender_pie("train")
        test_gender_fig = plot_gender_pie("test")
        train_ethnicity_fig = plot_ethnicity_distribution("train")
        test_ethnicity_fig = plot_ethnicity_distribution("test")

        st.subheader("General Overview (Train vs Test)")

        # Top: Total Samples
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Train Samples", f"{train_stats['num_samples']}")
        with col2:
            st.metric("Test Samples", f"{test_stats['num_samples']}")

        # Label distribution comparison
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Train Label Distribution**")
            train_stats["label_plot"].set_size_inches(4, 3)
            st.pyplot(train_stats["label_plot"])

        with col2:
            st.markdown("**Test Label Distribution**")
            test_stats["label_plot"].set_size_inches(4, 3)
            st.pyplot(test_stats["label_plot"])

        # Age comparison
        #st.subheader("Age Distribution (Binned Categories)")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Train Age Distribution**")
            train_age_fig.set_size_inches(4, 3)
            st.pyplot(train_age_fig)
        with col2:
            st.markdown("**Test Age Distribution**")
            test_age_fig.set_size_inches(4, 3)
            st.pyplot(test_age_fig)

        # Gender comparison
        #st.subheader("Gender Distribution")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Train Gender Distribution**")
            train_gender_fig.set_size_inches(4, 3)
            st.pyplot(train_gender_fig)
        with col2:
            st.markdown("**Test Gender Distribution**")
            test_gender_fig.set_size_inches(4, 3)
            st.pyplot(test_gender_fig)
        # Ethnicity comparison
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Train Ethnicity Distribution**")
            st.pyplot(train_ethnicity_fig)
        with col2:
            st.markdown("**Test Ethnicity Distribution**")
            st.pyplot(test_ethnicity_fig)
            # --- Section: Time-Series Feature Overview ---

        df_overview = generate_timeseries_overview_table(config)
        st.markdown("### Time-Series Feature Overview")
        st.dataframe(df_overview.style.set_properties(**{"text-align": "left"}), use_container_width=True)
   
    with main_tabs[1]: # "Potential useful samples"
        from utils.useful_example_extractor import load_misclassification_data, load_confidence_based_data

        #st.info("This section will show example patients that could be useful to explore for explainability insights.")

        example_type = st.selectbox(
            "Select a strategy to discover informative examples:",
            [
                "Misclassification-Based Example Selection",
                "High vs. Low Confidence Predictions"
            ]
        )

        if example_type == "Misclassification-Based Example Selection":
            st.markdown("#### Misclassification-Based Selection")
            try:
                misclassification_data = load_misclassification_data()

                if "error" in misclassification_data:
                    st.error(f"Error loading data: {misclassification_data['error']}")
                else:
                    # Category explanations
                    explanations = {
                        "TP": "**True Positives (TP)**: Patients correctly predicted as **died**.",
                        "FP": "**False Positives (FP)**: Patients incorrectly predicted as **died**, but actually survived.",
                        "FN": "**False Negatives (FN)**: Patients incorrectly predicted as **survived**, but actually died.",
                        "TN": "**True Negatives (TN)**: Patients correctly predicted as **survived**."
                    }

                    # Extract counts
                    TP = misclassification_data["TP"]["count"]
                    FP = misclassification_data["FP"]["count"]
                    FN = misclassification_data["FN"]["count"]
                    TN = misclassification_data["TN"]["count"]

                    # Confusion matrix values
                    cm = np.array([[TN, FP],
                                [FN, TP]])
                    labels = [["TN", "FP"], ["FN", "TP"]]
                    x_labels = ["Pred: Survived", "Pred: Died"]
                    y_labels = ["Actual: Survived", "Actual: Died"]

                    # Create compact confusion matrix plot
                    fig, ax = plt.subplots(figsize=(3.5, 3.2))  # Smaller figure
                    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
                    ax.set(xticks=np.arange(2),
                        yticks=np.arange(2),
                        xticklabels=x_labels,
                        yticklabels=y_labels,
                        xlabel='Predicted Label',
                        ylabel='Actual Label',
                        title='Confusion Matrix')

                    # Text annotations in each cell
                    for i in range(2):
                        for j in range(2):
                            ax.text(j, i, f"{cm[i, j]}\n({labels[i][j]})",
                                    ha="center", va="center",
                                    color="white" if cm[i, j] > np.max(cm)/2 else "black",
                                    fontsize=10)

                    ax.tick_params(labelsize=9)
                    ax.grid(False)

                    # Place figure in middle column
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        st.pyplot(fig)

                    # Expanders for each category
                    for category in ["TP", "FP", "FN", "TN"]:
                        count = misclassification_data[category]["count"]
                        indexes = misclassification_data[category]["indexes"]

                        with st.expander(f"{explanations[category]} (Count = {count})"):
                            if count > 0:
                                index_list = ", ".join(map(str, indexes))
                                st.markdown(f"**Test Data Indexes:** {index_list}")
                            else:
                                st.markdown("No examples in this category.")

            except Exception as e:
                st.error(f"Error displaying misclassification data: {e}")


        elif example_type == "High vs. Low Confidence Predictions":
                st.markdown("#### Confidence-Based Selection")

                # Explanation of high and low confidence predictions
                st.markdown("""
                - **High Confidence:** Predictions where the model's output probability is very close to 1 or 0, 
                indicating the model is highly certain about the prediction.
                - **Low Confidence:** Predictions where the model's output probability is closer to 0.5, 
                indicating the model is less certain and the prediction could go either way.
                """)

                try:
                    # Load confidence-based data using the utility function
                    confidence_data = load_confidence_based_data()

                    if "error" in confidence_data:
                        st.error(f"Error loading data: {confidence_data['error']}")
                    else:
                        # Display summary counts with collapsible menus
                        for category, info in confidence_data.items():
                            count = info['count']
                            indexes = info['indexes']

                            with st.expander(f"**{category} Predictions (Count = {count})**"):
                                if count > 0:
                                    index_list = ", ".join(map(str, indexes))
                                    st.markdown(f"**Test Data Indexes:** {index_list}")
                                else:
                                    st.markdown("No examples in this category.")

                except Exception as e:
                    st.error(f"Error displaying confidence-based data: {e}")

    with main_tabs[2]: # "Explainability"
        explainability_tabs = st.tabs(["Global", "Local", "Example-Based"])

        with explainability_tabs[0]:  #Global
            st.markdown("### Global Explainability")

            col1, col2 = st.columns([2, 1])  # Wider for main dropdown
            with col1:
                selected_use_case = st.selectbox(
                    "Global Explanation Type",
                    [
                        "Mean Attribution per Modality",
                        "Top 10 Time-Series Features Attributions",
                        "Temporal Evolution of Time-Series Attribution",
                        "Temporal Attribution Heatmap per Feature",
                        "Most Influential Words in Clinical Notes"
                    ]
                )

            with col2:
                methods = ["IG", "LIG", "LayerDeepLift", "SHAP"]
                if selected_use_case in [
                    "Temporal Attribution Heatmap per Feature",
                    "Most Influential Words in Clinical Notes"
                ]:
                    selected_method = st.selectbox("Select Post-hoc Method", methods, index=0)


            st.markdown("---")

            # === Plot and Display ===
            if selected_use_case == "Mean Attribution per Modality":
                fig, summary_table = generate_mean_attribution_per_modality_plot()
                if fig:
                    st.pyplot(fig)
                    
                else:
                    st.error("Could not generate global explainability plot.")
            
            elif selected_use_case == "Top 10 Time-Series Features Attributions":
                fig, summary_table = generate_top10_features_all_methods()
                if fig:
                    st.pyplot(fig)
                    
                else:
                    st.error("Could not generate top 10 features plot.")

            elif selected_use_case == "Temporal Evolution of Time-Series Attribution":
                fig, df_temporal = generate_temporal_evolution_plot(normalize=True)
                if fig:
                    st.pyplot(fig)
                    st.markdown("""
                         **Note:** The plot shows the *normalized* temporal evolution 
                        of time-series attributions, focusing on the *shape of changes over time* 
                        rather than absolute magnitude. This helps compare trends across methods.
                    """)
                else:
                    st.error("Could not generate temporal evolution plot.")
            
            elif selected_use_case == "Temporal Attribution Heatmap per Feature":
                if selected_method in methods:
                    fig = plot_temporal_attribution_heatmap(selected_method, normalize=True)
                    if fig:
                        st.pyplot(fig)
                    else:
                        st.error("Error generating the heatmap.")
                else:
                    st.warning("Please select a valid post-hoc method.")

            
            elif selected_use_case == "Most Influential Words in Clinical Notes":
                fig = generate_wordcloud_for_method(selected_method)
                if fig:
                    st.pyplot(fig)
                else:
                    st.error("Error generating the word cloud.")
                    
        with explainability_tabs[1]:  #Local Explainability
            if valid_index:
                info = get_patient_info_by_index(patient_index)

                icu_id = info["icu_id"]
                true_label = info["true_label"]
                text = info["text"]
                gender = info["gender"]
                ethnicity = info["ethnicity"]
                age = info["age"]
                time_series = info["time_series"]
                time_series_shape = info["time_series_shape"]
                
                # Extract mean attributions for each method
                methods = ["IG", "LIG", "LayerDeepLift", "SHAP"]
                attributions = {}
                active_attributions = {}
                for method in methods:
                    (gender_attr, ethnicity_attr, age_attr), (gender_active_attr, ethnicity_active_attr, age_active_attr) = get_mean_static_attributions(patient_index, method)
                    attributions[method] = [f"{gender_attr:.4e}", f"{ethnicity_attr:.4e}", f"{age_attr:.4e}"]
                    active_attributions[method] = [f"{gender_active_attr:.4e}", f"{ethnicity_active_attr:.4e}", f"{age_active_attr:.4e}"]

                try: 
                    local_tabs = st.tabs(["Overview", "Detailed","Combined Methods"])
                    with local_tabs[0]: #Overiview
                        st.markdown("### Overview Local Explainability")
                        col5, col6 = st.columns(2)  # Both columns have equal width

                        with col5:
                            # Step 1: Select Data Modality
                            data_modality2 = st.selectbox(
                                "Select Data Modality",
                                ["Static data", "Time Series data", "Textual data"],
                                key="overview_modality"
                            )
                        
                        with col6:
                            # Post-hoc methods (common for all data modalities)
                            post_hoc_method2 = st.selectbox(
                                "Select Post-hoc Method",
                                ["Compare all methods","IG", "LIG", "LayerDeepLift", "SHAP"],
                                key="overview_method"
                            )
                        
                        if data_modality2 == "Static data" :
                            if post_hoc_method2 == "Compare all methods":
                                #st.markdown("### Static Data Attribution - Compare All Methods")
                                # Integrate into the static data dictionary
                                static_data = {
                                    "Feature": ["Gender", "Ethnicity", "Age"],
                                    "Value": [gender, ethnicity, age],
                                    "IG": active_attributions["IG"],
                                    "LIG": active_attributions["LIG"],
                                    "LayerDeepLift": active_attributions["LayerDeepLift"],
                                    "SHAP": active_attributions["SHAP"]
                                }

                                static_df = pd.DataFrame(static_data)
                                header = pd.MultiIndex.from_tuples([
                                    ("Feature", ""),              
                                    ("Value", ""),                 
                                    ("Mean Attribution", "IG"),   
                                    ("Mean Attribution", "LIG"),
                                    ("Mean Attribution", "LayerDeepLift"),
                                    ("Mean Attribution", "SHAP")
                                ], names=["", "Method"])
                                static_df.columns = header
                                st.dataframe(static_df, width=800)
                            else:
                                strg = post_hoc_method2 + " feature mean attribution"
                                strg2 = post_hoc_method2 + " active feature attribution"

                                static_df4 = pd.DataFrame({
                                    "Feature": ["Gender:", "Ethnicity:", "Age:"],
                                    strg : attributions[post_hoc_method2],
                                    "Value": [gender, ethnicity, age],
                                    strg2: active_attributions[post_hoc_method2]
                                })
                                st.dataframe(static_df4, width=800)
                    
                        elif data_modality2 == "Time Series data":
                            if post_hoc_method2 == "Compare all methods":
                                # Return all methods at once (None means no filter)
                                ts_df = get_combined_time_series_attributions(patient_index)
                                
                                # Set MultiIndex header for columns
                                header = pd.MultiIndex.from_tuples([
                                    ("Feature", ""),
                                    ("Mean Value", ""),
                                    ("Mean Attribution", "IG"),
                                    ("Mean Attribution", "LIG"),
                                    ("Mean Attribution", "LayerDeepLift"),
                                    ("Mean Attribution", "SHAP")
                                ])
                                ts_df.columns = header
                            else:
                                # Only extract one method’s attribution column
                                ts_df = get_combined_time_series_attributions(patient_index, method_name=post_hoc_method2)
                                ts_df.columns = ["Feature", "Mean Value", f"{post_hoc_method2} Attribution"]

                            # Final formatting
                            ts_df.reset_index(drop=True, inplace=True)
                            st.dataframe(ts_df, width=1000)

                        elif data_modality2 == "Textual data":
                            if post_hoc_method2 == "Compare all methods":
                                methods = ["IG", "LIG", "LayerDeepLift", "SHAP"]
                                table_data = {"Method": [], "Top Attributed Words": []}

                                for method in methods:
                                    df = get_top_contributing_words_table(patient_index, method)
                                    words = df["Word"].tolist() if "Word" in df else ["N/A"]
                                    table_data["Method"].append(method)
                                    table_data["Top Attributed Words"].append(", ".join(words))

                                text_df = pd.DataFrame(table_data)
                                st.dataframe(text_df, width=900)

                            else:
                                word_df = get_top_contributing_words_table(patient_index, post_hoc_method2)
                                st.dataframe(word_df, width=700)
                                                                              
                    with local_tabs[1]: #Detailed                      
                        st.markdown("### Detailed Local Explainability")
                        col1, col2 = st.columns(2)  # Both columns have equal width

                        with col1:
                            data_modality = st.selectbox(
                                "Select Data Modality",
                                ["Static data", "Time Series data", "Textual data"],
                                key="detailed_modality"
                            )

                        with col2:
                            post_hoc_method = st.selectbox(
                                "Select Post-hoc Method",
                                ["Compare all methods","IG", "LIG", "LayerDeepLift", "SHAP"],
                                key="detailed_method"
                            )
                            
                        col3, col4 = st.columns(2) 
                        with col3:
                        
                            if data_modality == "Time Series data":
                                time_series_option = st.selectbox(
                                    "Select Time Series Use Case",
                                    [
                                        "Top 6 Features Attributions Over Time",
                                        "Attribution of Specific Feature Over Time",
                                        "Time Window Attributions (Average Absolute Value)"
                                    ]
                                )
                            
                            elif data_modality == "Textual data":
                                textual_option = st.selectbox(
                                    "Select Textual Use Case",
                                    ["Heatmap of Word Attributions"]
                                )

                        with col4:
                            if data_modality == "Time Series data" and time_series_option == "Attribution of Specific Feature Over Time":
                                feature_names = [
                                    'Capillary refill rate', 'Diastolic blood pressure', 
                                    'Fraction inspired oxygen', 'Glascow coma scale eye opening', 
                                    'Glascow coma scale motor response', 'Glascow coma scale total', 
                                    'Glascow coma scale verbal response', 'Glucose', 'Heart Rate', 
                                    'Height', 'Mean blood pressure', 'Oxygen saturation', 
                                    'Respiratory rate', 'Systolic blood pressure', 
                                    'Temperature', 'Weight', 'pH'
                                ]
                                selected_feature = st.selectbox(
                                    "Select Feature",
                                    feature_names
                                )
                            elif data_modality == "Time Series data" and time_series_option == "Time Window Attributions (Average Absolute Value)":
                                num_windows = st.number_input(
                                    "number of time windows",
                                    min_value=1, max_value=48, value=12,step=1
                                )

                            elif data_modality == "Textual data":
                                percentile = st.number_input(
                                    "Percentile",
                                    min_value=0, max_value=100, value=30
                                )

                        if data_modality == "Static data":
                            if post_hoc_method == "Compare all methods":
                                detailed_df = get_detailed_static_attributions(patient_index, method_name="Compare all methods")

                                
                                expected_cols = ["Feature", "Value", "IG", "LIG", "LayerDeepLift", "SHAP"]
                                if list(detailed_df.columns) == expected_cols:
                                    header = pd.MultiIndex.from_tuples(
                                        [("Feature", ""),
                                        ("Value", "")] +
                                        [("Attributions", method) for method in ["IG", "LIG", "LayerDeepLift", "SHAP"]]
                                    )
                                    detailed_df.columns = header
                                st.dataframe(detailed_df, width=900)

                            else:
                                detailed_df = get_detailed_static_attributions(patient_index, post_hoc_method)
                                st.dataframe(detailed_df, width=700)                            
                                                    

                        elif data_modality == "Time Series data":                        
                            if time_series_option == "Top 6 Features Attributions Over Time":
                                if post_hoc_method != "Compare all methods":
                                    fig = plot_patient_time_series_attributions(patient_index, post_hoc_method)
                                    if fig:
                                        st.pyplot(fig)
                                    else:
                                        st.error("Could not generate plot.")
                                else:
                                    st.markdown("#### Top-6 Time Series Feature Attributions – All Methods")
                                    st.markdown("This section shows how different explanation methods attribute importance to time-series features for the same patient.")

                                    methods = ["IG", "LIG", "LayerDeepLift", "SHAP"]
                                    cols = st.columns(2)

                                    for idx, method in enumerate(methods):
                                        fig = plot_patient_time_series_attributions(patient_index, method)
                                        if fig:
                                            with cols[idx % 2]:
                                                st.pyplot(fig, use_container_width=True)
                                                st.caption(f"{method}")
                                        else:
                                            st.warning(f"Could not generate plot for method: {method}")

                            
                            
                            elif time_series_option == "Attribution of Specific Feature Over Time":
                                if post_hoc_method == "Compare all methods":
                                    fig = plot_feature_over_time_all_methods(patient_index, selected_feature)
                                    if fig:
                                        st.pyplot(fig)
                                    else:
                                        st.error("Could not generate plot.")
                                else:
                                    fig = plot_feature_attribution_over_time(patient_index, selected_feature,post_hoc_method)
                                    if fig:
                                        st.pyplot(fig)
                                    else:
                                        st.error("Could not generate plot.")

                            elif time_series_option == "Time Window Attributions (Average Absolute Value)":
                                st.info(f"Each window will cover approximately {48 // num_windows} hours.")
                                if post_hoc_method != "Compare all methods":
                                    fig = plot_time_window_attributions(patient_index, post_hoc_method, num_windows)
                                    if fig:
                                            st.pyplot(fig)
                                    else:
                                            st.error("Could not generate plot.")
                                else:
                                    st.info("Please select a specific post-hoc method.")

                        elif data_modality == "Textual data":
                             if textual_option == "Heatmap of Word Attributions":
                                if post_hoc_method == "Compare all methods":
                                    st.markdown("Please select a specific post-hoc method to visualize attributions.")
                                else:
                                    html_output, threshold = visualize_text_attributions_with_auto_threshold(
                                        patient_index, post_hoc_method, percentile
                                    )
                                    st.markdown(html_output, unsafe_allow_html=True)
                                    st.caption(f"Computed Threshold for percentile {percentile}% = {threshold:.5f} ")

                    with local_tabs[2]:  #Combine
                        st.markdown("## Choose Methods per Modality")

                        if valid_index:
                            st.markdown("Select attribution method for each modality. For static data, only SHAP is supported.")

                            col1, col2, col3 = st.columns(3)
                            with col1:
                                static_method = st.selectbox(
                                    "Select Method for Static",
                                    ["SHAP"], key="combo_st_method"
                                )
                            with col2:
                                time_series_method = st.selectbox(
                                    "Select Method for Time-Series",
                                    ["LIG", "IG", "SHAP","LayerDeepLift"], key="combo_ts_method"
                                )
                            with col3:
                                text_method = st.selectbox(
                                    "Select Method for Text",
                                    ["LIG", "IG","SHAP","LayerDeepLift"], key="combo_text_method"
                                )

                            st.divider()
                            st.markdown(f"### Static data Attributions ({static_method})")
                            static_df4 = pd.DataFrame({
                                    "Feature": ["Gender:", "Ethnicity:", "Age:"],
                                    "Value": [gender, ethnicity, age],
                                    "Attribution": active_attributions[static_method]
                                })
                            st.dataframe(static_df4, width=800)

                            st.divider()
                            st.markdown(f"### Time-Series Attributions ({time_series_method})")

                            col3, col4 = st.columns(2)
                            with col3:
                                st.markdown("#### Top 6 Attributed Features Over Time")
                                ts_fig = plot_patient_time_series_attributions(patient_index, time_series_method)
                                if ts_fig:
                                    st.pyplot(ts_fig)
                                else:
                                    st.warning("Could not generate Top-6 plot.")

                            with col4:
                                st.markdown("#### Time Window Attributions (Avg Absolute Value)")
                                num_windows = st.number_input(
                                    "Number of Time Windows", min_value=1, max_value=48, value=12, key="combo_windows"
                                )
                                window_fig = plot_time_window_attributions(patient_index, time_series_method, num_windows)
                                if window_fig:
                                    st.pyplot(window_fig)
                                else:
                                    st.warning("Could not generate time window plot.")

                            st.divider()
                            st.markdown(f"### Textual Attributions ({text_method})")

                            percentile = st.slider("Attribution Threshold Percentile", 0, 100, 30, step=1, key="combo_text_thresh")
                            html_output, threshold = visualize_text_attributions_with_auto_threshold(patient_index, text_method, percentile)
                            st.markdown(html_output, unsafe_allow_html=True)
                            st.caption(f"Computed Threshold = {threshold:.5f}")

                        else:
                            st.info("Please enter a valid patient index in the left panel.")
                            
                except Exception as e:
                    st.error(f"Could not load patient info: {e}")
            else:
                st.info("Please enter a valid patient index in the left panel.")
             
        with explainability_tabs[2]: #Example Based
            from utils.example_based_explainer import find_train_neighbors, get_train_patient_info, compare_patients,get_patient_group_distribution
            # Load test embeddings once
            if "_test_data_embeddings" not in st.session_state:
                with open("test_data.json", "r") as f:
                    test_data = json.load(f)
                st.session_state._test_data_embeddings = test_data
            else:
                test_data = st.session_state._test_data_embeddings

            if valid_index and 0 <= patient_index < len(test_data):
                # ========= NEW: Show Data Distribution Summary =========
                st.markdown("### 📊 Data Distribution")
                try:
                    group_lines = get_patient_group_distribution(patient_index)
                    for line in group_lines:
                        st.markdown(f"- {line}")
                except Exception as e:
                    st.warning(f"Unable to display group distribution: {e}")

                st.divider()
                # ========= STEP 1: Show Neighbors =========
                test_embedding = np.array(test_data[patient_index]["embedding"])
                neighbors = find_train_neighbors(test_embedding)

                st.markdown("### 🔍 Top-5 Most Similar Training Patients")
                for n in neighbors:
                    with st.expander(f"Rank #{n['rank']} | Train Index: {n['train_index']} | ICU_ID: {n['icu_id']} | Label: {n['label']} | Distance: {n['distance']:.4f}"):
                        train_info = get_train_patient_info(n['train_index'])
                        st.markdown(f"**ICU_ID**: {train_info['icu_id']}")
                        st.markdown(f"**Label**: {'Died' if train_info['label'] == 1 else 'Survived'}")
                        st.markdown(f"**Gender**: {train_info['gender']}")
                        st.markdown(f"**Ethnicity**: {train_info['ethnicity']}")
                        st.markdown(f"**Age**: {train_info['age']}")
                        st.markdown("**Time Series data:**")

                        df_ts = pd.DataFrame(train_info["time_series"], columns=["Feature", "Values (48 Timesteps)"])
                        st.markdown(df_ts.to_html(index=False, escape=False), unsafe_allow_html=True)

                        st.markdown("**Notes:**")
                        st.text_area(label="", value=train_info['text'], height=100)
                        
                st.divider()

                # ========= STEP 2: Compare With a Specific Train Index =========
                st.markdown("### Compare Test patient With a Specific Train Sample")
                neighbor_indexes = [n['train_index'] for n in neighbors]

                # Show dropdown for selection
                train_index_input = st.selectbox(
                    "Select a training index to compare with the test patient:",
                    neighbor_indexes,
                    format_func=lambda idx: f"{idx}",
                    key="compare_input_select"
                )

                train_index = train_index_input
                try:
                    comparison = compare_patients(patient_index, train_index)
                    train_info = get_train_patient_info(train_index)
                    test_info = get_patient_info_by_index(patient_index)

                    # TEXT
                    st.markdown(f"**Token Overlap:** {comparison['token_overlap']['score']*100:.2f}%")
                    st.markdown(f"**Overlapping Tokens:** {', '.join(comparison['token_overlap']['words']) or 'None'}")

                    # STATIC
                    s = comparison['static_comparison']
                    st.markdown("**Static Feature Comparison:**")
                    st.markdown(f"- Gender: {s['gender'][0]} vs {s['gender'][1]} {'✅' if s['gender'][0] == s['gender'][1] else '❌'}")
                    st.markdown(f"- Ethnicity: {s['ethnicity'][0]} vs {s['ethnicity'][1]} {'✅' if s['ethnicity'][0] == s['ethnicity'][1] else '❌'}")
                    st.markdown(f"- Age Difference: {s['age_diff']} years")

                    # TIME SERIES
                    ts_score_dict = dict(comparison['ts_comparison'])

                    # Loop through each feature from train sample
                    for feature_name, _ in train_info["time_series"]:
                        score = ts_score_dict.get(feature_name, "N/A")
                        
                        st.markdown(f"- **{feature_name}**: {score}")

                        with st.expander("View comparaison plot"):
                            plot_html = plot_compare_time_series_evolution(test_info, train_info, feature_name)
                            st.markdown(plot_html, unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"Comparison failed: {e}")
            
            else:
                st.info("Please enter a valid test patient index in the left panel.")
                                
    with main_tabs[3]:  # "Patient complete information"
        if valid_index:
            try:
                info = get_patient_info_by_index(patient_index)

                icu_id = info["icu_id"]
                true_label = info["true_label"]
                text = info["text"]
                gender = info["gender"]
                ethnicity = info["ethnicity"]
                age = info["age"]
                time_series = info["time_series"]
                time_series_shape = info["time_series_shape"]

                # Section 1: Static Features and Clinical Notes Side-by-Side
                col_static, col_text = st.columns([1, 3])  # 1: narrow static, 3: wide text

                with col_static:
                    st.markdown("#### Static Features")
                    static_df = pd.DataFrame({
                        "Feature": ["Gender:", "Ethnicity:", "Age:"],
                        "Value": [gender, ethnicity, age]
                    })
                    st.markdown(static_df.to_html(index=False), unsafe_allow_html=True)

                with col_text:
                    st.markdown("#### Clinical Notes")
                    st.markdown(f"<div style='background-color:#f8f9fa;padding:10px;border-radius:5px; height:200px; overflow:auto'>{text}</div>", unsafe_allow_html=True)

                
                
                # Section 2: Time-Series Summary

                st.markdown("#### Time-Series Data")

                for i, (feature_name, values) in enumerate(info["time_series"]):
                    col1, col2, col3 = st.columns([1, 2, 4])

                    with col1:
                        st.markdown(f"**{feature_name}**")

                    with col2:
                        st.text_area("Values", values, height=80, disabled=True, label_visibility="collapsed")

                    with col3:
                        fig = plot_time_series_evolution(patient_index, feature_name)
                        if fig:
                            st.pyplot(fig) 
                           



            except Exception as e:
                st.error(f"Could not load patient info: {e}")
        else:
            st.info("Please enter a valid patient index in the left panel.")
    
    with main_tabs[4]:  # Attribution Evaluation Tab
        eval_tabs = st.tabs(["CLIX-M Evaluation","Attribution evaluation plots"])

        with eval_tabs[0]:  # Human-in-the-Loop (CLIX-M)

            with eval_tabs[0]:


                st.subheader("CLIX-M Evaluation")

                st.markdown("""
                This section allows clinicians and developers to evaluate how well the dashboard supports explainability based on the CLIX-M checklist.

                - 🩺 = Feedback required from **clinician**  
                - 🛠️ = Feedback required from **developer**  
                - All users may comment on any item.
                """)

                user_role = st.radio("Select your role:", options=["clinician", "developer"], horizontal=True)
                clixm_items = load_clixm_items()

                feedback = {}
                missing_required = []

                for item in clixm_items:
                    item_name = item["item_name"]
                    input_type = item.get("input_type", "text")
                    required_for_user = item.get(f"required_{user_role}", False)

                    # Prefix with emoji if required
                    icon = ""
                    if item["required_clinician"]:
                        icon += "🩺"
                    if item["required_developer"]:
                        icon += "🛠️"

                    st.markdown(f"### {icon} {item_name}")
                    comment_key = f"{item_name}_comment"
                    rating_key = f"{item_name}_rating"

                    # Text area (optional for both roles)
                    comment = st.text_area("Notes:", key=comment_key)
                    feedback[item_name] = {"comment": comment.strip()}

                    # Handle rating input
                    if input_type in ["dropdown", "scale"]:
                        if input_type == "dropdown":
                            options = ["Unknown"] + item["dropdown_options"]
                        else:
                            options = ["Unknown",0 , 1, 2, 3, 4, 5]

                        selected_option = st.radio(
                            f"Rate {item_name}",
                            options=options,
                            horizontal=True,
                            key=rating_key
                        )
                        feedback[item_name]["rating"] = selected_option

                        # Check if rating is required and valid
                        if required_for_user and selected_option == "Unknown":
                            st.warning(f"Please provide a valid rating for: {item_name}")
                            missing_required.append(item_name)

                    elif input_type == "text":
                        feedback[item_name]["rating"] = None

                # Submit button
                if st.button("💾 Submit Feedback"):
                    if missing_required:
                        st.error("Please provide all required ratings (cannot be 'Unknown') before submitting.")
                    else:
                        save_clixm_feedback(user_role, feedback)
                        st.success("Thank you! Your feedback has been saved.")



        with eval_tabs[1]:  # Evaluation Plots
            st.subheader("Cross-Method Attribution Consistency")
            st.markdown("""
            Understanding how explanation methods agree helps assess the reliability of feature attributions.
            Two types of correlation metrics to compare methods are used:

            - **Pearson correlation** measures the **linear relationship** between attribution values across methods. It captures whether the *actual values* (e.g., 0.1 vs 0.5) are aligned between two methods (value-level agreement).
            - **Spearman rank correlation** focuses on **ranking consistency**. It measures whether the *order* of feature importance (e.g., most to least important) is similar, even if the actual values differ (rank-level agreement).
            """)

            # ---------- Pearson Correlation Section ----------
            st.markdown("""
            ### Attribution Method Consistency: Pearson Correlation
            """)

            # Compute and plot Pearson correlation
            pearson_correlations = evaluate_internal_consistency()
            pearson_plots = plot_correlation_heatmaps(pearson_correlations)

            # Row 1: Pearson plots
            col1, col2, col3 = st.columns(3)

            with col1:
                st.pyplot(pearson_plots["static"])
                st.caption("**Static modality** (Pearson): Correlation of attribution scores across methods for demographics (e.g., gender, ethnicity, age).")

            with col2:
                st.pyplot(pearson_plots["text"])
                st.caption("**Text modality** (Pearson): Method agreement on word/token importance in clinical notes.")

            with col3:
                st.pyplot(pearson_plots["time_series"])
                st.caption("**Time-series modality** (Pearson): Agreement on feature-time relevance scores over a 48-hour ICU window.")

            # ---------- Spearman Correlation Section ----------
            st.markdown("""
            ### Attribution Method Consistency: Spearman Rank Correlation
            """)

            # Compute and plot Spearman correlation
            spearman_correlations = evaluate_spearman_rank_correlation()
            spearman_plots = plot_correlation_heatmaps(spearman_correlations)

            # Row 2: Spearman plots
            row2_col1, row2_col2, row2_col3 = st.columns(3)

            with row2_col1:
                st.pyplot(spearman_plots["static"])
                st.caption("**Static modality** (Spearman): Rank-order consistency of demographic feature importance across explanation methods.")

            with row2_col2:
                st.pyplot(spearman_plots["text"])
                st.caption("**Text modality** (Spearman): Do different methods prioritize the same clinical note tokens, even if their raw scores differ?")

            with row2_col3:
                st.pyplot(spearman_plots["time_series"])
                st.caption("**Time-series modality** (Spearman): Rank-based agreement on feature-time importance across the 48-hour ICU stay.")
                    
            st.markdown("---")
            st.markdown("### Explore Method Differences in Detail")
            st.markdown(
                """
                To explore how different post-hoc methods assign importance across time, the dashboard includes a dedicated view for time-series attribution comparisons. Users can visualize how the most relevant features evolve over time for a selected patient, and how attribution patterns differ between methods such as IG, SHAP, or LIG. This helps highlight method-level differences in temporal behavior and feature emphasis. ( Detailed Local explanations for TimeSeriesData)
                """
            )


            st.markdown("### Attribution Strength vs Model Confidence")
            st.write(
                "This plot investigates whether higher model confidence is reflected by stronger attributions. "
                "It helps assess whether the model's certainty aligns with the feature importance magnitude across different explanation methods."
            )

            confidence_plots = get_confidence_vs_attr_plots()
            conf_cols = st.columns(2)

            for i, method in enumerate(confidence_plots):
                with conf_cols[i % 2]:
                    st.pyplot(confidence_plots[method])

            st.markdown("---")
            st.markdown("### Attribution Strength by Outcome Type")
            st.markdown("""
            This section shows how strongly each method attributes importance across different prediction outcomes (TP, FP, FN, TN).
            Higher variance in strength across groups may indicate inconsistent or biased explanations.
            """)

            strength_plots = plot_attr_strength_by_outcome()

            cols = st.columns(2)
            method_names = list(strength_plots.keys())

            for i, method in enumerate(method_names):
                with cols[i % 2]:
                    st.pyplot(strength_plots[method])


    