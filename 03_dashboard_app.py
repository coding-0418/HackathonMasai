import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, roc_curve, auc
import joblib
import os

# Setup
st.set_page_config(page_title="ML Dashboard", layout="wide")
st.markdown("<h1 style='text-align:center; color:#2E8B57;'>📊 ML Model Evaluation Dashboard</h1>", unsafe_allow_html=True)

# File paths
results_path = "results/results_df.csv"
y_true_path = "results/y_true.npy"
y_proba_path = "results/y_proba.npy"
model_path = "results/best_model.pkl"
all_probas_path = "results/y_probas_all.npy"
model_names_path = "results/model_names.npy"
hyperparams_path = "results/hyperparams_results.csv"  # Path for hyperparameter tuning results

# Load data
df = pd.read_csv(results_path)
best_model_row = df.loc[df["F1-Score"].idxmax()]
best_model_name = best_model_row["Model"]

# Highlight best row
def highlight_best(row):
    return ['background-color: #2E8B57; color: white' if row["Model"] == best_model_name else '' for _ in row]

# 📋 Model performance table
st.subheader("📋 Model Performance Table")
st.dataframe(df.style.apply(highlight_best, axis=1), use_container_width=True)

# 📊 Metric bar chart
st.subheader("📊 Compare Models by Metric")
metric = st.selectbox("Choose metric", df.columns[1:], index=3)
fig = px.bar(df, x="Model", y=metric, color="Model", text_auto=True,
             color_discrete_sequence=px.colors.qualitative.Bold)
fig.update_layout(title=f"{metric} Across Models", title_font_size=20)
st.plotly_chart(fig, use_container_width=True)

# 📉 Side-by-side Confusion Matrix & ROC Curve
if os.path.exists(y_true_path) and os.path.exists(y_proba_path):
    y_true = np.load(y_true_path)
    y_proba = np.load(y_proba_path)
    y_pred = (y_proba >= 0.5).astype(int)

    st.subheader("📉 Visual Insights: Best Model")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Confusion Matrix")
        cm = confusion_matrix(y_true, y_pred)
        fig_cm, ax = plt.subplots(figsize=(4.5, 3.5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig_cm)

    with col2:
        st.markdown("#### ROC Curve")
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)
        fig_roc, ax = plt.subplots(figsize=(4.5, 3.5))
        ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}", color="darkorange")
        ax.plot([0, 1], [0, 1], "k--")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend()
        st.pyplot(fig_roc)

# 📈 ROC curves for all models
if os.path.exists(all_probas_path) and os.path.exists(model_names_path):
    st.subheader("📈 ROC Curves - All Models")
    y_probas_all = np.load(all_probas_path, allow_pickle=True)
    model_names = np.load(model_names_path, allow_pickle=True)

    fig_all, ax = plt.subplots(figsize=(7, 5))
    for i, probs in enumerate(y_probas_all):
        fpr, tpr, _ = roc_curve(y_true, probs)
        auc_score = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{model_names[i]} (AUC = {auc_score:.2f})")
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_title("All ROC Curves")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    st.pyplot(fig_all)

# 🔧 Hyperparameter Tuning Results (Highlighted)
if os.path.exists(hyperparams_path):
    st.subheader("🔧 Hyperparameter Tuning Results")

    # Load and display the hyperparameter tuning results
    hyperparams_df = pd.read_csv(hyperparams_path)

    # Highlight models that improved after tuning
    hyperparams_df['Improvement'] = hyperparams_df['Improvement'].apply(lambda x: 'background-color: #ffeb3b' if x == 'Yes' else '')  # Highlighting the models with improvement

    # Display the dataframe with highlighted improvements
    st.markdown("<div style='background-color:#f0f8f0; padding: 1rem; border-radius: 8px;'>", unsafe_allow_html=True)
    st.markdown("<h3 style='color: #2E8B57;'>🔧 Hyperparameter Tuning Results</h3>", unsafe_allow_html=True)

    # Display the results with the improvement highlighted
    st.write("Below are the hyperparameter tuning results, with models that showed improvement highlighted:")
    st.dataframe(hyperparams_df.style.applymap(lambda v: 'background-color: #ffeb3b' if v == 'Yes' else '', subset=['Improvement']), use_container_width=True)

    # Models that showed improvement
    improved_models = hyperparams_df[hyperparams_df['Improvement'] == 'Yes']
    if not improved_models.empty:
        st.markdown("<h4 style='color: #2E8B57;'>✅ Models That Showed Improvement After Tuning</h4>", unsafe_allow_html=True)
        st.dataframe(improved_models, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# 📊 Gauge chart for Best Model F1 Score
st.subheader("🎯 Best Model F1 Gauge")
col4, col5, col6 = st.columns([1, 2, 1])
with col5:
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=best_model_row["F1-Score"],
        title={'text': f"{best_model_name}", 'font': {'size': 20}},
        gauge={
            'axis': {'range': [0, 1]},
            'bar': {'color': "#2E8B57"},
            'steps': [
                {'range': [0, 0.6], 'color': "#ffcccc"},
                {'range': [0.6, 0.8], 'color': "#ffe699"},
                {'range': [0.8, 1], 'color': "#c6efce"},
            ],
        }
    ))
    fig_gauge.update_layout(margin=dict(t=20, b=20))
    st.plotly_chart(fig_gauge, use_container_width=True)

# 🏆 Final verdict
st.markdown("<hr>", unsafe_allow_html=True)
st.subheader("🏆 Final Verdict")

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown(f"""
    <div style="border-left: 6px solid #2E8B57; padding: 1rem; background-color: #f9fdf9; 
                border-radius: 8px; text-align: center;">
        <h2 style="color: #2E8B57;">✅ {best_model_name}</h2>
        <p><strong>F1 Score:</strong> <span style="color:#444;">{best_model_row['F1-Score']:.4f}</span></p>
        <p><strong>Accuracy:</strong> <span style="color:#444;">{best_model_row['Accuracy']:.4f}</span></p>
    </div>
    """, unsafe_allow_html=True)
