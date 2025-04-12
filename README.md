MLCompare360
A Streamlit Dashboard for Visual Comparison, Tuning & Evaluation of Multiple Classification Models

Overview
MLCompare360 is a comprehensive machine learning evaluation suite that empowers users to easily compare classification models, optimize them through hyperparameter tuning, and visualize their performance — all within a clean, intuitive dashboard.
From model training to precision-recall analysis, ROC visualization to final verdicts — every step is streamlined in one elegant application.
Designed for:
Data scientists


ML practitioners


Hackathon judges


Anyone needing rapid, informed model comparisons



Why MLCompare360?
In fast-paced data workflows or competitive hackathons, you often face these bottlenecks:
Comparing models manually is tedious


Metrics analysis is scattered


Hyperparameter tuning clutters your pipeline


Visuals live in isolated Jupyter notebooks


MLCompare360 solves these with:
✔ A centralized, professional-grade dashboard
 ✔ Automatic evaluation of multiple classifiers
 ✔ Visual insights that are ready to present

Key Features
• Train & evaluate classifiers: Logistic Regression, Random Forest, SVM, XGBoost, LightGBM
 • Visualize metrics: Accuracy, Precision, Recall, F1-Score, ROC AUC
 • Dashboard Components:
Performance table (highlighted best model)


Metric-wise bar chart


Confusion Matrix (best model)


ROC Curve (best and all models)


Per-model dropdown confusion matrix


Final Verdict Gauge


Hyperparameter tuning highlights



Tech Stack
Data Handling: pandas, numpy


Model Training: scikit-learn, xgboost, lightgbm


Hyperparameter Tuning: GridSearchCV, RandomizedSearchCV


Visualization: matplotlib, seaborn, plotly


Dashboard UI: streamlit



Dashboard Highlights
Component
Description
Performance Table
Side-by-side comparison with best model highlighted
Metric Bar Chart
Visual comparison of selected metric
Confusion Matrix
Heatmap of predictions vs actual (best model)
ROC Curve
AUC visualization for best & all models
Per-Model Dropdown
View confusion matrix for selected model
Final Verdict Gauge
Gauge chart showing F1-score of best model
Hyperparameter Tuning
Highlights models with improvement after tuning


Project Structure
bash
CopyEdit
MLCompare360/
├── 01_train_models.py         # Train multiple models and save results
├── 03_dashboard_app.py        # Streamlit dashboard to visualize and compare models
├── sample_dataset.csv         # Dataset used for training and evaluation
├── /results/                  # Stores all evaluation outputs
│   ├── results_df.csv         # Metrics for all models
│   ├── y_true.npy             # True labels from test set
│   ├── y_proba.npy            # Probabilities from best model
│   ├── y_probas_all.npy       # Probabilities from all models
│   ├── model_names.npy        # Model identifiers
│   ├── best_model.pkl         # Serialized best-performing model
│   └── hyperparams_results.csv # Tuning result comparisons


How to Run
Clone the Repository:


bash
CopyEdit
git clone https://github.com/your-username/MLCompare360.git
cd MLCompare360

Install Dependencies:


nginx
CopyEdit
pip install -r requirements.txt

Train Models:


nginx
CopyEdit
python 01_train_models.py

Launch the Dashboard:


arduino
CopyEdit
streamlit run 03_dashboard_app.py


Ideal Use Cases
Academic / capstone ML projects


Hackathons & coding competitions


Rapid internal experimentation


Client / stakeholder model presentations


Business model benchmarking & selection




Team
Made with love by Team IITM-IITMD-BA-2502-Team-2-67f7f1f1
Contributors:
Dhanyashree Karnam


Devanshi Katiyar


Keshav Ahuja
