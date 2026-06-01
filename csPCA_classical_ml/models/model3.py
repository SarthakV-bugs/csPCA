import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    classification_report, accuracy_score, roc_curve, auc, confusion_matrix,
    precision_recall_curve, average_precision_score, balanced_accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb

# =====================
# Setup - Create Folders
# =====================
os.makedirs("plots", exist_ok=True)
os.makedirs("models", exist_ok=True)

# =====================
# Data Loading
# =====================
def load_data():
    X = np.load("/home/ibab/PycharmProjects/ML-PROJECT/preprocessing/X.npy")
    y = np.load("/home/ibab/PycharmProjects/ML-PROJECT/preprocessing/y.npy")
    return X, y

# =====================
# Preprocessing
# =====================
def preprocess_data(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, 'models/scaler.pkl')
    return X_scaled

# =====================
# PCA
# =====================
def apply_pca(X_scaled, n_components):
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    joblib.dump(pca, 'models/pca.pkl')
    return X_pca, pca

def plot_pca_variance(pca):
    plt.figure(figsize=(8, 5))
    plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('PCA - Explained Variance')
    plt.grid(True)
    plt.savefig('plots/pca_explained_variance.png')
    plt.close()

# =====================
# Models
# =====================
def initialize_models():
    return {
        "XGBoost": xgb.XGBClassifier(eval_metric='logloss'),
        "Random Forest": RandomForestClassifier(),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "SVM": SVC(probability=True),
        "KNN": KNeighborsClassifier(),
        "Gradient Boosting": GradientBoostingClassifier(),
        "Decision Tree": DecisionTreeClassifier()
    }

# =====================
# Hyperparameter Tuning
# =====================
def hyperparameter_tuning(models, X_train, y_train):
    param_grids = {
        "XGBoost": {
            "n_estimators": [100, 200],
            "learning_rate": [0.05, 0.1],
            "max_depth": [3, 5]
        },
        "Random Forest": {
            "n_estimators": [100, 200],
            "max_depth": [3, 5, 7]
        },
        "Logistic Regression": {
            "C": [0.1, 1, 10]
        },
        "SVM": {
            "C": [0.1, 1, 10],
            "kernel": ['rbf'],
        },
        "KNN": {
            "n_neighbors": [3, 5, 7]
        },
        "Gradient Boosting": {
            "n_estimators": [100, 200],
            "learning_rate": [0.05, 0.1]
        },
        "Decision Tree": {
            "max_depth": [3, 5, 7]
        }
    }
    best_models = {}
    for name, model in models.items():
        print(f"Tuning {name}...")
        grid = GridSearchCV(model, param_grids[name], cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
        grid.fit(X_train, y_train)
        best_models[name] = grid.best_estimator_
        print(f"Best Params for {name}: {grid.best_params_}")
    return best_models

# =====================
# Stacking
# =====================
def stacking_ensemble(models):
    estimators = [(name, model) for name, model in models.items()]
    stacking_model = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
    return stacking_model

# =====================
# Evaluation
# =====================
def plot_confusion(y_test, y_pred, model_name):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(f'plots/{model_name}_confusion_matrix.png')
    plt.close()

def plot_roc(y_test, y_pred_proba, model_name):
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend()
    plt.savefig(f'plots/{model_name}_roc_curve.png')
    plt.close()

def plot_precision_recall(y_test, y_pred_proba, model_name):
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    avg_prec = average_precision_score(y_test, y_pred_proba)
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label=f'AP = {avg_prec:.2f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {model_name}')
    plt.legend()
    plt.savefig(f'plots/{model_name}_precision_recall.png')
    plt.close()

# def feature_importance(model, model_name, X_train):
#     if hasattr(model, "feature_importances_"):
#         importances = model.feature_importances_
#         indices = np.argsort(importances)[::-1]
#         plt.figure(figsize=(10, 6))
#         plt.title(f"Feature Importances - {model_name}")
#         plt.bar(range(X_train.shape[1]), importances[indices], align="center")
#         plt.savefig(f'plots/{model_name}_feature_importances.png')
#         plt.close()

# def model_interpretability(model, X_train, model_name):
#     if isinstance(model, (RandomForestClassifier, GradientBoostingClassifier, xgb.XGBClassifier)):
#         explainer = shap.TreeExplainer(model)
#         shap_values = explainer.shap_values(X_train)
#         shap.summary_plot(shap_values, X_train, show=False)
#         plt.title(f'SHAP Summary - {model_name}')
#         plt.savefig(f'plots/{model_name}_shap_summary.png')
#         plt.close()

def evaluate_models(models, X_train, y_train, X_test, y_test):
    results = {}
    metrics_table = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

        acc = accuracy_score(y_test, y_pred)
        bal_acc = balanced_accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else 0

        metrics_table.append([name, acc, bal_acc, prec, rec, f1, roc_auc])

        results[name] = acc
        print(f"\nModel: {name}")
        print(classification_report(y_test, y_pred))

        # Save Model
        joblib.dump(model, f'models/{name}.pkl')

        # Plots
        plot_confusion(y_test, y_pred, name)
        if y_pred_proba is not None:
            plot_roc(y_test, y_pred_proba, name)
            plot_precision_recall(y_test, y_pred_proba, name)
        # feature_importance(model, name, X_train)
        # model_interpretability(model, X_train, name)

    # Save metrics table
    import pandas as pd
    df_metrics = pd.DataFrame(metrics_table, columns=["Model", "Accuracy", "Balanced Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"])
    df_metrics.to_csv("plots/model_performance_summary.csv", index=False)
    print("\nModel Performance Summary saved as 'plots/model_performance_summary.csv'")
    return results

# =====================
# Plot Model Comparison
# =====================
def plot_model_comparison(results):
    results = dict(sorted(results.items(), key=lambda item: item[1], reverse=True))
    plt.figure(figsize=(12, 7))
    sns.barplot(x=list(results.keys()), y=list(results.values()), palette='viridis')
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.ylabel('Accuracy', fontsize=14)
    plt.title('Model Comparison', fontsize=16, fontweight='bold')
    plt.grid(axis='y', linestyle='--')
    for i, v in enumerate(results.values()):
        plt.text(i, v + 0.01, f"{v:.2f}", ha='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig('plots/model_comparison.png')
    plt.close()

# =====================
# MAIN
# =====================
def main():
    X, y = load_data()
    X_scaled = preprocess_data(X)

    X_pca, pca = apply_pca(X_scaled, n_components=175)
    plot_pca_variance(pca)

    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42, stratify=y)

    models = initialize_models()
    best_models = hyperparameter_tuning(models, X_train, y_train)

    # stacking_model = stacking_ensemble(best_models)
    # best_models["Stacking Ensemble"] = stacking_model

    results = evaluate_models(best_models, X_train, y_train, X_test, y_test)

    plot_model_comparison(results)

if __name__ == "__main__":
    main()
