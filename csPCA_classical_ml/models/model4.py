import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, roc_curve, auc, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
import seaborn as sns  # For confusion matrix visualization


def load_data():
    X = np.load("/home/ibab/PycharmProjects/ML-PROJECT/preprocessing/X.npy")
    y = np.load("/home/ibab/PycharmProjects/ML-PROJECT/preprocessing/y.npy")
    return X, y


def preprocess_data(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled


def apply_pca(X_scaled, n_components):
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    return X_pca, pca


def train_test_split_data(X_pca, y):
    return train_test_split(X_pca, y, test_size=0.2, random_state=42, stratify=y)


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


def hyperparameter_tuning(models, X_train, y_train):
    param_grids = {
        "XGBoost": {
            "n_estimators": [100, 200, 300],
            "learning_rate": [0.01, 0.1, 0.2],
            "max_depth": [3, 5, 7],
            "subsample": [0.6, 0.8, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0]
        },
        "Random Forest": {
            "n_estimators": [100, 200, 300],
            "max_depth": [3, 5, 7],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4]
        },
        "Logistic Regression": {
            "C": [0.1, 1, 10],
            "solver": ['liblinear', 'saga']
        },
        "SVM": {
            "C": [0.1, 1, 10],
            "kernel": ['rbf'],
            "gamma": ['scale', 'auto']
        },
        "KNN": {
            "n_neighbors": [3, 5, 7, 9],
            "weights": ['uniform', 'distance'],
            "algorithm": ['auto', 'ball_tree', 'kd_tree']
        },
        "Gradient Boosting": {
            "n_estimators": [100, 200],
            "learning_rate": [0.01, 0.05, 0.1],
            "max_depth": [3, 5, 7],
        },
        "Decision Tree": {
            "max_depth": [3, 5, 7],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4]
        }
    }

    best_models = {}
    for name, model in models.items():
        grid_search = GridSearchCV(model, param_grids[name], cv=5, n_jobs=-1, scoring='accuracy', verbose=2)
        grid_search.fit(X_train, y_train)
        best_models[name] = grid_search.best_estimator_
    return best_models


def evaluate_models(models, X_train, y_train, X_test, y_test):
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results[name] = acc
    return results


def plot_model_comparison(results):
    results = dict(sorted(results.items(), key=lambda item: item[1], reverse=True))
    models = list(results.keys())
    accuracies = list(results.values())

    plt.figure(figsize=(12, 7))
    colors = plt.cm.viridis(np.linspace(0.3, 0.8, len(models)))
    bars = plt.bar(models, accuracies, color=colors, edgecolor='black')

    for bar in bars:
        height = bar.get_height()
        plt.annotate(f"{height:.2f}",
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 5),
                     textcoords="offset points",
                     ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylabel("Accuracy", fontsize=14)
    plt.title("Model Comparison", fontsize=16, fontweight='bold')
    plt.ylim(0, max(accuracies) + 0.1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


def plot_pca_explained_variance(X_scaled):
    pca = PCA()
    pca.fit(X_scaled)
    plt.figure(figsize=(8, 5))
    plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('PCA - Explained Variance vs Number of Components')
    plt.grid(True)
    plt.show()


def plot_roc_curve(y_test, y_pred_proba):
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.show()


def plot_confusion_matrix(y_test, y_pred, model_name):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1'],
                yticklabels=['Class 0', 'Class 1'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def plot_class_distribution(y):
    plt.figure(figsize=(8, 6))
    sns.countplot(x=y, palette='Set2')
    plt.title('Class Distribution')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.show()


def main():
    X, y = load_data()
    X_scaled = preprocess_data(X)
    n_components = 175
    X_pca, pca = apply_pca(X_scaled, n_components)
    X_train, X_test, y_train, y_test = train_test_split_data(X_pca, y)

    models = initialize_models()
    best_models = hyperparameter_tuning(models, X_train, y_train)

    # Evaluate models and get results
    results = evaluate_models(best_models, X_train, y_train, X_test, y_test)

    # Model Comparison Plot
    plot_model_comparison(results)

    # PCA Explained Variance Plot
    plot_pca_explained_variance(X_scaled)

    # ROC Curve for the best model (use the best model based on highest accuracy)
    best_model_name = max(results, key=results.get)
    best_model = best_models[best_model_name]
    best_model.fit(X_train, y_train)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    plot_roc_curve(y_test, y_pred_proba)

    # Confusion Matrix for the best model
    y_pred = best_model.predict(X_test)
    plot_confusion_matrix(y_test, y_pred, best_model_name)

    # Class Distribution Plot
    plot_class_distribution(y)


if __name__ == '__main__':
    main()
