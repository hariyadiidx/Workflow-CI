import pandas as pd
import mlflow
import optuna
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import os
import warnings
warnings.filterwarnings('ignore')

mlflow.set_tracking_uri("file://" + os.path.abspath("mlruns"))

print("Memuat dataset untuk pelatihan...")
df = pd.read_csv("dataset_preprocessing/credit_processed.csv")
X = df.drop('Target_Default', axis=1)
y = df['Target_Default']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def objective(trial):
    with mlflow.start_run(nested=True):
        # Optimasi rentang pencarian hyperparameter
        n_estimators = trial.suggest_int("n_estimators", 80, 200)
        max_depth = trial.suggest_int("max_depth", 4, 8)
        min_samples_split = trial.suggest_int("min_samples_split", 4, 12)

        # KUNCI UTAMA: Ditambahkan class_weight='balanced' untuk mengatasi ketimpangan data
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            class_weight='balanced',
            random_state=42
        )
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        mlflow.log_params({
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split
        })
        mlflow.log_metrics({
            "accuracy": acc,
            "f1_score": f1
        })

        custom_env = {
            'channels': ['conda-forge'],
            'dependencies': [
                'python=3.10.12',
                'pip',
                {'pip': ['mlflow==2.19.0', 'pandas', 'scikit-learn', 'numpy', 'optuna', 'matplotlib', 'seaborn']}
            ],
            'name': 'mlflow-env'
        }
        mlflow.sklearn.log_model(clf, "model", conda_env=custom_env)

    return f1 

if __name__ == "__main__":
    print("Memulai Hyperparameter Tuning dengan Optuna (30 Trials)...")
    study = optuna.create_study(direction="maximize")
    # Kita naikkan pencarian ke 30 kali uji coba agar model jauh lebih matang
    study.optimize(objective, n_trials=30)

    print("\n✅ Tuning Selesai! Model terbaik telah disimpan.")
