
import pandas as pd
import mlflow
import dagshub
import optuna
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import os
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# KONFIGURASI DAGSHUB & MLFLOW (KRITERIA ADVANCED)
# ==============================================================================
# HAPUS TANDA PAGAR (#) pada dua baris di bawah ini dan isi dengan kredensial Anda
# jika sudah siap menjalankan eksperimen untuk dikirim ke DagsHub.

# dagshub.init(repo_owner='USERNAME_GITHUB_ANDA', repo_name='NAMA_REPO_DAGSHUB_ANDA', mlflow=True)
# mlflow.set_experiment("Eksperimen_Credit_Scoring_Optuna")

# Untuk uji coba lokal di Colab saat ini, kita gunakan tracking lokal terlebih dahulu:
mlflow.set_tracking_uri("file://" + os.path.abspath("mlruns"))
mlflow.set_experiment("Eksperimen_Lokal")

# ==============================================================================
# 1. DATA LOADING & SPLITTING
# ==============================================================================
print("Memuat dataset untuk pelatihan...")
# Menggunakan path relatif agar sesuai dengan folder kerja Kriteria 2 dan 3
df = pd.read_csv("dataset_preprocessing/credit_processed.csv")
X = df.drop('Target_Default', axis=1)
y = df['Target_Default']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y);

# ==============================================================================
# 2. DEFINISI FUNGSI OBJEKTIF OPTUNA & MLFLOW LOGGING
# ==============================================================================
def objective(trial):
    # MLflow start_run(nested=True) penting saat digabungkan dengan Optuna
    with mlflow.start_run(nested=True):

        # A. Mendefinisikan ruang pencarian Hyperparameter
        n_estimators = trial.suggest_int("n_estimators", 50, 150)
        max_depth = trial.suggest_int("max_depth", 3, 10)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 10)

        # B. Inisialisasi dan Pelatihan Model
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=42
        )
        clf.fit(X_train, y_train);

        # C. Prediksi dan Evaluasi
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred);

        # D. MANUAL LOGGING (KRITERIA SKILLED/ADVANCED - Dilarang pakai autolog)
        mlflow.log_params({
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split
        })
        mlflow.log_metrics({
            "accuracy": acc,
            "f1_score": f1
        });

        # E. MEMBUAT ARTEFAK TAMBAHAN 1: Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(4,3))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f"Confusion Matrix (Trial {trial.number})")
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        cm_path = f"confusion_matrix_trial_{trial.number}.png"
        plt.savefig(cm_path)
        mlflow.log_artifact(cm_path)
        plt.close();

        # F. MEMBUAT ARTEFAK TAMBAHAN 2: Feature Importance
        plt.figure(figsize=(5,3))
        importances = clf.feature_importances_
        sns.barplot(x=importances, y=X.columns, palette='viridis')
        plt.title(f"Feature Importance (Trial {trial.number})")
        feat_path = f"feature_importance_trial_{trial.number}.png"
        plt.savefig(feat_path)
        mlflow.log_artifact(feat_path)
        plt.close();

        # G. Log Model Machine Learning
        mlflow.sklearn.log_model(clf, "model");

    return f1 # Kita mengoptimalkan berdasarkan skor F1 (karena target imbalanced)

# ==============================================================================
# 3. EKSEKUSI OPTIMASI HYPERPARAMETER
# ==============================================================================
if __name__ == "__main__":
    print("Memulai Hyperparameter Tuning dengan Optuna...")
    # Menggunakan 3 trials untuk demonstrasi/uji coba agar cepat.
    # Di eksperimen sesungguhnya, gunakan 10-50 trials.
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=3);

    print("\n✅ Tuning Selesai!") # Fix: double escape the backslash
    print("Parameter Terbaik:", study.best_params)
    print("F1-Score Terbaik:", study.best_value)
