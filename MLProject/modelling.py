import pandas as pd
import numpy as np
import os
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

if __name__ == "__main__":
    print("=== Memulai Eksekusi Skrip Pelatihan Model Standar (modelling.py) ===")
    
    # 1. KONFIGURASI TRACKING MLFLOW LOKAL (Sesuai Alamat Localhost Kriteria 2)
    # Menggunakan path absolut agar aman dieksekusi di Codespaces maupun runner GitHub Actions
    mlflow.set_tracking_uri("file://" + os.path.abspath("mlruns"))
    # mlflow.set_experiment("Eksperimen_Lokal")
    
    # 2. MENGAKTIFKAN AUTOLOG (Wajib di kriteria Basic untuk file modelling.py)
    mlflow.autolog()
    print("🔹 MLflow Autolog berhasil diaktifkan.")

    # 3. DATA LOADING (Membaca data hasil preprocessing)
    # Jalur fleksibel (fallback path) agar tidak terjadi path error di berbagai environment
    dataset_path = "dataset_preprocessing/credit_processed.csv"
    if not os.path.exists(dataset_path):
        dataset_path = "Membangun_model/dataset_preprocessing/credit_processed.csv"
    if not os.path.exists(dataset_path):
        dataset_path = "namadataset_preprocessing/credit_processed.csv"
    if not os.path.exists(dataset_path):
        dataset_path = "Eksperimen_SML_Hariyadi/dataset_preprocessing/credit_processed.csv"

    print(f"🔹 Memuat dataset bersih dari: {dataset_path}")
    df = pd.read_csv(dataset_path)
    
    # Memisahkan Fitur (X) dan Target (y)
    X = df.drop('Target_Default', axis=1)
    y = df['Target_Default']

    # 4. DATA SPLITTING (Membagi data training dan testing dengan Stratify)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 5. RETRAINING MODEL & RECORDING VIA MLFLOW
    print("🔹 Memulai training model Random Forest standar...")
    with mlflow.start_run() as run:
        # Ditambahkan class_weight='balanced' untuk mengatasi imbalanced dataset (F1-Score 0)
        clf = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            class_weight='balanced',  # Solusi penyeimbang bobot kelas target
            random_state=42
        )
        clf.fit(X_train, y_train)
        
        # 6. EVALUASI MODEL
        y_pred = clf.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        
        print("\n==========================================")
        print("          HASIL EVALUASI MODEL STANDAR    ")
        print("==========================================")
        print(f"-> Accuracy  : {acc:.4f}")
        print(f"-> F1-Score  : {f1:.4f} (Sudah Diperbaiki)")
        print(f"-> Precision : {precision:.4f}")
        print(f"-> Recall    : {recall:.4f}")
        print("==========================================")
        
        print("\n✅ Proses eksekusi modelling.py selesai!")
        print(f"ℹ️ Seluruh log dan biner model otomatis disimpan di bawah Run ID: {run.info.run_id}")
