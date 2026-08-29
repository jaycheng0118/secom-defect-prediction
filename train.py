"""
Semiconductor Yield Anomaly Detection & Root Cause Analysis Pipeline
Dataset: UCI SECOM (Semiconductor Manufacturing)
Model: LightGBM with Cost-Sensitive Weighting & TreeSHAP Attribution
"""

import os
import urllib.request
import numpy as np
import pandas as pd
import lightgbm as lgb
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report,
    precision_recall_curve,
    auc,
    roc_auc_score,
    confusion_matrix
)

# 1. 建立輸出資料夾
os.makedirs("assets", exist_ok=True)
os.makedirs("data", exist_ok=True)


def load_secom_data():
    """自動下載並載入 UCI SECOM 資料集 (590 感測器特徵)"""
    data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/secom/secom.data"
    label_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/secom/secom_labels.data"
    
    data_path = "data/secom.data"
    label_path = "data/secom_labels.data"

    if not os.path.exists(data_path):
        print("📥 正在下載 UCI SECOM 資料集...")
        urllib.request.urlretrieve(data_url, data_path)
        urllib.request.urlretrieve(label_url, label_path)
        print("✅ 下載完成！")

    # 讀取特徵與標籤 (SECOM 標籤: -1 代表良品/Pass, 1 代表不良品/Fail)
    X = pd.read_csv(data_path, sep=" ", header=None)
    y_raw = pd.read_csv(label_path, sep=" ", header=None)
    
    # 轉換標籤: 0 為 Pass, 1 為 Fail (異常)
    y = (y_raw[0] == 1).astype(int)
    
    # 為感測器命名 Sensor_0 ~ Sensor_589
    X.columns = [f"Sensor_{i}" for i in range(X.shape[1])]
    
    print(f"📊 原始資料維度: 特徵 {X.shape} | 正樣本比例 (Fail Rate): {y.mean():.2%}")
    return X, y


def preprocess_features(X_train, X_test, missing_threshold=0.5):
    """資料清洗：過濾高缺失/零變異特徵、中位數填補"""
    print("⚙️ 執行特徵工程與缺失值清洗...")
    
    # 1. 移除缺失率超過 50% 的感測器特徵
    missing_ratio = X_train.isnull().mean()
    valid_cols = missing_ratio[missing_ratio < missing_threshold].index
    X_train_filtered = X_train[valid_cols]
    X_test_filtered = X_test[valid_cols]

    # 2. 移除零變異特徵 (Zero Variance / 常數特徵)
    variance = X_train_filtered.var()
    non_zero_cols = variance[variance > 1e-5].index
    X_train_filtered = X_train_filtered[non_zero_cols]
    X_test_filtered = X_test_filtered[non_zero_cols]

    # 3. 中位數填補 (避免資料洩漏，以訓練集統計量 fit)
    imputer = SimpleImputer(strategy="median")
    X_train_imputed = pd.DataFrame(
        imputer.fit_transform(X_train_filtered),
        columns=non_zero_cols,
        index=X_train.index
    )
    X_test_imputed = pd.DataFrame(
        imputer.transform(X_test_filtered),
        columns=non_zero_cols,
        index=X_test.index
    )

    print(f"✨ 特徵篩選完成: 剩餘 {X_train_imputed.shape[1]} 個關鍵感測器特徵")
    return X_train_imputed, X_test_imputed


def train_lightgbm_model(X_train, y_train, X_test, y_test):
    """訓練代價敏感 LightGBM 模型以應對極端不平衡數據"""
    print("🚀 開始訓練 LightGBM 異常偵測模型...")

    # 計算正負樣本權重比 (約 1:14)
    pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)

    model = lgb.LGBMClassifier(
        n_estimators=150,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=6,
        scale_pos_weight=pos_weight,  # 加強對良率異常的懲罰權重
        random_state=42,
        verbosity=-1
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric="auc"
    )

    # 預測機率
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # 根據半導體業務情境自訂決策閥值 (重視 Recall)
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
    pr_auc_val = auc(recall, precision)
    roc_auc_val = roc_auc_score(y_test, y_pred_proba)

    y_pred_binary = (y_pred_proba >= 0.5).astype(int)

    print("\n" + "=" * 50)
    print("📈 模型驗證結果 (Test Evaluation Metrics):")
    print(f"• PR-AUC (Precision-Recall AUC): {pr_auc_val:.4f}")
    print(f"• ROC-AUC Score:                 {roc_auc_val:.4f}")
    print("\n分類報告 (Classification Report):")
    print(classification_report(y_test, y_pred_binary, target_names=["Pass (良品)", "Fail (瑕疵)"]))
    print("=" * 50)

    return model, y_pred_proba


def explain_with_shap(model, X_train, X_test):
    """利用 TreeSHAP 進行根本原因分析 (RCA) 並輸出圖表"""
    print("\n🔍 正在執行 SHAP 可解釋性分析 (Root Cause Analysis)...")
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # 處理 binary classification 的 SHAP 結構
    if isinstance(shap_values, list):
        shap_values_to_plot = shap_values[1]
    else:
        shap_values_to_plot = shap_values

    # 1. 產出全局特徵重要性圖表 (Summary Plot)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values_to_plot, X_test, max_display=10, show=False)
    plt.title("Top 10 Sensors Impacting Yield Anomaly (SHAP Attribution)", fontsize=12)
    plt.tight_layout()
    
    summary_plot_path = "assets/shap_summary.png"
    plt.savefig(summary_plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"📊 全局特徵歸因圖已儲存至: {summary_plot_path}")


def main():
    # 1. 讀取數據
    X, y = load_secom_data()

    # 2. 切分訓練集與測試集 (80/20 Stratified Split)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. 特徵預處理
    X_train_proc, X_test_proc = preprocess_features(X_train, X_test)

    # 4. 訓練模型
    model, _ = train_lightgbm_model(X_train_proc, y_train, X_test_proc, y_test)

    # 5. SHAP 歸因分析
    explain_with_shap(model, X_train_proc, X_test_proc)
    
    print("\n🎉 Pipeline 執行完畢！可將產出的 assets/ 圖片與程式碼同步至 GitHub。")


if __name__ == "__main__":
    main()
