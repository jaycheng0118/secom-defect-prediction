"""
專案名稱：UCI SECOM 半導體良率異常偵測與感測器根因分析 (RCA) Pipeline
作者：Jay
目前版次：v1.2.0 (Last Updated: 2026-09)

版本更新紀錄 (Changelog):
- v1.0.0: 完成基本資料載入、初版 LightGBM 訓練與混淆矩陣輸出。
- v1.1.0: 修正資料洩漏問題（將 Imputer 與常數過濾改在 train set fit）；加入 scale_pos_weight 處理 ~14:1 不平衡。
- v1.2.0: 
    1. 新增 PR-AUC 作為少數類評估主力指標。
    2. 樹模型加入 max_depth=4 與 subsample 抑制過擬合（樣本僅 1567 筆）。
    3. 串接 TreeSHAP 萃取影響良率前 12 大異常感測器，輸出特徵蜂群圖。

note：
1. 感測器特徵達 590 個但樣本數少，標準樹模型極易死背雜訊。
2. 缺失值嚴重，雖 LightGBM 原生支援 NaN，但 TreeExplainer 在部分空值路徑會不穩定，前處理統一中位數補值。
3. 瑕疵品僅約 6.6%，不可單看 Accuracy 或 ROC-AUC，須關注 Precision-Recall 權衡。
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
    confusion_matrix,
)

# 建立輸出目錄
os.makedirs("output", exist_ok=True)
os.makedirs("data", exist_ok=True)


def get_raw_secom():
    base_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/secom/"
    data_file = "data/secom.data"
    label_file = "data/secom_labels.data"

    if not os.path.exists(data_file):
        print("[INFO] 本機沒找到快取，從 UCI 抓取 SECOM 資料集...")
        urllib.request.urlretrieve(base_url + "secom.data", data_file)
        urllib.request.urlretrieve(base_url + "secom_labels.data", label_file)

    X = pd.read_csv(data_file, sep=r"\s+", header=None)
    y_raw = pd.read_csv(label_file, sep=r"\s+", header=None)

    # 原標籤: -1 為 Pass(良品), 1 為 Fail(瑕疵品) -> 轉為 0 與 1
    y = (y_raw[0] == 1).astype(int)
    X.columns = [f"sensor_{i}" for i in range(X.shape[1])]

    print(f"[DEBUG] 載入完成: X={X.shape}, 正例(Fail)佔比: {y.mean():.4f}")
    return X, y


def clean_and_impute(X_train, X_test, missing_cutoff=0.45, var_cutoff=1e-5):
    """
    特徵前處理 (v1.1.0 重構):
    1. 缺失率 > 45% 視為無效感測器直接剔除。
    2. 剔除常數/零變異特徵。
    3. 統計量與 Imputer 一律僅在 X_train 上 fit，嚴格杜絕 Data Leakage。
    """
    # 1. 高缺失過濾
    na_ratios = X_train.isnull().mean()
    keep_cols = na_ratios[na_ratios < missing_cutoff].index
    print(f"[INFO] 剔除缺失率 >= {missing_cutoff*100}% 的感測器: {X_train.shape[1] - len(keep_cols)} 欄")

    X_tr_sub = X_train[keep_cols]
    X_te_sub = X_test[keep_cols]

    # 2. 零變異過濾 (防範單一值造成樹分歧無意義)
    col_vars = X_tr_sub.var()
    valid_cols = col_vars[col_vars > var_cutoff].index
    print(f"[INFO] 剔除近似零變異感測器: {len(keep_cols) - len(valid_cols)} 欄")

    X_tr_sub = X_tr_sub[valid_cols]
    X_te_sub = X_te_sub[valid_cols]

    # 3. 中位數填補 (滿足 TreeExplainer 數值穩定性)
    imputer = SimpleImputer(strategy="median")
    X_tr_imp = pd.DataFrame(
        imputer.fit_transform(X_tr_sub),
        columns=valid_cols,
        index=X_train.index
    )
    X_te_imp = pd.DataFrame(
        imputer.transform(X_te_sub),
        columns=valid_cols,
        index=X_test.index
    )

    print(f"[INFO] 最終進入模型特徵數: {X_tr_imp.shape[1]}")
    return X_tr_imp, X_te_imp


def fit_lgbm(X_train, y_train, X_test, y_test):
    # 正負樣本加權計算 (~14:1)
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_weight = neg_count / pos_count

    # v1.2.0 防禦性參數配置：限制樹深與隨機採樣抑制 Overfitting
    clf = lgb.LGBMClassifier(
        n_estimators=180,
        learning_rate=0.03,
        num_leaves=15,
        max_depth=4,
        scale_pos_weight=scale_weight,
        subsample=0.8,
        colsample_bytree=0.7,
        random_state=42,
        verbosity=-1,
    )

    clf.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)],
    )

    probs = clf.predict_proba(X_test)[:, 1]

    # 驗證指標計算
    prec, rec, _ = precision_recall_curve(y_test, probs)
    pr_auc = auc(rec, prec)
    roc_score = roc_auc_score(y_test, probs)

    print("-" * 50)
    print(f"[EVAL] 驗證集 PR-AUC : {pr_auc:.4f}")
    print(f"[EVAL] 驗證集 ROC-AUC: {roc_score:.4f}")

    # 以 0.5 預設閥值檢視初步混淆矩陣
    preds_default = (probs >= 0.5).astype(int)
    print("\n[EVAL] 混淆矩陣 (Threshold=0.5):")
    print(confusion_matrix(y_test, preds_default))
    print("\n[EVAL] 分類表現報告:")
    print(classification_report(y_test, preds_default, digits=4))
    print("-" * 50)

    return clf, probs


def run_shap_analysis(model, X_train, X_test):
    """
    Root Cause Analysis (RCA):
    利用 TreeSHAP 追溯推升良率異常機率最劇烈的前 12 大感測器特徵
    """
    print("[INFO] 正在計算測試集 TreeSHAP 歸因權重...")
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_test)

    # 確保抓取正例 (Fail=1) 的歸因陣列
    if isinstance(shap_vals, list):
        shap_vals_target = shap_vals[1]
    else:
        shap_vals_target = shap_vals

    # 繪製全域特徵重要性蜂群圖
    plt.figure(figsize=(9, 5))
    shap.summary_plot(shap_vals_target, X_test, max_display=12, show=False)
    plt.title("Top 12 Sensor Attributions on Yield Failure (SHAP)", fontsize=11, pad=10)
    plt.tight_layout()

    out_path = "output/sensor_shap_summary_v1.2.0.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[INFO] SHAP 歸因分析圖已產出: {out_path}")


if __name__ == "__main__":
    # 1. 載入原始資料
    X_raw, y_raw = get_raw_secom()

    # 2. 分層切分訓練集與測試集 (80/20 Stratified)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_raw, y_raw, test_size=0.20, random_state=42, stratify=y_raw
    )

    # 3. 資料清洗與補值
    X_tr_clean, X_te_clean = clean_and_impute(X_tr, X_te)

    # 4. 模型訓練與少數類評估
    model, test_probs = fit_lgbm(X_tr_clean, y_tr, X_te_clean, y_te)

    # 5. TreeSHAP 根因分析產圖
    run_shap_analysis(model, X_tr_clean, X_te_clean)
