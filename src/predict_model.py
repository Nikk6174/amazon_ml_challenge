# src/models/predict_model.py
import os
import time
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge

# Optional imports for a full ensemble
try:
    import xgboost as xgb
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

try:
    from catboost import CatBoostRegressor
    _HAS_CAT = True
except ImportError:
    _HAS_CAT = False

# ---------------- Metric Function ----------------
def smape(y_true, y_pred, eps=1e-9):
    """
    Calculates the Symmetric Mean Absolute Percentage Error (SMAPE).
    A robust implementation to avoid division by zero.
    """
    numerator = np.abs(y_true - y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    # Add a small epsilon where the denominator is close to zero
    denominator = np.where(denominator < eps, eps, denominator)
    return np.mean(numerator / denominator) * 100.0

# ---------------- Model Factory Functions ----------------
def make_lgb(use_gpu=True):
    """Creates a LightGBM Regressor with good baseline parameters."""
    params = dict(
        objective='regression_l1', n_estimators=1500, learning_rate=0.03,
        num_leaves=40, random_state=42, n_jobs=-1,
        colsample_bytree=0.8, subsample=0.8
    )
    if use_gpu:
        params.update(device='gpu')
    return lgb.LGBMRegressor(**params)

def make_xgb(use_gpu=True):
    """Creates an XGBoost Regressor."""
    params = {
        "random_state": 42, "n_estimators": 1000, "learning_rate": 0.03,
        "verbosity": 0, "n_jobs": -1,
    }
    if use_gpu:
        params["tree_method"] = "gpu_hist"
    return xgb.XGBRegressor(**params)

def make_cat(use_gpu=True):
    """Creates a CatBoost Regressor."""
    params = {
        "random_seed": 42, "iterations": 2000, "learning_rate": 0.03,
        "verbose": 0,
    }
    if use_gpu:
        params["task_type"] = "GPU"
    return CatBoostRegressor(**params)

# ---------------- Cross-Validation / OOF Routine ----------------
def get_oof_preds(name, factory_fn, X, y_log, X_test, n_splits=5, use_gpu=True):
    """
    Performs K-Fold cross-validation to generate out-of-fold (OOF) predictions.
    """
    oof_preds = np.zeros(X.shape[0], dtype=float)
    test_preds_folds = []
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X)):
        print(f"[{name}] Fold {fold+1}/{n_splits}")
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr = y_log[tr_idx]

        try:
            model = factory_fn(use_gpu)
            model.fit(X_tr, y_tr)
        except Exception as e:
            print(f"[{name}] GPU training failed: {e}. Falling back to CPU.")
            model = factory_fn(False)
            model.fit(X_tr, y_tr)

        oof_preds[val_idx] = model.predict(X_val)
        test_preds_folds.append(model.predict(X_test))

    test_preds_mean = np.mean(np.vstack(test_preds_folds), axis=0)
    cv_sm = smape(np.expm1(y_log), np.expm1(oof_preds))
    print(f"[{name}] CV SMAPE: {cv_sm:.4f}%")
    return oof_preds, test_preds_mean, cv_sm

# In src/models/predict_model.py

# In src/models/predict_model.py

# In src/models/predict_model.py

def main():
    print("=== Final Multimodal Model Pipeline (Text + Image + Stacking) ===")
    t0 = time.time()

    # --- Define all file paths ---
    TRAIN_FEATURES_PATH = 'data/processed/train_features_text.csv'
    TRAIN_TEXT_EMBED_PATH = 'data/processed/text_embeddings_distilbert.npy'
    TRAIN_IMG_EMBED_PATH = 'data/processed/image_embeddings_resnet50.npy'
    TRAIN_IMG_IDS_PATH = 'data/processed/image_ids_order.npy' # <-- NEW: Path to your image IDs

    TEST_RAW_PATH = 'data/raw/test.csv'
    TEST_FEATURES_PATH = 'data/processed/test_features_text.csv'
    TEST_TEXT_EMBED_PATH = 'data/processed/test_text_embeddings_distilbert.npy'
    TEST_IMG_EMBED_PATH = 'data/processed/test_image_embeddings_resnet50.npy'
    TEST_IMG_IDS_PATH = 'data/processed/test_image_ids_order.npy' # <-- NEW: Path for test image IDs

    SUBMISSION_PATH = 'submissions/submission_final_ensemble.csv'
    os.makedirs('submissions', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    # --- 1. Load all training data ---
    print("Loading all training features and embeddings...")
    train_df = pd.read_csv(TRAIN_FEATURES_PATH)
    train_text_emb = np.load(TRAIN_TEXT_EMBED_PATH)
    train_img_emb = np.load(TRAIN_IMG_EMBED_PATH)
    train_img_ids = np.load(TRAIN_IMG_IDS_PATH) # <-- NEW: Load the image IDs

    # --- 2. Align all training features using a merge ---
    print("Aligning training data...")
    # Create a DataFrame for the image embeddings with their corresponding IDs
    img_emb_df = pd.DataFrame(train_img_emb, index=train_img_ids.astype(int))
    
    # Merge the main DataFrame with the image embeddings using the sample_id
    train_df = train_df.merge(img_emb_df, left_on='sample_id', right_index=True, how='left')

    # --- 3. Handle the 4 missing images by filling NaNs ---
    # We fill missing image embeddings with the average embedding value for each feature
    image_feature_columns = img_emb_df.columns
    train_df[image_feature_columns] = train_df[image_feature_columns].fillna(train_df[image_feature_columns].mean())
    print(f"Missing image embeddings handled: {train_df[image_feature_columns].isnull().sum().sum()} NaNs remain.")

    # --- 4. Reconstruct the final feature matrix (X_train) ---
    required_cols = ['item_size', 'pack_count', 'total_quantity', 'unit_target_encoded', 'brand_target_encoded']
    existing_cols = [c for c in required_cols if c in train_df.columns]
    X_num = train_df[existing_cols].values.astype(float)
    X_img = train_df[image_feature_columns].values
    
    # Now all arrays are guaranteed to have 75,000 rows
    X_train = np.hstack([X_num, train_text_emb, X_img])
    y_train_log = np.log1p(train_df['price'].values)
    print("X_train shape:", X_train.shape)

    # --- 5. Repeat the exact same process for the test data ---
    print("\nLoading and aligning test data...")
    test_raw = pd.read_csv(TEST_RAW_PATH)
    test_df = pd.read_csv(TEST_FEATURES_PATH)
    test_text_emb = np.load(TEST_TEXT_EMBED_PATH)
    test_img_emb = np.load(TEST_IMG_EMBED_PATH)
    test_img_ids = np.load(TEST_IMG_IDS_PATH)

    test_img_emb_df = pd.DataFrame(test_img_emb, index=test_img_ids.astype(int))
    test_df = test_df.merge(test_img_emb_df, left_on='sample_id', right_index=True, how='left')
    test_df[image_feature_columns] = test_df[image_feature_columns].fillna(train_df[image_feature_columns].mean()) # Use train mean to fill test NaNs

    X_test_num = test_df[existing_cols].values.astype(float)
    X_test_img = test_df[image_feature_columns].values
    X_test = np.hstack([X_test_num, test_text_emb, X_test_img])
    print("X_test shape:", X_test.shape)
    
    # ... The rest of your script (OOF CV, training, submission) remains exactly the same ...
    model_factories = [('lgbm', make_lgb), ('xgb', make_xgb) if _HAS_XGB else None, ('cat', make_cat) if _HAS_CAT else None]
    model_factories = [m for m in model_factories if m is not None]
    
    oof_preds, test_preds, cv_scores = {}, {}, {}
    for name, factory in model_factories:
        print("\n" + "-" * 60)
        oof, testp, cv_sm = get_oof_preds(name, factory, X_train, y_train_log, X_test)
        oof_preds[name], test_preds[name], cv_scores[name] = oof, testp, cv_sm

    print("\nTraining stacking meta-learner (Ridge) on OOF predictions...")
    oof_matrix = np.vstack([oof_preds[n] for n in cv_scores.keys()]).T
    test_matrix = np.vstack([test_preds[n] for n in cv_scores.keys()]).T
    
    meta_learner = Ridge(alpha=1.0)
    meta_learner.fit(oof_matrix, y_train_log)
    
    stacked_oof_price = np.expm1(meta_learner.predict(oof_matrix))
    print(f"Stacked meta-learner OOF SMAPE: {smape(np.expm1(y_train_log), stacked_oof_price):.4f}%")

    print("\nGenerating final predictions using the stacked model...")
    stacked_log_preds = meta_learner.predict(test_matrix)
    final_price_preds = np.expm1(stacked_log_preds)
    final_price_preds[final_price_preds < 0] = 0.0

    submission = pd.DataFrame({'sample_id': test_raw['sample_id'], 'price': final_price_preds})
    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"\n✅ Submission file successfully created at: {SUBMISSION_PATH}")
    print(submission.head())
    
    total_time = time.time() - t0
    print(f"All done in {total_time / 60:.2f} minutes.")

if __name__ == "__main__":
    main()