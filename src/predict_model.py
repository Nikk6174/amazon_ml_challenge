import pandas as pd
import numpy as np
import lightgbm as lgb
import os
import joblib # For saving the model

def main():
    """
    Trains the final model on all training data, generates predictions on the
    test set, saves the model, and creates the submission file.
    """
    print("--- Starting Final Model Training & Prediction Pipeline ---")

    # --- 1. Define File Paths (Relative to project root) ---
    TRAIN_FEATURES_PATH = 'data/processed/train_features_text.csv'
    TRAIN_EMBEDDINGS_PATH = 'data/processed/text_embeddings_distilbert.npy'
    TEST_RAW_PATH = 'data/raw/test.csv'
    TEST_FEATURES_PATH = 'data/processed/test_features_text.csv'
    TEST_EMBEDDINGS_PATH = 'data/processed/test_text_embeddings_distilbert.npy'
    MODEL_OUTPUT_PATH = 'models/lgbm_text_only_final.pkl'
    SUBMISSION_PATH = 'submissions/submission_text_only.csv'

    # --- 2. Load and Prepare Training Data ---
    print("Loading and preparing training data...")
    train_df = pd.read_csv(TRAIN_FEATURES_PATH)
    train_text_embeddings = np.load(TRAIN_EMBEDDINGS_PATH)
    
    train_numerical_features = train_df[['item_size', 'pack_count', 'total_quantity']].values
    X_train = np.hstack([train_numerical_features, train_text_embeddings])
    y_train_log = np.log1p(train_df['price'].values)
    
    print(f"Training data shape: {X_train.shape}")

    # --- 3. Train the Final LightGBM Model ---
    print("\nTraining LightGBM model on the full dataset...")
    lgbm = lgb.LGBMRegressor(
        objective='regression_l1',
        n_estimators=1500,
        learning_rate=0.03,
        num_leaves=40,
        random_state=42,
        n_jobs=-1,
        colsample_bytree=0.8,
        subsample=0.8
    )
    lgbm.fit(X_train, y_train_log)
    print("Model training complete.")

    # --- 4. Save the Trained Model ---
    joblib.dump(lgbm, MODEL_OUTPUT_PATH)
    print(f"Model saved to: {MODEL_OUTPUT_PATH}")

    # --- 5. Load and Prepare Test Data ---
    print("\nLoading and preparing test data...")
    test_raw_df = pd.read_csv(TEST_RAW_PATH)
    test_features_df = pd.read_csv(TEST_FEATURES_PATH)
    test_text_embeddings = np.load(TEST_EMBEDDINGS_PATH)
    
    test_numerical_features = test_features_df[['item_size', 'pack_count', 'total_quantity']].values
    X_test = np.hstack([test_numerical_features, test_text_embeddings])
    
    print(f"Test data shape: {X_test.shape}")

    # --- 6. Generate Predictions ---
    print("\nGenerating predictions on the test set...")
    log_predictions = lgbm.predict(X_test)
    final_predictions = np.expm1(log_predictions)
    final_predictions[final_predictions < 0] = 0 # Ensure no negative prices
    print("Predictions generated.")

    # --- 7. Create and Save Submission File ---
    print("\nCreating submission file...")
    submission_df = pd.DataFrame({
        'sample_id': test_raw_df['sample_id'],
        'price': final_predictions
    })
    
    # Ensure the 'submissions' directory exists
    os.makedirs('submissions', exist_ok=True)
    submission_df.to_csv(SUBMISSION_PATH, index=False)
    
    print(f"✅ Submission file successfully created at: {SUBMISSION_PATH}")
    print(submission_df.head())

if __name__ == '__main__':
    main()