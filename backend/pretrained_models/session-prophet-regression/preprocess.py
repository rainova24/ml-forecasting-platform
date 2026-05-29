import pandas as pd

def preprocess_evaluate(df: pd.DataFrame, target_col: str, date_col: str, test_size: float):
    """
    Preprocess data for session-prophet-regression evaluation.
    Requires ds column and categorical regressors.
    Aligned with prophet_time_series_planning.ipynb.
    """
    df = df.copy()
    
    # 1. Date preprocessing
    # Note: main.py already handles Indonesian month translation and 
    # converts the date column to datetime objects before this function is called.
    df['ds'] = df[date_col]
    df['y'] = df[target_col]
    
    # 2. Add Dummy features for categories
    # One-hot encode the category column to use as extra regressors in Prophet
    categorical_columns = pd.get_dummies(df['category'], prefix='cat', dtype=int)
    df = pd.concat([df, categorical_columns], axis=1)

    # CRITICAL FIX: Prophet requires the exact column names used during training.
    # We explicitly define the expected features from the notebook and fill missing ones with 0.
    expected_features = ['cat_Burnout', 'cat_Interrupted', 'cat_Learning', 'cat_Normal']
    for col in expected_features:
        if col not in df.columns:
            df[col] = 0

    category_features = expected_features
    print("Engineered Regressors:", category_features)
    
    # 3. Train/Test Split
    n = len(df)
    test_size_clamped = max(0.05, min(0.5, test_size))
    test_start_idx = int(n * (1.0 - test_size_clamped))
    
    test_df = df.iloc[test_start_idx:]
    
    # 4. Prepare Prophet Input Dataframe
    print(category_features)
    prophet_test = test_df[['ds'] + category_features]
    y_test = test_df['y'].values
    test_dates = test_df['ds'].dt.strftime('%Y-%m-%d').tolist()
    
    # For Prophet, the 'evaluate_pretrained_with_dataset' endpoint in main.py expects X_test to be the dataframe
    return prophet_test, y_test, test_dates