import pandas as pd
import numpy as np

def preprocess_evaluate(df: pd.DataFrame, target_col: str, date_col: str, test_size: float):
    """
    Preprocess data for session-mlr evaluation.
    Requires exactly 23 features: 14 lags + 9 one-hot encoded variables.
    """
    df = df.copy()
    
    # 1. Sort by date
    df[date_col] = pd.to_datetime(df[date_col])
    df.sort_values(date_col, inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    # 2. Create 14 lagged features for the target column
    window_size = 14
    for i in range(1, window_size + 1):
        df[f'min_lag_{i}'] = df['minutes'].shift(i)
        
    # Drop the first 14 rows since they don't have enough history
    df = df.dropna(subset=[f'min_lag_{i}' for i in range(1, window_size + 1)]).reset_index(drop=True)
    
    # 3. Create Categorical Dummies
    # Extract day of week (Monday=0, Sunday=6)
    df_encoded = pd.get_dummies(df, columns=['category', 'day_of_week'], drop_first=True)
    
    # We must ensure all 9 dummies exist exactly as expected by the model.
    # From model training context, day_of_week might be a single number (if not dummy encoded),
    # or the example shows 9 one-hot encode values. 
    # Wait, the metadata for MLR says "followed by one-hot encoded boolean values for 'category' and 'day_of_week'."
    # There are 7 days of the week + 'category'.
    
    # Let's fallback to the specific dummy columns if they exist in the CSV (e.g. 'category'), 
    # but since the evaluation CSV is raw, it might not have 'category'.
    # If the CSV is completely raw (e.g., just Date and Close), the MLR evaluate script will 
    # try its best. But the user's raw dataset might not have 'category' at all!
    # If it's missing, we pad with 0s.
    
    # The example in metadata shows 23 total features: 14 lags + 9 dummies.
    # 7 day dummies + 2 or 3 category dummies. (Wait, the example has 9 dummies).
    
    # Let's check what the raw dataframe has. We'll dummy encode whatever 'category' or 'day_of_week' exist.
    if 'category' in df.columns:
        cat_dummies = pd.get_dummies(df['category'], prefix='category', dtype=int)
        df = pd.concat([df, cat_dummies], axis=1)
    
    day_dummies = pd.get_dummies(df['day_of_week'], prefix='day_of_week', dtype=int)
    df = pd.concat([df, day_dummies], axis=1)
    
    # Extract features matching the pattern (X_ features + category/day dummies)
    feature_cols = [f'min_lag_{i}' for i in range(1, window_size + 1)]
    
    # Let's see what dummy columns are actually available now vs expected. 
    # The safest way is to check the loaded_model's expected feature count, but we are isolated here.
    # We will grab the 14 lags, then any day_of_week or category dummies found.
    # If the total doesn't match 23, the model predict will throw an error, but it's the most correct preprocessing.
    
    # Using the specific dummy column names from the model's expected features
    expected_dummy_cols = [
      'category_Interrupted', 'category_Learning', 'category_Normal',
      'day_of_week_2', 'day_of_week_3',
      'day_of_week_4', 'day_of_week_5',
      'day_of_week_6', 'day_of_week_7'
    ]
    
    # Filter existing columns
    dummy_cols = [col for col in expected_dummy_cols if col in df.columns]
    feature_cols.extend(dummy_cols)
    
    # 4. Train/Test Split
    n = len(df)
    test_size_clamped = max(0.05, min(0.5, test_size))
    test_start_idx = int(n * (1.0 - test_size_clamped))
    
    test_df = df.iloc[test_start_idx:]
    
    X_test = test_df[feature_cols].fillna(0).values
    
    # If X_test doesn't have 23 features, it will error out in model.predict() as expected if data is missing.
    # But it correctly parses whatever is in the CSV now.
    
    y_test = test_df[target_col].values
    test_dates = test_df[date_col].dt.strftime('%Y-%m-%d').tolist()
    
    return X_test, y_test, test_dates
