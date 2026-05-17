import joblib
import pandas as pd
import os

def get_forecast():
    # 1. Paths relative to the PROJECT 1 root
    model_path = os.path.join('models', 'random_forest_model.pkl')
    data_path = os.path.join('Data', 'processed_hhs_data.csv')

    # 2. Load the artifacts
    model = joblib.load(model_path)
    df = pd.read_csv(data_path, index_col='Date', parse_dates=True)

    # 3. Predict for the most recent day (Example logic)
    latest_features = df.tail(1).drop(columns=['Children in HHS Care'])
    prediction = model.predict(latest_features)
    
    return prediction, df.tail(30) # Return prediction and last month of data
