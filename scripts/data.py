import requests
import pandas as pd
from dotenv import dotenv_values



def get_params(series_id, frequency, start_date, end_date):
    """Get parameters for API request."""
    config = dotenv_values("scripts/.env")
    api_key =  config["FRED_API_KEY"]
    params = {
        "api_key": api_key,
        "series_id": series_id,
        "observation_start": start_date,
        "observation_end": end_date,
        "file_type": "json",
        "frequency": frequency
    }
    return params

def get_data(series_id, feature_name, base_url, frequency, start_date, end_date):
    """Fetch data from the API, process it, and save to CSV."""
    response = requests.get(base_url, params=get_params(series_id, frequency, start_date, end_date))
    
    if response.status_code == 200:
        data = response.json()
        # Convert to DataFrame
        observations = data.get("observations", [])
        df = pd.DataFrame(observations)
        
        if not df.empty:
            # Process DataFrame
            df["value"] = df["value"].astype(float).round(4)
            df = df[["date", "value"]]
            df.rename(columns={"value": feature_name}, inplace=True)
            return df
        else:
            print(f"No data available for {series_id}")
            
    else:
        print(f"Error fetching {series_id}: {response.status_code}, {response.text}")

def fetch_and_save_data(feature_dict, base_url, start_date, end_date):
    combined_df = pd.DataFrame()

    for frequency, features in feature_dict.items():
        for series_id, feature_name in features.items():
            df = get_data(series_id, feature_name, base_url, frequency, start_date, end_date)
            
            if not df.empty:
                if combined_df.empty:
                    combined_df = df
                else:
                    combined_df = combined_df.merge(df, on="date", how="outer")

    combined_df["Year"] = pd.DatetimeIndex(combined_df["date"]).year
    combined_df["Month"] = pd.DatetimeIndex(combined_df["date"]).month

    # Filling missing values in the Per_Capita_GDP column using linear interpolation
    combined_df["Per_Capita_GDP"] = combined_df["Per_Capita_GDP"].interpolate()
    combined_df.ffill(inplace=True)
    combined_df.to_csv('dataset.csv', index=False)
    
    return combined_df

