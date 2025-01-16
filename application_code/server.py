import pandas as pd
from glassflow import PipelineDataSource
import time
from datetime import datetime
import json
import numpy as np


def replace_nan(value):
    if isinstance(value, float) and np.isnan(value):
        return None
    return value

# Read the CSV file
df = pd.read_csv('ais.csv')

pipeline_id = "401cb58b-612e-478b-90c0-5f67f5ee485c"
pipeline_access_token = "4tfyFrqkrpn97Mww5r3PhAEynemEHPMatAEbBACZaGRSDM7WKTmdeVT7gb5ATw4hfMQAfyP97P4HHXwFbKRaA5pvJxpqcmKNFjBx5JPHkzKyqKYYgW88wFhZFK9X4EhB"

source = PipelineDataSource(
    pipeline_id=pipeline_id,
    pipeline_access_token=pipeline_access_token
)

while True:
    for _, row in df.iterrows():
        ais_data = row.to_dict()
     
        ais_data = {k: replace_nan(v) for k, v in ais_data.items()}
        ais_data["Timestamp"] = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        
        try:
            source.publish(ais_data)
            print(f"Published Data: {ais_data}")
        except Exception as e:
            print(f"Error publishing data: {e}")
        
        time.sleep(5)  # Wait for 5 seconds before sending the next data