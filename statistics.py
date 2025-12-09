import requests
import pandas as pd
import time
import json

URL = "http://192.168.50.2:8000/experiment"

payload = {
    "scheduler": "bpfland",
    "cpu": 1,
    "cpu_method": "matrixprod",
    "io": 0,
    "mem_load": 0,
    "vm_workers": 0,
    "duration": 5,
    "interval": 0.25
}

results = []

for i in range(50):
    print(f"Running iteration {i+1}/50 ...")

    try:
        response = requests.post(URL, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()

        results.append(data['result'])

    except Exception as e:
        print(f"Error on iteration {i+1}: {e}")
        continue

    # Slight wait between runs (optional)
    time.sleep(1)

# Convert collected results to a dataframe
df = pd.DataFrame(results)

print("\n=== DataFrame ===")
print(df)

# Compute summary statistics
summary = df.describe()

print("\n=== Summary Statistics (mean, std, etc.) ===")
print(summary)

# Save to CSV
df.to_csv("results.csv", index=False)
summary.to_csv("summary_csv", index=False)

print("\nSaved results to results.csv")
