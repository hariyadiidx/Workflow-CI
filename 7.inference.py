import requests
import time
import random

print("🤖 Robot Inference mulai mengirim data ke model...")

while True:
    data = {
        "dataframe_split": {
            "columns": ["Age", "Income", "Loan_Amount"],
            "data": [[random.randint(20, 60), random.randint(3000, 100000), random.randint(1000, 50000)]]
        }
    }
    
    try:
        response = requests.post("http://127.0.0.1:5001/invocations", json=data)
        print(f"Status: {response.status_code} | Hasil: {response.text.strip()}")
    except Exception as e:
        print("Menunggu server model siap...")
        
    time.sleep(3)