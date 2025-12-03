import os
import json
import mlflow.pyfunc
import pandas as pd

model = None

def init():
    global model
    model_path = os.getenv("AZUREML_MODEL_DIR")
    model = mlflow.pyfunc.load_model(model_path)

def run(raw_data):
    try:
        payload = json.loads(raw_data)
        columns = payload["input_data"]["columns"]
        data = payload["input_data"]["data"]
        
        df = pd.DataFrame(data=data, columns=columns)
        preds = model.predict(df)
        
        return {"predictions": list(map(str, preds))}
        
    except Exception as e:
        print(f"❌ Error in prediction: {str(e)}")
        return {"error": str(e)}
