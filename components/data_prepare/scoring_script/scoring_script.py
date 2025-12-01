import os
import json
import torch

global model

def init():
    """
    Loads the PyTorch model artifact registered via MLflow.
    """
    global model
    
    model_root = os.getenv('AZUREML_MODEL_DIR')
    model_path = os.path.join(model_root, 'data/model.pth') 
    
    try:
        model = torch.load(model_path, map_location='cpu')
        model.eval() 
        print("PyTorch Model loaded and set to evaluation mode.")

    except Exception as e:
        print(f"Error loading PyTorch model: {e}")

def run(raw_data):
    """
    Accepts raw JSON data, converts it to PyTorch tensor, and returns a prediction.
    """
    try:
        data_list = json.loads(raw_data)['data'] 
        input_tensor = torch.tensor(data_list, dtype=torch.float32)

        with torch.no_grad():
            output = model(input_tensor)
            predictions = output.tolist() 
            
        return json.dumps({"predictions": predictions})

    except Exception as e:
        error = str(e)
        return json.dumps({"error": error})