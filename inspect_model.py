import torch

# Load the model
checkpoint = torch.load('stock_prediction_model.pth', map_location=torch.device('cpu'))

# Print the structure
print("Keys in checkpoint:", checkpoint.keys() if isinstance(checkpoint, dict) else "Model is not a dictionary")
print("\nModel structure:")
for key, value in checkpoint.items() if isinstance(checkpoint, dict) else []:
    if isinstance(value, dict):
        print(f"{key}:")
        for k, v in value.items():
            print(f"  {k}: {type(v)}")
    else:
        print(f"{key}: {type(value)}")
