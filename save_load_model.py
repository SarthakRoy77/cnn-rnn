import torch
from gru import model

load_model_bool = True

def save_model(path, models):
    print("Saving Model")
    torch.save(models.state_dict(), path)
    print("Model Saved")

def load_model(checkpoint):
    print("Loading Model")
    model.load_state_dict(checkpoint['state_dict'])
    print("Model Loaded")

save_model("test_model.pth.tar", models=model)
check_point = {"state_dict" : model.state_dict()}
if load_model_bool:
    load_model(check_point)