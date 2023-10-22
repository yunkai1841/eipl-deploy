from loader import model_loader
from torchinfo import summary
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for model_name in ["sarnn", "cnnrnn", "cnnrnnln", "caebn"]:
    print(model_name)
    model = model_loader(model_name, device)
    summary(model)
