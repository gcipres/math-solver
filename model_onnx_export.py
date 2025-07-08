# export_to_onnx.py
import torch
import torch.nn as nn

# Modelo igual al del entrenamiento
class DigitClassifier(nn.Module):
    def __init__(self):
        super(DigitClassifier, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Cargar el modelo
model = DigitClassifier()
model.load_state_dict(torch.load("digit_classifier.pth", map_location=torch.device('cpu')))
model.eval()

# Exportar a ONNX
dummy_input = torch.randn(1, 1, 28, 28)
torch.onnx.export(model, dummy_input, "digit_classifier.onnx",
                  input_names=['input'], output_names=['output'],
                  dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})

print("✅ Exportado como digit_classifier.onnx")