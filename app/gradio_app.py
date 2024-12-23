import gradio as gr
import torch
from PIL import Image
import torchvision.transforms as transforms
from src.model.resnet import ResNet50

# Load model
model = ResNet50()
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

# Preprocessing
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

def predict(image):
    img = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(img)
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    return {idx: float(prob) for idx, prob in enumerate(probabilities)}

iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=5),
    examples=[["example1.jpg"], ["example2.jpg"]]
)

iface.launch() 