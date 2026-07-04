import torch
import cv2
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
from torchvision import models
import torch.nn as nn

class ResNetEncoder(nn.Module):
    def __init__(self, feature_dim=64):
        super().__init__()
        resnet = models.resnet18(weights=None)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(resnet.fc.in_features, feature_dim)
        self.bn = nn.BatchNorm1d(feature_dim)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.bn(self.fc(x))

class Classifier(nn.Module):
    def __init__(self, feature_dim=64, num_classes=7):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(32, num_classes),
        )
    def forward(self, x):
        return self.net(x)

device = torch.device("cpu")
E_exp = ResNetEncoder(64).to(device)
C = Classifier(64, 7).to(device)

print("Loading local .pth files...")
E_exp.load_state_dict(torch.load('expression_encoder.pth', map_location=device))
C.load_state_dict(torch.load('fer_classifier.pth', map_location=device))
E_exp.eval()
C.eval()

inference_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

emotion_classes = ["Neutral", "Angry", "Happy", "Disgust", "Sad", "Surprise", "Fear"]
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

image_path = "test_image.jpg"
img_bgr = cv2.imread(image_path)

if img_bgr is not None:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face_rgb = cv2.cvtColor(img_bgr[y:y+h, x:x+w], cv2.COLOR_BGR2RGB)
        input_tensor = inference_transform(face_rgb).unsqueeze(0).to(device)

        with torch.no_grad():
            features = E_exp(input_tensor)
            logits = C(features)
            probabilities = F.softmax(logits, dim=1).squeeze().numpy() * 100

        pred_idx = np.argmax(probabilities)

        cv2.rectangle(img_bgr, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(img_bgr, f"{emotion_classes[pred_idx]}: {probabilities[pred_idx]:.1f}%",
                    (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("DICE-FER Local Inference", img_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print(f"Could not find {image_path}. Make sure the image is in the folder!")
