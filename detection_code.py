
import cv2
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torch.nn.functional as F

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Face detection model using MTCNN
mtcnn = MTCNN(keep_all=True, device=device)

# Define the FaceNet model with a classifier
class FaceNetWithClassifier(torch.nn.Module):
    def __init__(self, base_model, num_classes):
        super(FaceNetWithClassifier, self).__init__()
        self.base_model = base_model
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(512, num_classes),
            torch.nn.LogSoftmax(dim=1)
        )
    
    def forward(self, x):
        embeddings = self.base_model(x)  # Generate embeddings from the base model
        return self.classifier(embeddings)  # Pass through the classifier


num_classes = 12

# Load the FaceNet model and classifier
facenet_base = InceptionResnetV1(pretrained='vggface2', classify=False).eval()
model = FaceNetWithClassifier(facenet_base, num_classes).to(device)
model.load_state_dict(torch.load('facenet_finetuned.pth', map_location=device))
  # Load trained model weights
model.eval()

# Transformation for input images
transform = transforms.Compose([    
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # Normalize pixel values to [-1, 1]
])

# Class labels (replace with actual class names from your dataset)
class_labels = ["ali_taheri", "amirhossein_gholizadeh", "amirhossin_kheiri", "aria_dadnavi","else" ,"kian_khatibi", "mahdi_azimi", "maria_sabouri", "reyhane_mohasseli", "saina_najafi", "sobhan_hosseini", "zahra_maham"]

# Function to detect and recognize faces in a frame with a confidence filter
def detect_and_recognize_frame(frame, confidence_threshold=0.95):
    # Convert OpenCV frame (BGR) to PIL image (RGB)
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    # Detect faces and get bounding boxes with their confidence scores
    boxes, probs = mtcnn.detect(img)
    
    if boxes is not None:
        draw = ImageDraw.Draw(img)
        try:
            # Load a larger font
            font = ImageFont.truetype("arial.ttf", 30)  # You can adjust the size here
        except IOError:
            font = ImageFont.load_default()  # Fallback to default font if Arial is not available

        for box, prob in zip(boxes, probs):
            if prob >= confidence_threshold:  # Apply the confidence filter
                # Draw bounding box
                draw.rectangle(box.tolist(), outline='red', width=3)
                
                # Crop the face and preprocess it
                face = img.crop((box[0], box[1], box[2], box[3]))
                face_tensor = transform(face).unsqueeze(0).to(device)
                
                # Predict the class
                with torch.no_grad():
                    outputs = model(face_tensor)
                    probabilities = F.softmax(outputs, dim=1)
                    confidence, predicted = torch.max(probabilities, 1)
                    confidence_score = confidence.item()
                    if(confidence_score>=confidence_threshold):
                        label = class_labels[predicted.item()]  
                    else:
                        label = "else"
                        confidence_score = -1
                
                # Prepare the label text and confidence text
                confidence_text = f"Confidence: {confidence_score*100:.2f}%"
                label_text = f"{label}"
                
                # Annotate the image with the predicted label and confidence score
                draw.text((box[0], box[1] - 10), label_text, font=font, fill='yellow')
                draw.text((box[0], box[1] - 40), confidence_text, font=font, fill='yellow')

    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)  # Convert back to OpenCV format

# Start webcam feed
cap = cv2.VideoCapture(1)

print("Press 'q' to exit the webcam feed.")
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Process the frame for face detection and recognition
    frame_with_boxes = detect_and_recognize_frame(cv2.flip(frame,1))
    
    # Display the frame
    cv2.imshow('Face Recognition',frame_with_boxes)
    
    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
