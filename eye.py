from math import trunc
from transformers import AutoModelForCausalLM
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger().setLevel(logging.ERROR)

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

class VisionSystem:
    def __init__(self, model_name, model_revision, target="face"):
        # Initialize variables and parameters
        self.model_name = model_name              # Name of the detection model 
        self.model_revision = model_revision      # Version or revision of the model
        self.target = target                      # Target object to detect (e.g., "face")
        self.cap = None                           # OpenCV webcam object
        self.frame = None                         # Captured frame from webcam
        self.image = None                         # Converted frame in PIL Image format
        self.model = None                         # Loaded AI model
        self.detections = []                      # List to store detection results

    async def init_camera(self):
        # Initialize webcam
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise Exception("❌ Cannot open webcam")

    async def load_model(self):
        # Load the Moondream model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            revision=self.model_revision,
            trust_remote_code=True,
            device_map={"": "cuda"}  # Use GPU for faster inference
        )

    async def capture_frame(self):
        # Capture a single frame from the webcam
        ret, self.frame = self.cap.read()
        if not ret:
            raise Exception("❌ Failed to capture frame")
        return self.frame
        
    async def adjust_brightness(self, alpha=1.0, beta=15):
        # Adjust brightness and contrast of the captured frame
        adjusted = cv2.convertScaleAbs(self.frame, alpha=alpha, beta=beta)
        frame_rgb = cv2.cvtColor(adjusted, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
        self.image = Image.fromarray(frame_rgb)                # Convert to PIL Image for model input
        cv2.imwrite("./webcam.jpg", adjusted)


    async def caption_image(self):
        # Detect objects in the image using the model
        if self.model is None or self.image is None:
            raise Exception("Model or image not initialized.")
        return self.model.caption(self.image, length="normal")

    async def detect_face(self):
        # Detect objects in the image using the model
        if self.model is None or self.image is None:
            raise Exception("Model or image not initialized.")
        result = self.model.detect(self.image, self.target)    # Run detection on the image
        self.detections = result["objects"]                    # Save the detection results
        print(f"✅ Found {len(self.detections)} {self.target}(s)")
        result = self.model.detect(self.image, self.target)
        return len(self.detections) > 0

    async def visualize(self):
        # Visualize detection results with bounding boxes
        if not self.detections:
            print("❌ No detections to visualize.")
            return

        plt.figure(figsize=(10, 10))
        plt.imshow(self.image)
        ax = plt.gca()

        for obj in self.detections:
            # Convert normalized coordinates to pixel values
            x_min = obj["x_min"] * self.image.width
            y_min = obj["y_min"] * self.image.height
            x_max = obj["x_max"] * self.image.width
            y_max = obj["y_max"] * self.image.height
            width = x_max - x_min
            height = y_max - y_min

            print(f"📍 ({x_min:.1f}, {y_min:.1f}) to ({x_max:.1f}, {y_max:.1f})")

            # Draw red rectangle around detected object
            rect = patches.Rectangle(
                (x_min, y_min), width, height,
                linewidth=2, edgecolor='r', facecolor='none'
            )
            ax.add_patch(rect)

            # Add label text
            ax.text(
                x_min, y_min, self.target,
                color='white', fontsize=12,
                bbox=dict(facecolor='green', alpha=0.5)
            )

        plt.axis('off')                         # Hide axis
        plt.savefig("output_with_detections.jpg")  # Save the image with detections
        plt.show()                              # Display the image

    async def kill_cam(self):
        self.cap.release()              # Release the webcam
        cv2.destroyAllWindows()        # Close any OpenCV windows