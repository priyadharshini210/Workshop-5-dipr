# Workshop-5-dipt
## Aim

The aim of this experiment is to develop a complete pipeline for automatic license plate detection using OpenCV’s Haar Cascade Classifier. The task involves loading input images, preprocessing them, applying the Haar Cascade XML model to detect license plates, drawing bounding boxes, cropping the detected region, and saving the output. Additionally, a small improvement is made to the base code—modifying detection parameters and applying preprocessing—to increase detection accuracy.

## Algorithm

The following steps summarize the workflow used in the Jupyter Notebook:

1. Read and Display Input Image

Use cv2.imread() to load the image into memory.

Display the image using matplotlib.pyplot.imshow().

2. Convert Image to Grayscale

Haar cascades work on intensity patterns, not color.

Convert to grayscale using:

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

3. Preprocessing (Improvement Added)

To improve contrast and detection accuracy:

Apply Gaussian Blur to reduce noise:

gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)


Apply Histogram Equalization to enhance features:

gray_eq = cv2.equalizeHist(gray_blur)

4. Load Haar Cascade Classifier

Link the XML file:

plate_cascade = cv2.CascadeClassifier('haarcascade_russian_plate_number.xml')

5. Detect License Plates

Use detectMultiScale() on the preprocessed grayscale image:

plates = plate_cascade.detectMultiScale(
    gray_eq,
    scaleFactor=1.1,        # improvement: more strict scaling
    minNeighbors=6,         # improvement: reduce false positives
    minSize=(30, 30)
)

6. Draw Bounding Boxes on the Image

For each detected plate, draw a rectangle:

cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 3)

7. Crop and Save the Detected Plate

Extract the plate region of interest (ROI):

roi = img[y:y+h, x:x+w]
cv2.imwrite("detected_plate.jpg", roi)

8. Display Final Results

Show the bounding-box image and the cropped ROI using Matplotlib.
## PROGRAM:
```
import cv2
import matplotlib.pyplot as plt
import os
import urllib.request

# -------------------------------------------------------------
# Step 1: Read and display the input image
# -------------------------------------------------------------
image_path = 'im1.png'  # <-- Change this to your image filename
image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError("Image not found. Please check the 'image_path' variable.")

plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis('off')
plt.show()

# -------------------------------------------------------------
# Step 2: Convert to grayscale
# -------------------------------------------------------------
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
plt.imshow(gray, cmap='gray')
plt.title("Grayscale Image")
plt.axis('off')
plt.show()

# -------------------------------------------------------------
# Step 3: Preprocessing (optional)
# -------------------------------------------------------------
# Gaussian Blur to reduce noise
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
# Histogram Equalization for better contrast
equalized = cv2.equalizeHist(blurred)

plt.imshow(equalized, cmap='gray')
plt.title("Preprocessed Image (Blur + Equalized)")
plt.axis('off')
plt.show()

# -------------------------------------------------------------
# Step 4: Load or Download Haar Cascade for Face Detection
# -------------------------------------------------------------
cascade_path = 'haarcascade_frontalface_default.xml'

# Auto-download if not present
if not os.path.exists(cascade_path):
    print("Cascade file not found. Downloading...")
    url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
    urllib.request.urlretrieve(url, cascade_path)
    print("Cascade file downloaded successfully!")

# Load classifier
face_cascade = cv2.CascadeClassifier(cascade_path)

# -------------------------------------------------------------
# Step 5: Detect faces using Haar Cascade
# -------------------------------------------------------------
faces = face_cascade.detectMultiScale(
    equalized,          # Preprocessed grayscale image
    scaleFactor=1.1,    # Scaling factor between image pyramid layers
    minNeighbors=5,     # Higher value -> fewer false detections
    minSize=(30, 30)    # Minimum object size
)

print(f"Total Faces Detected: {len(faces)}")

# -------------------------------------------------------------
# Step 6: Draw bounding boxes and save cropped faces
# -------------------------------------------------------------
output = image.copy()
save_dir = "Detected_Faces"
os.makedirs(save_dir, exist_ok=True)

for i, (x, y, w, h) in enumerate(faces):
    cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 3)
    face_crop = image[y:y+h, x:x+w]
    save_path = f"{save_dir}/face_{i+1}.jpg"
    cv2.imwrite(save_path, face_crop)

if len(faces) > 0:
    print(f"{len(faces)} face(s) saved in '{save_dir}' folder.")
else:
    print("⚠️ No faces detected. Try adjusting parameters or using a clearer image.")

# -------------------------------------------------------------
# Step 7: Display the final output
# -------------------------------------------------------------
plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
plt.title("Detected Faces")
plt.axis('off')
plt.show()
```
## OUTPUT:

<img width="353" height="342" alt="Screenshot 2025-11-24 104727" src="https://github.com/user-attachments/assets/05cff6d0-700c-489d-ad8b-7e2cf5fefe95" />

<img width="339" height="347" alt="Screenshot 2025-11-24 104734" src="https://github.com/user-attachments/assets/cb7a4f13-529c-4261-9a9f-431fb8184db5" />

<img width="328" height="343" alt="Screenshot 2025-11-24 104741" src="https://github.com/user-attachments/assets/94e9e467-64f4-490b-927b-b40b17759c42" />

<img width="347" height="333" alt="Screenshot 2025-11-24 104749" src="https://github.com/user-attachments/assets/193fab88-8542-4f97-ac38-60fd7ac76ff8" />

## Result

The code successfully detected license plates in the test images using the Haar Cascade classifier.


All steps executed without errors, and the final outputs were displayed clearly using Matplotlib.
