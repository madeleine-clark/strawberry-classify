from flask import Flask, request, jsonify, render_template, send_file
from werkzeug.utils import secure_filename
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import os
from skimage import color

# Define model architecture
class StrawberryCNN(torch.nn.Module):
    def __init__(self):
        super(StrawberryCNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = torch.nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = torch.nn.Conv2d(32, 64, 3, padding=1)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.fc1 = torch.nn.Linear(64 * 18 * 18, 128)
        self.dropout1 = torch.nn.Dropout(0.5)
        self.fc2 = torch.nn.Linear(128, 64)
        self.dropout2 = torch.nn.Dropout(0.5)
        self.fc3 = torch.nn.Linear(64, 1)

    def forward(self, x):
        x = self.pool(torch.nn.functional.relu(self.conv1(x)))
        x = self.pool(torch.nn.functional.relu(self.conv2(x)))
        x = self.pool(torch.nn.functional.relu(self.conv3(x)))
        x = x.view(-1, 64 * 18 * 18)
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.dropout2(x)
        x = torch.sigmoid(self.fc3(x))
        return x

app = Flask(__name__)
UPLOAD_FOLDER = '/Users/Katie/Desktop/StrawWeb/strawberry-classify/uploads'
PROCESSED_FOLDER = '/Users/Katie/Desktop/StrawWeb/strawberry-classify/processed'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

# Load your trained model
model = StrawberryCNN()
model.load_state_dict(torch.load('/Users/Katie/Desktop/StrawWeb/strawberry-classify/strawberry_classifier.pth', map_location=torch.device('cpu')))
model.eval()

# model2 = StrawberryCNN()
# model2.load_state_dict(torch.load('/Users/Katie/Desktop/StrawWeb/strawberry-classify/strawberry_filter_regions.pth', map_location=torch.device('cpu')))
# model2.eval()

# Transformations and prediction function
def transform_and_predict(image, model):
    transform = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image_tensor)
        prediction = torch.round(output).item()
    
    label = "Unripe" if prediction == 1 else "Ripe"
    print(f"Predicted: {label}, Model Output: {output}")  # Debug statement to check model output and final label
    return label


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Process image
        processed_path = process_image(file_path)
        return send_file(processed_path, mimetype='image/jpeg')
    else:
        return jsonify({'error': 'Invalid file type'}), 400

def detect_strawberries(orig_img, img_num):
    hsv_img = color.rgb2hsv(orig_img)

    # Color range for filtering
    lower_thres = np.array([0, 0.204, 0.576])
    upper_thres = np.array([1, 1, 1])

    detected_strawberries = np.logical_and(hsv_img >= lower_thres, hsv_img <= upper_thres).all(axis=2)
    # Convert from boolean to uint8 image
    detected_strawberries = detected_strawberries.astype(np.uint8) * 255

    # Get locations of detected areas
    contours, _ = cv2.findContours(detected_strawberries, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create copy of the original image and draw bounding boxes
    img_with_boxes = orig_img.copy()

    # Filter out areas that aren't big enough to be strawberries
    min_area_thres = 3000

    bounding_boxes = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area_thres:
            x, y, w, h = cv2.boundingRect(contour)

            # Saving bounding boxes
            coordinates = [x, y, w, h]

            # Create placeholder for label after classification
            label = "None"
            bounding_boxes.append((coordinates, img_num, label))

            create_cropped_img(x, y, x+h, y+h, orig_img, img_num)
            cv2.rectangle(img_with_boxes, (x, y), (x + w, y + h), (255, 0, 0), 5)
            img_num += 1
            
    for coordinates, img_num, label in bounding_boxes:
        if coordinates is not None:
            x, y, w, h = coordinates

            # Show bounding box
            cv2.rectangle(img_with_boxes, (x, y), (x + w, y + h), (255, 0, 0), 5)

            # Create and show label
            font_scale = 1
            font_thickness = 2
            t_size = cv2.getTextSize(label, 0, fontScale=font_scale, thickness=font_thickness)[0]
            cv2.rectangle(img_with_boxes, (x, y), (x + t_size[0] + 5, y - t_size[1] - 5), (255, 0, 0), -1)
            cv2.putText(img_with_boxes, label, (x, y), cv2.FONT_HERSHEY_DUPLEX, font_scale, (0, 0 , 0), font_thickness, lineType=cv2.LINE_AA)

    return detected_strawberries, img_with_boxes, img_num, bounding_boxes

def process_image(image_path):
    img = Image.open(image_path)
    img_array = np.array(img)
    _, img_with_boxes, _, bounding_boxes = detect_strawberries(img_array, 0)

    for i, (coordinates, img_num, label) in enumerate(bounding_boxes):
        if coordinates:
            x, y, w, h = coordinates
            cropped_img = img_array[y:y+h, x:x+w]
            pil_img = Image.fromarray(cropped_img)
            new_label = transform_and_predict(pil_img, model)
            print(f"Bounding box {i}: coordinates=({x}, {y}, {w}, {h}), label before={label}, after={new_label}")  # Detailed debug info
            bounding_boxes[i] = (coordinates, img_num, new_label)

    img_with_boxes = draw_boxes_and_labels(img_array, bounding_boxes)
    processed_path = os.path.join(PROCESSED_FOLDER, os.path.basename(image_path))
    cv2.imwrite(processed_path, cv2.cvtColor(img_with_boxes, cv2.COLOR_RGB2BGR))  # Convert to BGR before saving with OpenCV
    return processed_path

def draw_boxes_and_labels(img, bounding_boxes):
    for coordinates, _, label in bounding_boxes:
        if coordinates:
            x, y, w, h = coordinates
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 5)
            cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
    return img

def create_cropped_img(x1, y1, x2, y2, orig_img, i):
    cropped_img = orig_img[y1:y2, x1:x2]
    img_bgr = cv2.cvtColor(cropped_img, cv2.COLOR_RGB2BGR)
    dir_path = "./cropped"
    img_path = "./cropped/img_" + str(i) + "_.jpg"
    if not os.path.exists(dir_path):
        mode = 0o777
        os.makedirs(dir_path, mode)
    cv2.imwrite(img_path, img_bgr)
    return

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'gif'}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002, debug=True)
