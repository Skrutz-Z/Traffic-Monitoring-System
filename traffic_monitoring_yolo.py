import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Set the path to the YOLO configuration files
yolo_config_path = "yolov3.cfg"
yolo_weights_path = "yolov3.weights"
yolo_names_path = "coco.names"

# Load YOLO
net = cv2.dnn.readNet(yolo_weights_path, yolo_config_path)

# Load class labels
with open(yolo_names_path, "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Set up YOLO parameters (updated to avoid indexing issues)
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# Define the path to the KITTI dataset images
image_data_dir = r"KITTI/2011_09_26_drive_0117/image_02/data"

# Load and process random images from KITTI dataset for vehicle detection
image_files = sorted(os.listdir(image_data_dir))
selected_files = image_files[:10]  # Process the first 10 images for demonstration

for img_name in selected_files:
    img_path = os.path.join(image_data_dir, img_name)
    frame = cv2.imread(img_path)

    if frame is not None:
        height, width, channels = frame.shape

        # Prepare the frame for YOLO
        blob = cv2.dnn.blobFromImage(frame, scalefactor=1/255.0, size=(416, 416), swapRB=True, crop=False)
        net.setInput(blob)

        # Run the forward pass
        outs = net.forward(output_layers)

        # Analyze the output
        class_ids = []
        confidences = []
        boxes = []

        vehicle_detected = False
        non_vehicle_detected = False

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                # Debug: Print the confidence and detected class label
                print(f"Detected class: {classes[class_id]}, Confidence: {confidence}")

                # We are only interested in detecting vehicles
                if confidence > 0.2:  # Reduced confidence threshold to allow more detections
                    label = classes[class_id]
                    if label in ["car", "bus", "truck", "motorbike"]:
                        vehicle_detected = True
                    else:
                        non_vehicle_detected = True

                    # Get the bounding box coordinates
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Apply non-maximum suppression to remove redundant boxes
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.3)

        # Draw bounding boxes
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                color = (0, 255, 0)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Display and save the result
        plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        plt.title("Vehicle Detection - By Shubh Rakesh Nahar")
        plt.axis('off')
        plt.show()

        # Save the output (optional)
        output_dir = "sample_output"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"detected_{img_name}")
        cv2.imwrite(output_path, frame)

        # Print detection status
        if vehicle_detected and non_vehicle_detected:
            print(f"Image {img_name}: Non-vehicles detected & vehicles detected")
        elif vehicle_detected:
            print(f"Image {img_name}: Vehicles detected")
        elif non_vehicle_detected:
            print(f"Image {img_name}: Non-vehicles detected")
        else:
            print(f"Image {img_name}: No objects detected")
