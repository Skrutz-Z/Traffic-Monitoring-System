# Traffic Monitoring System

A real-time traffic monitoring system using YOLO (You Only Look Once) object detection to identify and track vehicles in traffic footage. The system processes images from the KITTI dataset, detects various vehicle types (cars, buses, trucks, motorbikes), and generates visual outputs with bounding boxes. Built with OpenCV and Python, it demonstrates practical computer vision applications in traffic analysis.

## Features

- Real-time vehicle detection using YOLOv3
- Support for multiple vehicle types (cars, buses, trucks, motorbikes)
- Visual output with bounding boxes and labels
- Confidence score display for each detection
- Automatic image saving with detection results

## Prerequisites

- Python 3.x
- OpenCV
- NumPy
- Matplotlib

## Required Files

- YOLOv3 configuration file (`yolov3.cfg`)
- YOLOv3 weights file (`yolov3.weights`)
- COCO class names file (`coco.names`)
- KITTI dataset images

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/traffic-monitoring-system.git
cd traffic-monitoring-system
```

2. Install required packages:
```bash
pip install opencv-python numpy matplotlib
```

3. Download YOLOv3 files:
- Download `yolov3.weights` from [YOLO website](https://pjreddie.com/darknet/yolo/)
- Download `yolov3.cfg` and `coco.names` from the YOLO GitHub repository

## Usage

1. Place the required YOLO files in the project directory
2. Update the `image_data_dir` path in `traffic_monitoring_yolo.py` to point to your KITTI dataset
3. Run the script:
```bash
python traffic_monitoring_yolo.py
```

## Output

The system will:
- Process images from the specified directory
- Display detection results in real-time
- Save processed images in the `sample_output` directory
- Print detection status for each processed image

## Author

Shubh Rakesh Nahar

## License

This project is licensed under the MIT License - see the LICENSE file for details. 