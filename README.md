# 🚗 YOLOv8 Vehicle Detection and Counting using SORT

A real-time vehicle detection, tracking, and counting system built using **YOLOv8**, **OpenCV**, and the **SORT tracking algorithm**.

The system detects vehicles from a traffic surveillance video, tracks each vehicle with a unique ID, and counts vehicles when they cross a predefined virtual counting line.

---

## Features

- Real-time vehicle detection using YOLOv8
- Multi-object tracking using SORT
- Vehicle counting using a virtual line
- Supports:
  - Car
  - Motorcycle
  - Bicycle
  - Truck
- Region of Interest (ROI) masking for improved detection
- Displays unique tracking IDs
- Live vehicle count on the video

---

## Technologies Used

- Python
- OpenCV
- Ultralytics YOLOv8
- NumPy
- SORT Tracking
- cvzone

---

## Project Structure

```
Vehicle-Counter/
│
├── main.py
├── sort.py
├── requirements.txt
├── README.md
├── mask.png
│
├── Road-Videos/
│   └── Video1.mp4
│
└── Yolo-Weights/
    └── yolov8n.pt
```

---

## Installation

Clone the repository

```bash
git clone https://github.com/yourusername/Vehicle-Counter.git
```

Move into the project directory

```bash
cd Vehicle-Counter
```

Create a virtual environment (optional)

```bash
python -m venv .venv
```

Activate it

Windows

```bash
.venv\Scripts\activate
```

Install the required packages

```bash
pip install -r requirements.txt
```

---

## Usage

Run the project

```bash
python main.py
```

Press **Q** to quit.

---

## Model

This project currently uses **YOLOv8 Nano (`yolov8n.pt`)** because it is lightweight and runs efficiently on CPU-only systems.

For better detection accuracy, especially on distant or small vehicles, it is recommended to use a GPU and switch to one of the larger models:

- `yolov8m.pt` (Recommended)
- `yolov8l.pt`
- `yolov8x.pt`

Simply replace:

```python
model = YOLO("../Yolo-Weights/yolov8n.pt")
```

with

```python
model = YOLO("../Yolo-Weights/yolov8m.pt")
```

or

```python
model = YOLO("../Yolo-Weights/yolov8l.pt")
```

---

## Notes

- This project is optimized for CPU execution using **YOLOv8 Nano**.
- Vehicle counting is performed using a virtual counting line.
- Detection quality depends on camera angle, video resolution, lighting conditions, and the selected YOLO model.
- Larger YOLO models provide higher detection accuracy but require significantly more computational resources.

---

## Future Improvements

- Multi-lane vehicle counting
- Speed estimation
- Vehicle classification statistics
- CSV/Database logging
- Web dashboard
- ByteTrack/DeepSORT integration
- Real-time webcam/IP camera support

---

## License

This project is intended for educational and learning purposes.
