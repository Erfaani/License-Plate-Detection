# Iranian License Plate Detection from Vehicle Images

## Overview
This project focuses on detecting Iranian license plates from vehicle images using the YOLO (You Only Look Once) object detection model. The goal is to build a model that can accurately identify license plates in images of vehicles, enabling automation for various applications, including traffic monitoring and vehicle registration systems.

## Features
- **YOLO-based Object Detection:** The model is built using YOLO, a state-of-the-art object detection algorithm, to identify and locate Iranian license plates in vehicle images.
- **High Accuracy:** The model has been trained and tested on a custom dataset of vehicle images, achieving high precision and recall in detecting license plates.
- **Real-time Inference:** The trained model can detect license plates with real-time processing, providing quick and efficient results.

## Requirements
- Python 3.x
- PyTorch
- Ultralytics YOLO library
- OpenCV
- Matplotlib
- Other dependencies (see `requirements.txt`)

You can download the dataset directly from this link:

👉 **[Download the Dataset Here](https://www.kaggle.com/datasets/nimapourmoradi/car-plate-detection-yolov8)**
## Installation

1. Clone the repository:
   
   git clone https://github.com/Erfaani/iranian-license-plate-detection.git
   
2. Install the required dependencies:
   
   pip install -r requirements.txt
   
3. Download the trained model or train your own:
   - To use a pre-trained model, simply load the model using YOLO and follow the usage instructions below.
   - To train your own model, follow the instructions in the training section.


## Training (Optional)
To train the model on your own dataset:
1. Prepare a dataset with images of vehicles and corresponding labels in YOLO format.
2. Modify the configuration files to suit your dataset.
3. Train the model using the following command:
   ```bash
   python train.py --data data.yaml --cfg yolov5s.yaml --weights '' --batch-size 16 --epochs 50
   ```

## Usage

1. Load the pre-trained model:
   ```python
   from ultralytics import YOLO
   model = YOLO("notebooks/runs/detect/train/weights/best.pt")
   ```

2. Run predictions on an image:
   ```python
   results = model.predict("notebooks/runs/detect/train/predict")
   print(results.pandas().xywh)
   ```

3. The model will output the bounding box coordinates of the detected license plate in the image.

4. The detected images with annotated bounding boxes will be saved in the specified directory.


## Results
The model performs efficiently on a variety of vehicle images. Example outputs include the coordinates of the detected license plates, which can be used for further analysis or integration with vehicle registration systems.

## License
This project is licensed under the MIT License

## Contributing
Feel free to open issues, fork the project, and submit pull requests. Contributions are always welcome!

## Contact & Feedback

I would love to hear your thoughts and suggestions about this project. Feel free to reach out to me for any questions or feedback:

**Email:** [Erfanjouybar@gmail.com]  
**GitHub:** [[Github Link](https://github.com/Erfaani/)]  
**LinkedIn:** [[Linkedin link](https://www.linkedin.com/in/erfanjouybar)]  