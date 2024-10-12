# Plantro-X: Tomato Leaf Disease Detection

This project implements a tomato leaf disease detection system using the YOLOv5 model. The application utilizes a pre-trained YOLOv5 model to detect diseases in tomato leaves by processing images captured from real-time camera.

## Features
- **Object Detection:** Utilizes the YOLOv5 model to detect various diseases in tomato leaves.
- **Training and Testing:** The model is trained on a custom dataset and can be tested using validation images.
- **Model Conversion:** Converts the YOLOv5 model to TFLite format for deployment on Android devices.

## Prerequisites
- Python 3.x
- The following Python libraries must be installed:
  - `torch`
  - `opencv-python`
  - `IProgress`
  - `tqdm`
  
You can install the required libraries using pip:
```bash
pip install torch opencv-python IProgress tqdm
```
## Project Structure
```bash
.
├── yolov5/                # Cloned YOLOv5 repository containing the model and training scripts
│   ├── detect.py          # Script for running inference with the trained model
│   ├── train.py           # Script for training the YOLOv5 model
│   ├── export.py          # Script for exporting the model to different formats
│   └── ...                # Other YOLOv5 files and dependencies
├── Output Images/         # Folder to save output images with detection results
├── Plantro-X.ipynb       # Main Jupyter Notebook for training and testing the model
└── README.md              # Project README file
```
## How It Works
Clone YOLOv5 Repository: The YOLOv5 repository is cloned from GitHub, which contains the necessary scripts for training and inference.
Dataset Preparation: The dataset's dataset.yaml file is moved to the YOLOv5 data directory, which contains the information required to train the model on custom data.
Model Training: The YOLOv5 model is trained using the custom dataset.
Model Evaluation: The model is tested on validation images to evaluate its performance.
Model Conversion: The trained model is converted to TFLite format for deployment on Android devices.

## Running the Project
Open the Jupyter Notebook:
`jupyter notebook Plantro-X.ipynb`

Follow the instructions in the notebook to mount your Google Drive and clone the YOLOv5 repository.

Prepare your dataset and ensure the dataset.yaml file is correctly set up.

Run the training script to train the YOLOv5 model on your dataset.

Test the model using validation images to see the detection results.

Convert the trained model to TFLite format for Android deployment.

## Example Output
After running the training and testing scripts, the results will be saved in the Output Images folder, showing the detected diseases in the tomato leaves.

## Contributing
Contributions, issues, and feature requests are welcome! Feel free to check the issues page for new ideas.

## License
This project is licensed under the MIT License.

## Acknowledgments
YOLOv5 for object detection functionalities.
PyTorch for deep learning capabilities.
Google Colab for providing a cloud-based Jupyter notebook environment.
