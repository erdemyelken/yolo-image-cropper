# YOLO Image Cropper

This Python script uses the YOLO object detection model to crop objects from images in a specified input folder and save them in a specified output folder.

## Installation

This script requires the following packages:

- `ultralytics`
- `Pillow`

You can install these packages via pip:

     pip install -r requirements.txt

## Usage

To use the script, you need to provide the path to the YOLO model file, the path to the input folder containing the images, and the path to the output folder where the cropped images will be saved.

```python
if __name__ == "__main__":  
    model_path = '../models/yolov8x.pt'
    input_folder = '../inputdata'
    output_folder = '../outputdata'

    cropper = ImageCropper()
    cropper.init(model_path, input_folder, output_folder)

    try:
        cropper.prepare_model()
        cropper.detect_objects()
    except Exception as e:
        print(f"Error: An error occurred while running the program: {e}")
        exit(1)

    print("Completed successfully.")

The init method initializes the class with the model path, input folder, and output folder.

The prepare_model method loads the YOLO model from the specified file path.

The detect_objects method loops through each image in the input folder, performs object detection using the YOLO model, and crops the first detected object from the image. The cropped image is then saved to the output folder.

