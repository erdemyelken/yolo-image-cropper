import os
from ultralytics import YOLO
from PIL import Image

class ImageCropper:
    def init(self, model_path, input_folder, output_folder):
        self.model_path = model_path
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.prepare_model()
    
    def prepare_model(self):
        try:
            self.model = YOLO(self.model_path)
        except FileNotFoundError:
            print(f"Error: Model file not found at {self.model_path}")
            raise
        except Exception as e:
            print(f"Error: An error occurred while loading the model: {e}")
            raise

    def detect_objects(self):
        if not os.path.exists(self.input_folder):
            print(f"Error: Input folder not found at {self.input_folder}")
            return
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        for filename in os.listdir(self.input_folder):
            if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.jpeg'):
                image_path = os.path.join(self.input_folder, filename)
                try:
                    with Image.open(image_path) as img:
                        print(f'Processing {filename}')
                        predictions = self.model.predict(img)
                        if len(predictions) == 0:
                            print(f"Warning: No objects detected in {image_path}. Skipping image.")
                            continue
                        print(predictions[0].boxes.cls.tolist())
                        boxes = predictions[0].boxes.xyxy.tolist()
                        if len(boxes) > 0:
                            x1, y1, x2, y2 = boxes[0]
                            image = img.crop((x1, y1, x2, y2))
                            image.save(os.path.join(self.output_folder, filename))
                            print(f"Successfully processed {image_path} and saved to {self.output_folder}")
                        else:
                            print(f"Warning: No bounding box found in {image_path}. Skipping image.")
                except Exception as e:
                    print(f"Error: An error occurred while processing {image_path}: {e}")

    def crop_and_save(self, image, predictions, filename):
        if predictions[0].boxes.shape[0]>0:
            x1, y1, x2, y2 = predictions[0].boxes.xyxy[0].tolist()
            image = image.crop((x1, y1, x2, y2))
            image.save(os.path.join(self.output_folder, filename))
        else:
            print(f"Warning: No bounding box found in {filename}. Skipping image.")

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
