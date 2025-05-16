from src.exceptions.exceptions import CustomException
from src.entity.config_entity import PlateDetectionConfig
from src.entity.artifact_entity import PlateDetectionArtifact
from src.logs.logging import logging
import os,sys
import cv2
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from ultralytics import YOLO
from paddleocr import paddleocr
import time

class PlateDetection:
    def __init__(self):
        self.plate_detection_config=PlateDetectionConfig()
       
    def covertBGRtoRGB(self):
        try:
            img=cv2.imread(self.plate_detection_config.)
            img_rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            return img_rgb
        except Exception as e:
            raise CustomException(e,sys)

    def detect_plate(self,img_rgb):
        try:
            model=YOLO(self.plate_detection_config.model_file_path)
            results=model(img_rgb)
            logging.info(f"model prediction on the image:{results}")
            for result in results:
                boxes = result.boxes  # Access boxes for each result
                logging.info(f"box coordinates on the image:{boxes}")
            # Check if any boxes were detected
            logging.info(f"Number of detected boxes: {len(results[0].boxes)}")
            if len(results[0].boxes) == 0:
                logging.info("No license plates detected.")
            return results
        except Exception as e:
            raise CustomException(e,sys)
    
    def extract_box_coord(self,results,img_rgb):
        try:

            # Extract bounding box coordinates (xywh format), confidence, and class ids
            xywh = results[0].boxes.xywh  # [x_center, y_center, width, height]
            conf = results[0].boxes.conf  # Confidence scores
            cls = results[0].boxes.cls    # Class IDs (for license plates)

            # Since we have only one box, we'll print the details of the first box
            x_center, y_center, width, height = xywh[0]
            logging.info(f"Bounding box coordinates (center_x, center_y, width, height): {x_center}, {y_center}, {width}, {height}")

            # Convert to x1, y1, x2, y2 for cropping (Top-left, Bottom-right corners)
            x1 = int(x_center - width / 2)
            y1 = int(y_center - height / 2)
            x2 = int(x_center + width / 2)
            y2 = int(y_center + height / 2)

            logging.info(f"Cropping coordinates: ({x1}, {y1}), ({x2}, {y2})")
            img_height, img_width, _ = img_rgb.shape
            logging.info(f"Image dimensions: {img_width}x{img_height}")
            # Ensure the coordinates are within the bounds of the image
            img_height, img_width, _ = img_rgb.shape
            if x1 < 0: x1 = 0
            if y1 < 0: y1 = 0
            if x2 > img_width: x2 = img_width
            if y2 > img_height: y2 = img_height

            logging.info(f"Adjusted coordinates: ({x1}, {y1}), ({x2}, {y2})")

            # Crop the license plate using the adjusted bounding box
            cropped_plate = img_rgb[y1:y2, x1:x2]

            # Check if the cropped image has valid dimensions
            logging.info(f"Cropped image dimensions: {cropped_plate.shape}")
            os.makedirs(self.plate_detection_config.cropped_image_path, exist_ok=True) 
            cv2.imwrite(self.plate_detection_config.cropped_image_path
                        , cropped_plate)
            logging.info(f"Cropped image saved at: {self.plate_detection_config.cropped_image_path}")
            return PlateDetectionArtifact(cropped_image_path=self.plate_detection_config.cropped_image_path )
        except Exception as e:  
            raise CustomException(e,sys)

if __name__=="__main__":
    plate_detection=PlateDetection()
    img_rgb=plate_detection.covertBGRtoRGB()
    results=plate_detection.detect_plate(img_rgb)
    plate_detection.extract_box_coord(results,img_rgb)  
        




        
        
        


                
        
    

        
