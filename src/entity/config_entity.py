from src.exceptions.exceptions import CustomException
from src.logs.logging import logging
from src.constants import *

import os,sys
class TrainerConfig:
    def __init__(self):
        self.modle_name=MODEL_NAME
        self.model_path=MODEL_PATH
        self.cropped_image_path=CROPPED_IMAGE_PATH
        self.data=DATA
        self.cropped_image_name=CROPPED_IMAGE_NAME
class PlateDetectionConfig:
    def __init__(self,trainer_config:TrainerConfig):
        image_path=os.path.join(DATA,'numberplate.jpeg')
        model_file_path=os.path.join(MODEL_PATH,MODEL_NAME)
        cropped_image_path=os.path.join(CROPPED_IMAGE_PATH,CROPPED_IMAGE_NAME)
        

        