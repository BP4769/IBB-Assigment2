import torch
import cv2
from tqdm import tqdm
import glob
import csv
from Helpers import *

# Model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='OneDrive_1_06-11-2022\yolo5s.pt')  # local model


# Image
def yolo_all():
    scores = []
    for image_path in tqdm( sorted(glob.glob("OneDrive_1_06-11-2022/ear_data/test/*.png")) , desc="Reading images... "):
        image = cv2.imread(image_path)[..., ::-1]
        imgHeight, imgWidth = image.shape[:2]
        
        with open(image_path[0:-4]+".txt") as f:
            xCenterGT, yCenterGT, widthGT, heightGT = [float(x) for x in next(f).split()][1:]
            boxGT = calculate_pixel_coordinates(xCenterGT, yCenterGT, heightGT, widthGT, imgHeight, imgWidth)
            
            # Inference
            results = model(image)
            
            # I mark cases where no ear was detected (although there is one ear on every photo) with a
            # negative value, so I can count False Negatives when calculating Recall
            iou_score = -1
            if len(results.xyxy[0]) == 0:
                scores.append(iou_score)
            else:
                iou_score = 0
                # There is only one ear in each image but in the case the alghorithm would detect more ears I
                # would iterate through all detections and compute their scores.
                for result in results.xyxy[0]:
                    xmin, ymin, xmax, ymax = result[0:4]
                
                    iou_score = calculate_iou(boxGT, [xmin, ymin, xmax, ymax])
                    scores.append(iou_score)
            
        f.close()
        
    with open('yolo-Scores.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(scores)
    f.close()


            
            
def yolo_single(image_path):
    # img_number = image_path[-8:-4]
    image = cv2.imread(image_path)[..., ::-1]
    imgHeight, imgWidth = image.shape[:2]
    
    with open(image_path[0:-4]+".txt") as f:
        xCenterGT, yCenterGT, widthGT, heightGT = [float(x) for x in next(f).split()][1:]
        boxGT = calculate_pixel_coordinates(xCenterGT, yCenterGT, heightGT, widthGT, imgHeight, imgWidth)
        
        # Inference
        results = model(image)
        results.show()
        if len(results.xyxy[0]) == 0:
            iou_score = -1
            print(f"No ear detected")
        else:
            iou_score = 0
            # There is only one ear in each image but in the case the alghorithm would detect more ears I
            # would iterate through all detections and compute their scores.
            for result in results.xyxy[0]:
                xmin, ymin, xmax, ymax = result[0:4]
            
                iou_score = calculate_iou(boxGT, [xmin, ymin, xmax, ymax])
                print(f"Score of detection: {iou_score}")
        
        
    
def main():
    # yolo_single("OneDrive_1_06-11-2022/ear_data/test/0502.png")
    yolo_all()
    
main()
    
