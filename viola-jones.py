import cv2 as cv
from tqdm import tqdm
import glob
import csv
from Helpers import *
import numpy as np



def detect(image, rightEarsCascade, leftEarsCascade, scaleFactor = 1.1, minNeighbors = 3, show=False):
    image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    image_gray = cv.equalizeHist(image_gray)
    
    # Detect right ears
    rightEars = rightEarsCascade.detectMultiScale(image_gray, scaleFactor=scaleFactor, minNeighbors=minNeighbors)
    # Detect left ears
    leftEars = leftEarsCascade.detectMultiScale(image_gray)
    
    detected = []
    
    for (x,y,w,h) in rightEars:
        box = calculate_coordinates(x,y,w,h)
        detected.append(box)
        
    for (x,y,w,h) in leftEars:
        box = calculate_coordinates(x,y,w,h)
        detected.append(box)
        
    if show:
        for (x,y,w,h) in rightEars:
            topLeft = (x, y)
            bottomRight = (x+w, y+h)
            image = cv.rectangle(image, topLeft, bottomRight, (255, 0, 255), 4)
        
        for (x,y,w,h) in leftEars:
            topLeft = (x, y)
            bottomRight = (x+w, y+h)
            image = cv.rectangle(image, topLeft, bottomRight, (255, 0, 255), 4)
    
        imS = image_resize(image, 720)  
        cv.imshow('Image - Ears detection', imS)
        cv.waitKey(0)
        
    return detected
        
        
        

def viola_jones_all(rightEarsCascade, leftEarsCascade, scaleFactor = 1.1, minNeighbors = 3, write_csv=True):
    scores = []
    for image_path in tqdm( sorted(glob.glob("Support Files/ear_data/test/*.png")) , desc="Reading images... "):
        image = cv.imread(image_path)
        boxes = detect(image, rightEarsCascade, leftEarsCascade, scaleFactor=scaleFactor, minNeighbors=minNeighbors)
        imgHeight, imgWidth = image.shape[:2]
        
        with open(image_path[0:-4]+".txt") as f:
            xCenterGT, yCenterGT, widthGT, heightGT = [float(x) for x in next(f).split()][1:]
            xminGT, yminGT, xmaxGT, ymaxGT = calculate_pixel_coordinates(xCenterGT, yCenterGT, heightGT, widthGT, imgHeight, imgWidth)
        
            # I mark cases where no ear was detected (although there is one ear on every photo) with a
            # negative value, so I can count False Negatives when calculating Recall
            iou_score = 0
            if write_csv:
                iou_score = -1
            if len(boxes) > 0:
                iou_score = 0
                for box in boxes:
                    iou_score = calculate_iou([xminGT, yminGT, xmaxGT, ymaxGT], box)
                    scores.append(iou_score)
                
            else:
                scores.append(iou_score)
            
            
        f.close()
        
    if write_csv:
        with open('viola-jones-Scores.csv', 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(scores)
        f.close()
    else:
        return scores
    
    

def viola_jones_single(image_path, rightEarsCascade, leftEarsCascade):
    image = cv.imread(image_path)
    detect(image, rightEarsCascade, leftEarsCascade, show=True)




def test_parameteres(rightEarsCascade, leftEarsCascade):

    scaleFactors = [1.02, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 3.0]

    with open("test_parameters_0_neigbours_result.txt", 'w') as f:
        f.write(f"Testing parameters of the Viola-Jones alghorithm:\n")

        maximum = -3
        maxParams = [0.0, 0]

        for scaleFactor in scaleFactors:
            for minNeighbours in [0]:
                print(f"computing for ScaleFactor = {scaleFactor} and minNeighbours = {minNeighbours}")
                scores = viola_jones_all(rightEarsCascade, leftEarsCascade, scaleFactor=scaleFactor, minNeighbors=minNeighbours, write_csv=False)
                average = sum(scores) / len(scores)

                if average > maximum:
                    maximum = average
                    maxParams[0] = scaleFactor
                    maxParams[1] = minNeighbours

                f.write(f"  ScaleFactor = {scaleFactor},  minNeighbours = {minNeighbours}\n")
                f.write(f"      mIoU: {average}\n")

        f.write(f"\n\nBest parameters and average score:\n")
        f.write(f"ScaleFactor = {maxParams[0]},  minNeighbours = {maxParams[1]} with mIoU: {maximum}\n")



def save_scores_parameters(rightEarsCascade, leftEarsCascade):
    with open('viola-jones-Scores.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)

        scaleFactors = [1.02, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.5, 1.75, 2.0, 3.0]

        for scaleFactor in scaleFactors:
            for minNeighbours in range(0, 10, 1):
                print(f"computing for ScaleFactor = {scaleFactor} and minNeighbours = {minNeighbours}")
                scores = viola_jones_all(rightEarsCascade, leftEarsCascade, scaleFactor=scaleFactor, minNeighbors=minNeighbours, write_csv=False)
                
                row = [scaleFactor, minNeighbours] + scores

                writer.writerow(row)
    f.close()

def main():
    rightEarsCascade = cv.CascadeClassifier()
    leftEarsCascade = cv.CascadeClassifier()

    if not rightEarsCascade.load("Support Files/haarcascade_mcs_rightear.xml"):
        print('--(!)Error loading right ear cascade')
        exit(0)
    if not leftEarsCascade.load("Support Files/haarcascade_mcs_leftear.xml"):
        print('--(!)Error loading left ear cascade')
        exit(0)
        
    # viola_jones_single("Support Files/ear_data/test/0501.png", rightEarsCascade, leftEarsCascade)
    # viola_jones_all(rightEarsCascade, leftEarsCascade, scaleFactor = 1.05, minNeighbors = 0)
    # test_parameteres(rightEarsCascade, leftEarsCascade)
    save_scores_parameters(rightEarsCascade, leftEarsCascade)
    
main()

 



