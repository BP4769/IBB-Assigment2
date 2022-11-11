import cv2 as cv
from tqdm import tqdm
import glob
import csv
from Helpers import *



def detect(image, rightEarsCascade, leftEarsCascade, show=False):
    image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    image_gray = cv.equalizeHist(image_gray)
    
    # Detect right ears
    rightEars = rightEarsCascade.detectMultiScale(image_gray)
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
        
        
        

def viola_jones_all(rightEarsCascade, leftEarsCascade):
    scores = []
    for image_path in tqdm( sorted(glob.glob("OneDrive_1_06-11-2022/ear_data/test/*.png")) , desc="Reading images... "):
        image = cv.imread(image_path)
        boxes = detect(image, rightEarsCascade, leftEarsCascade)
        imgHeight, imgWidth = image.shape[:2]
        
        with open(image_path[0:-4]+".txt") as f:
            xCenterGT, yCenterGT, widthGT, heightGT = [float(x) for x in next(f).split()][1:]
            xminGT, yminGT, xmaxGT, ymaxGT = calculate_pixel_coordinates(xCenterGT, yCenterGT, heightGT, widthGT, imgHeight, imgWidth)
        
            # I mark cases where no ear was detected (although there is one ear on every photo) with a
            # negative value, so I can count False Negatives when calculating Recall
            iou_score = -1
            if len(boxes) > 0:
                iou_score = 0
                for box in boxes:
                    iou_score += calculate_iou([xminGT, yminGT, xmaxGT, ymaxGT], box)
                    scores.append(iou_score)
                
            else:
                scores.append(iou_score)
            
            
        f.close()
        
        
    with open('viola-jones-Scores.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(scores)
    f.close()
    
    

def viola_jones_single(image_path, rightEarsCascade, leftEarsCascade):
    image = cv.imread(image_path)
    detect(image, rightEarsCascade, leftEarsCascade, show=True)




def main():
    rightEarsCascade = cv.CascadeClassifier()
    leftEarsCascade = cv.CascadeClassifier()

    if not rightEarsCascade.load("OneDrive_1_06-11-2022/haarcascade_mcs_rightear.xml"):
        print('--(!)Error loading right ear cascade')
        exit(0)
    if not leftEarsCascade.load("OneDrive_1_06-11-2022/haarcascade_mcs_leftear.xml"):
        print('--(!)Error loading left ear cascade')
        exit(0)
        
    # viola_jones_single("OneDrive_1_06-11-2022/ear_data/test/0501.png", rightEarsCascade, leftEarsCascade)
    viola_jones_all(rightEarsCascade, leftEarsCascade)
    
main()

 



