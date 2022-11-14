import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def computePR (scores, confidence_table, threshold):
    df = pd.DataFrame({'IOU': scores,
                   'Confidence': confidence_table})   
    df['IOU'] = df['IOU'].astype(float)
    df['Confidence'] = df['Confidence'].astype(float)
    df_sorted = df.sort_values(by=['Confidence'], ascending=False)
    sorted_scores = np.array(df_sorted['IOU'].to_numpy())

    TP = np.where(sorted_scores >= threshold, 1, 0)
    FP = np.where(sorted_scores < threshold, 1, 0)
    total_true_ears = 500

    very_small = 0.000000001 # to avoid division with 0
    
    TP_cumulative = np.cumsum(TP)
    FP_cumulative = np.cumsum(FP)

    precisions = np.divide(TP_cumulative, TP_cumulative+FP_cumulative+very_small)
    recalls = TP_cumulative/total_true_ears

    return precisions, recalls


def plotPRCurve(scores, confidence_table, threshold):
    
    precisions, recalls = computePR(scores, confidence_table, threshold)
    fig, ax = plt.subplots()
    ax.plot(recalls, precisions)
    ax.set_xlabel("Recall", size=14)
    ax.set_ylabel("Precision", size=14)
    ax.set_title("Precision to recall at different thresholds", size=20, pad=25)
    plt.show()
        
    
def computeAP():
    with open("yolo-scores.csv", "r") as file:
        csvreader = csv.reader(file)
        scores = np.array(next(csvreader)).astype(float)
        confidence_table = np.array(next(csvreader))

        average_precisions = []
        thresholds = np.linspace(0.0, 1.0, num=101)
        for threshold in thresholds:
            precisions, recalls = computePR(scores, confidence_table, threshold)
            average_precision = np.trapz(precisions, recalls)
            average_precisions.append(average_precision)

        fig, ax = plt.subplots()
        ax.plot(thresholds, average_precisions)
        ax.set_xlabel("Thresholds", size=14)
        ax.set_ylabel("AP", size=14)
        ax.set_title("Average precision with respect to increasing IoU threshold.", size=14, pad=25)
        plt.show()



def computeAccuracy(data, threshold):
    TP = sum(float(i) >= threshold for i in data)
    FP =  sum(float(i) < threshold and float(i) >= 0 for i in data)
    FN = sum(float(i) < 0 for i in data)
    TN = 0

    accuracy = (TP + TN)/(TP + TN + FN + FP)
    return accuracy



def computemIoU(data):
    return sum(data)/len(data)



def plotmiou():

    yolo = getYoloScore()
    vj_dict = getVJScore()

    plt.figure(figsize=(12, 8))
    plt.axhline(y=yolo, label="YOLO")
    plt.annotate("YOLO", (9, yolo))

    x_axis = np.arange(0,10)
    for key in vj_dict:
        entry = vj_dict[key]
        plt.plot(x_axis, entry, label = str(key))
        if entry[9] > 0.1:
            plt.annotate(str(key), (x_axis[9], entry[9]))

    plt.xlabel("minNeigbour", size=14)
    plt.ylabel("mIoU", size=14)
    plt.title("Average IoU", size=20, pad=25)
    plt.legend(markerscale=0.5, title="scale factor", fancybox=True)
    plt.xticks(x_axis)
    plt.show()




def plotAccuracy():

    with open("viola-jones-Scores.csv", "r") as file:
        csvreader = list(csv.reader(file))

        vj_optimal_0= np.array(csvreader[0][2:]).astype(float)
        vj_optimal_5 = np.array(csvreader[5][2:]).astype(float)
    file.close()
    with open("yolo-scores.csv", "r") as file:
        csvreader = csv.reader(file)
        yolo_scores = np.array(next(csvreader)).astype(float)
    file.close()

    thresholds = np.linspace(0.0, 1.0, num=101)
    yolo_accuracy = []
    vj0_accuracy = []
    vj5_accuracy = []
    for threshold in thresholds:
        yolo_accuracy.append(computeAccuracy(yolo_scores, threshold))
        vj5_accuracy.append(computeAccuracy(vj_optimal_5, threshold))
        vj0_accuracy.append(computeAccuracy(vj_optimal_0, threshold))


    plt.figure(figsize=(12, 8))

    plt.plot(thresholds, yolo_accuracy, label = "YOLO")
    plt.plot(thresholds, vj5_accuracy, label = "Viola-Jones minNiegborus=5")
    plt.plot(thresholds, vj0_accuracy, label = "Viola-Jones minNiegborus=0")

    plt.xlabel("Thresholds", size=14)
    plt.ylabel("Accuracy", size=14)
    plt.title("Average over all thresholds", size=20, pad=25)
    plt.legend(markerscale=0.5, fancybox=True)
    plt.show()



def getYoloScore():
    with open("yolo-scores.csv", "r") as file:
        csvreader = csv.reader(file)
        scores = np.array(next(csvreader)).astype(float)
        confidence_table = np.array(next(csvreader))

        # plotPRCurve(scores, confidence_table,  0.5)
        # computeAP(scores, confidence_table)
    file.close()

    return computemIoU(scores)




def getVJScore():
    mean_iou_dict = {}
    # accuracy_table = {}
    with open("viola-jones-Scores.csv", "r") as file:
        csvreader = csv.reader(file)
        table = []
        for data in csvreader:
            scaleFactor = float(data[0])
            minNeighbour = int(data[1])
            scores = np.array(data[2:]).astype(float)
            miou = computemIoU(scores)

            table.append(miou)

            if (minNeighbour == 9):
                mean_iou_dict[scaleFactor] = table
                table = []


    file.close()

    return mean_iou_dict



def main():
    
    # plotmiou()

    # plotAccuracy()

    computeAP()
    
main()
    