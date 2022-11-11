import csv
import matplotlib.pyplot as plt
import numpy as np

def plotScores(data):
    precision_table = []
    recall_table = []
    for threshold in np.arange(0.0, 1.0, 0.01):
        TP = sum(float(i) >= threshold for i in data)
        FP =  sum(float(i) < threshold and float(i) >= 0 for i in data)
        FN = sum(float(i) < 0 for i in data)
        
        print(f"\n\nAt threshold: {threshold}")
        print(f"tp = {TP},  fp = {FP},  fn = {FN}")
        
        precision = TP/(TP+FP)
        recall = TP/(TP+FN)
        
        print(f"Precision = {precision},   Recall = {recall}")
        
        precision_table.append(precision)
        recall_table.append(recall)
            
        
    fig, ax = plt.subplots()
    ax.plot(recall_table, precision_table)
    ax.set_xlabel("Recall", size=14)
    ax.set_ylabel("Precision", size=14)
    ax.set_title("Precision to recall at different thresholds", size=20, pad=25)
    plt.show()
        
    
def main():
    with open("yolo-scores.csv", "r") as file:
        csvreader = csv.reader(file)
        data = next(csvreader)
        plotScores(data)
    file.close()
    
    # with open("viola-jones-scores.csv", "r") as file:
    #     csvreader = csv.reader(file)
    #     data = next(csvreader)
    #     plotScores(data)
    # file.close()
    
main()
    