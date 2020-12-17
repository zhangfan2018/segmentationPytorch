
import matplotlib.pyplot as plt

thres = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
         20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34]

tp_remove = [0, 4, 5, 6, 7, 9, 11, 14, 16, 21, 23, 27, 28, 32, 38, 42,
             48, 50, 50, 51, 53, 54, 55, 55, 55, 55, 55, 55, 55, 56, 56]

fp_remove = [0, 105, 125, 151, 177, 197, 220, 238, 255, 274, 292, 305,
             318, 328, 335, 343, 355, 367, 379, 385, 389, 394, 396, 397, 397, 398, 398, 398, 398, 398, 398]

all_tp = 654
all_fp = 890
all_pred_tps = 671

all_recall = []
all_precision = []
for i in range(len(tp_remove)):
    recall = (all_tp - tp_remove[i]) / 792
    precision = (all_tp - tp_remove[i]) / (all_pred_tps + all_fp - fp_remove[i])
    all_recall.append(recall)
    all_precision.append(precision)

recall_thres_16mm = [all_recall[12] for i in range(len(all_recall))]
recall_thres_10mm = [all_recall[6] for i in range(len(all_recall))]
recall_thres_5mm = [all_recall[1] for i in range(len(all_recall))]

best_point = [all_recall[6], all_precision[6]]
print(all_recall[12], all_precision[12])


plt.figure()
plt.plot(all_recall, all_precision)
plt.plot(recall_thres_16mm, all_precision, '--g', label="threshold=16mm")
plt.plot(recall_thres_10mm, all_precision, '-.g', label="threshold=10mm")
plt.plot(recall_thres_5mm, all_precision, ':g', label="threshold=5mm")
plt.plot(best_point[0], best_point[1], "^r", label="best threshold")
plt.legend()
plt.title("The RP curve of fracture detection in test set")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.grid()
plt.show()