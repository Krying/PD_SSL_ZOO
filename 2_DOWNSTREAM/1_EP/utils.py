from re import L
import torch
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc, RocCurveDisplay, roc_auc_score

def AUROC(args, acc_targets, y_pred_prob, epoch, mode):
    fpr, tpr, th = roc_curve(acc_targets, y_pred_prob) ### ROC curve를 그리기 위한 fpr, tpr, th 얻기
    youden = np.argmax(tpr-fpr)
    idx_05 = np.abs(th - 0.5).argmin()
    plt.figure(figsize=(5,5))
    plt.plot(fpr, tpr, color="navy", lw=2, label="ROC ET/PD (area = %0.4f)" % auc(fpr, tpr))
    plt.plot([0, 1], [0, 1], color="gray", lw=2, linestyle="--")
    plt.scatter(fpr[youden], tpr[youden], color='red', label="Youden index") # Youden Index point
    plt.scatter(fpr[idx_05], tpr[idx_05], color='blue', label="Threshold 0.5") # Threshold=0.5 point

    plt.title('ROC curve', fontsize=12)
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.01])
    plt.xlabel("False Positive Rate", fontsize=14)
    plt.ylabel("True Positive Rate", fontsize=14)
    plt.legend(loc="lower right", fontsize=12)
    plt.savefig(args.log_dir_png + f'{mode}_{epoch}_auroc.png')
    auroc = auc(fpr, tpr) #PD AUROC
    plt.close()

    return auroc 

from itertools import cycle
import matplotlib.pyplot as plt
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import average_precision_score, precision_recall_curve

def AUPRC(args, acc_targets, y_pred_prob, epoch, mode):
    precision, recall, _ = precision_recall_curve(acc_targets, y_pred_prob)
    average_precision = average_precision_score(acc_targets, y_pred_prob, average="micro")
    _, ax = plt.subplots(figsize=(5, 5))

    display = PrecisionRecallDisplay(recall=recall, precision=precision, average_precision=average_precision)
    display.plot(ax=ax, name=f"PRC ET/PD", color="navy")

    handles, labels = display.ax_.get_legend_handles_labels()
    ax.legend(handles=handles, labels=labels, loc="best", fontsize=12)
    ax.set_title("PRC curve", fontsize=14)

    plt.xlabel("Recall", fontsize=14)
    plt.ylabel("Precision", fontsize=14)
    plt.savefig(args.log_dir_png + f'{mode}_{epoch}_auprc.png')
    plt.close()
    
    return average_precision

def save_checkpoint(save_file_path, model):
    save_dict = {
        'state_dict': model.state_dict(),
    }
    torch.save(save_dict, save_file_path)
