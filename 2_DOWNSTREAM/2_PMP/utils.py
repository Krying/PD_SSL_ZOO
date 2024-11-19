import torch
import numpy as np
from itertools import cycle
import matplotlib.pyplot as plt
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import average_precision_score, precision_recall_curve

def AUROC(args, acc_targets_0, acc_targets_1, acc_targets_2, y_pred_prob_0, y_pred_prob_1, y_pred_prob_2, Y_test, y_score, epoch, mode):
    acc_targets = [acc_targets_0, acc_targets_1, acc_targets_2]
    y_pred_list = [y_pred_prob_0, y_pred_prob_1, y_pred_prob_2]
    auroc = []
    fpr_for_macro, tpr_for_macro, roc_auc_for_macro = dict(), dict(), dict()

    fpr, tpr, th = roc_curve(acc_targets[0], y_pred_list[0]) ### ROC curve를 그리기 위한 fpr, tpr, th 얻기
    youden = np.argmax(tpr-fpr)
    idx_05 = np.abs(th - 0.5).argmin()
    plt.figure(figsize=(8,8))
    plt.style.use("seaborn-v0_8")
    plt.plot(fpr, tpr, color="navy", lw=2, label="ROC PD (area = %0.4f)" % auc(fpr, tpr))
    plt.plot([0, 1], [0, 1], color="gray", lw=2, linestyle="--")
    plt.scatter(fpr[youden], tpr[youden], color='red') # Youden Index point
    plt.scatter(fpr[idx_05], tpr[idx_05], color='blue') # Threshold=0.5 point
    auroc.append(auc(fpr, tpr)) #PD AUROC
    fpr_for_macro[0] = fpr
    tpr_for_macro[0] = tpr

    fpr, tpr, th = roc_curve(acc_targets[1], y_pred_list[1]) ### ROC curve를 그리기 위한 fpr, tpr, th 얻기
    youden = np.argmax(tpr-fpr)
    idx_05 = np.abs(th - 0.5).argmin()
    plt.plot(fpr, tpr, color="turquoise", lw=2, label="ROC MSA (area = %0.4f)" % auc(fpr, tpr))
    plt.scatter(fpr[youden], tpr[youden], color='red') # Youden Index point
    plt.scatter(fpr[idx_05], tpr[idx_05], color='blue') # Threshold=0.5 point
    auroc.append(auc(fpr, tpr)) #MSA AUROC
    fpr_for_macro[1] = fpr
    tpr_for_macro[1] = tpr

    fpr, tpr, th = roc_curve(acc_targets[2], y_pred_list[2]) ### ROC curve를 그리기 위한 fpr, tpr, th 얻기
    youden = np.argmax(tpr-fpr)
    idx_05 = np.abs(th - 0.5).argmin()
    plt.plot(fpr, tpr, color="darkorange", lw=2, label="ROC PSP (area = %0.4f)" % auc(fpr, tpr))
    plt.plot([0, 1], [0, 1], color="gray", lw=2, linestyle="--")
    plt.scatter(fpr[youden], tpr[youden], color='red', label="Youden index") # Youden Index point
    plt.scatter(fpr[idx_05], tpr[idx_05], color='blue', label="Threshold 0.5") # Threshold=0.5 point
    auroc.append(auc(fpr, tpr)) #PSP AUROC
    fpr_for_macro[2] = fpr
    tpr_for_macro[2] = tpr

    for i in range(3):
        roc_auc_for_macro[i] = auc(fpr_for_macro[i], tpr_for_macro[i])

    fpr_grid = np.linspace(0.0, 1.0, 1000)

    # Interpolate all ROC curves at these points
    mean_tpr = np.zeros_like(fpr_grid)

    for i in range(args.num_class):
        mean_tpr += np.interp(fpr_grid, fpr_for_macro[i], tpr_for_macro[i])  # linear interpolation

    # Average it and compute AUC
    mean_tpr /= args.num_class

    fpr_for_macro["macro"] = fpr_grid
    tpr_for_macro["macro"] = mean_tpr
    roc_auc_for_macro["macro"] = auc(fpr_for_macro["macro"], tpr_for_macro["macro"])
    plt.grid(True)

    plt.title('ROC curve', fontsize=12)
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.01])
    plt.xlabel("False Positive Rate", fontsize=14)
    plt.ylabel("True Positive Rate", fontsize=14)
    # print(f"Y_test : {Y_test}")
    # print(f"Y_score : {y_score}")

    print("##############################################################################################################################")
    #Avg - macro
    macro_roc_auc_ovr = roc_auc_score(Y_test, y_score, multi_class="ovr", average="macro")
    print(f"Macro-averaged One-vs-Rest ROC AUC score: {macro_roc_auc_ovr:.4f}")
    auroc.append(macro_roc_auc_ovr)
    
    #Avg - micro
    micro_roc_auc_ovr = roc_auc_score(Y_test, y_score, multi_class="ovr", average="micro")
    print(f"Micro-averaged One-vs-Rest ROC AUC score: {micro_roc_auc_ovr:.4f}")

    fpr, tpr, _ = roc_curve(Y_test.ravel(), y_score.ravel())
    roc_auc = auc(fpr, tpr)
    print(f"Micro-averaged One-vs-Rest ROC AUC score: {roc_auc:.4f}")
    auroc.append(micro_roc_auc_ovr) #AVG -AUC

    plt.plot(fpr, tpr, color="purple", label="ROC AVG (micro) (area = %0.4f)" % roc_auc,  linestyle=":", linewidth=4)
    plt.plot(fpr_for_macro["macro"], tpr_for_macro["macro"], color="crimson", label="ROC AVG (macro) (area = %0.4f)" % roc_auc_for_macro['macro'], linestyle=":", linewidth=4)

    plt.legend(loc="lower right", fontsize=12)
    plt.savefig(args.log_dir_png + f'{mode}_{epoch}_auroc.png')
    plt.close()

    return auroc #list [PD_AUROC, MSA_AUROC, PSP_AUROC, AVG_AUROC_macro, AVG_AUROC_micro]

def make_unique(input_list):
    seen = set()
    output_list = []

    for value in input_list:
        while value in seen:
            value += 1e-7
        seen.add(value)
        output_list.append(value)

    return output_list

def AUPRC(args, Y_test, y_score, epoch, mode):
    # For each class
    precision, recall, average_precision = dict(), dict(), dict()
    precision_for_macro, recall_for_macro, average_precision_for_macro = dict(), dict(), dict()

    auprc = []
    num_class = int(args.num_class)
    #class-wise Precisino-Recall and average_precision_score
    for i in range(num_class):
        
        if len(y_score[:, i]) != len(list(set(y_score[:, i]))):
            y_score[:, i] = make_unique(y_score[:, i])
            
        precision[i], recall[i], _ = precision_recall_curve(Y_test[:, i], y_score[:, i])

        average_precision[i] = average_precision_score(Y_test[:, i], y_score[:, i])
        auprc.append(average_precision[i])

        recall_for_macro[i] = recall[i]
        precision_for_macro[i] = precision[i]
        average_precision_for_macro[i] = average_precision[i]
        
    #overall micro-precision and micro-recall + micro average_precision_score
    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], _ = precision_recall_curve(
        Y_test.ravel(), y_score.ravel()
    )
    average_precision["micro"] = average_precision_score(Y_test, y_score, average="micro")
    average_precision["macro"] = average_precision_score(Y_test, y_score, average="macro")
    
    auprc.append(average_precision["macro"]) #auprc[3]
    auprc.append(average_precision["micro"]) #auprc[4]
        
    # setup plot details
    colors = cycle(["navy", "turquoise", "darkorange"])

    _, ax = plt.subplots(figsize=(8, 8))
    plt.style.use("seaborn-v0_8")

    class_list = ['PD', 'MSA', 'PSP']
    #Each class Precision-Recall
    for i, color in zip(range(num_class), colors):
        display = PrecisionRecallDisplay(
            recall=recall[i],
            precision=precision[i],
            average_precision=average_precision[i],
        )
        class_item = class_list[i]
        display.plot(ax=ax, name=f"PRC {class_item}", color=color)
        
    #Macro Average Precision-Recall
    display = PrecisionRecallDisplay(
        recall=(recall_for_macro[0]+recall_for_macro[1]+recall_for_macro[2])/3,
        precision=(precision_for_macro[0]+precision_for_macro[1]+precision_for_macro[2])/3,
        average_precision=(average_precision_for_macro[0]+average_precision_for_macro[1]+average_precision_for_macro[2])/3,
    )
    display.plot(ax=ax, name="PRC AVG (macro)", color="crimson", linestyle=":", linewidth=3)

    #Micro Average Precision-Recall
    display = PrecisionRecallDisplay(
        recall=recall["micro"],
        precision=precision["micro"],
        average_precision=average_precision["micro"],
    )
    display.plot(ax=ax, name="PRC AVG (micro)", color="purple", linestyle=":", linewidth=3)

    # add the legend for the iso-f1 curves
    handles, labels = display.ax_.get_legend_handles_labels()
    # set the legend and the axes
    plt.grid(True)
    ax.legend(handles=handles, labels=labels, loc="best", fontsize=12)
    ax.set_title("PRC curve", fontsize=14)
    plt.xlabel("Recall", fontsize=14)
    plt.ylabel("Precision", fontsize=14)
    plt.savefig(args.log_dir_png + f'{mode}_{epoch}_auprc.png')
    plt.close()
    
    return auprc #list [PD_AUROC, MSA_AUROC, PSP_AUROC, AVG_AUROC_macro, AVG_AUROC_micro]

def save_checkpoint(save_file_path, model):
    save_dict = {
        'state_dict': model.state_dict(),
    }
    torch.save(save_dict, save_file_path)
