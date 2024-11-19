import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, max_error, median_absolute_error

def get_RMSE(Y_TRUE, Y_PRED):
    list_diff_square = [(x - y)**2 for x, y in zip(Y_PRED, Y_TRUE)]
    mse = np.mean(list_diff_square)
    rmse = np.sqrt(mse)
    return rmse

def get_MAE(Y_TRUE, Y_PRED):
    list_mae = [(x - y) for x, y in zip(Y_PRED, Y_TRUE)]
    mae = np.mean(np.abs(list_mae))
    return mae

def get_max_error(Y_TRUE, Y_PRED):
    max_er = max_error(Y_PRED , Y_TRUE)
    return max_er

def get_uncentered_R_squared(Y_TRUE, Y_PRED):
    list_square = [(x)**2 for x in Y_PRED]
    SS_tot_uncentered = np.sum(list_square)
    list_diff_square = [(x - y)**2 for x, y in zip(Y_PRED, Y_TRUE)]
    SS_res = np.sum(list_diff_square)

    R_squared_uncentered = 1- SS_res/SS_tot_uncentered
    return R_squared_uncentered

def get_CCC(Y_TRUE, Y_PRED):
    x_bar = np.mean(Y_TRUE)
    y_bar = np.mean(Y_PRED)

    list_var_x = [(x - x_bar)**2 for x in Y_TRUE]
    list_var_y = [(y - y_bar)**2 for y in Y_PRED]

    var_x = (np.sum(list_var_x)) / len(Y_TRUE)
    var_y = (np.sum(list_var_y)) / len(Y_PRED)

    list_covar = [(x - x_bar)*(y - y_bar) for x, y in zip(Y_TRUE, Y_PRED)]
    covar = np.sum(list_covar) / len(Y_TRUE)

    rho_c= (2 * covar) / (var_x + var_y + (x_bar - y_bar)**2)
    
    return rho_c

def get_evaluation_metrics(args, epoch, Y_TRUE, Y_PRED):
    rmse = get_RMSE(Y_TRUE, Y_PRED)
    mae_value = get_MAE(Y_TRUE, Y_PRED)
    abs_max_error_value = get_max_error(Y_TRUE, Y_PRED)
    median_abs_error_value = median_absolute_error(Y_TRUE, Y_PRED)
    
    r2_scipy_value = r2_score(Y_TRUE, Y_PRED)
    r2_inverse = r2_score( Y_PRED, Y_TRUE)
    resid = [(x - y) for x, y in zip(Y_PRED, Y_TRUE)]
    r2_uncentered = get_uncentered_R_squared(Y_TRUE, Y_PRED)
    Lin_concor = get_CCC(Y_TRUE, Y_PRED)
    
    print("RMSE: ", rmse)
    print("MAE: ", mae_value)
    print("R2_scipy: ", r2_scipy_value)
    print("R2_scipy_inverse:", r2_inverse)
    print("R2_uncentered: ", r2_uncentered)
    print("Lin Concordance correlation coefficient: ", Lin_concor)

    max_resid = np.max(np.abs(resid))
    mean_resid = np.mean(np.abs(resid))
    std_resid = np.std(resid)
    median_resid = np.median(np.abs(resid))
    
    print("\nmax resid: ", max_resid)
    print("mean resid: ", mean_resid)
    print("std resid: ", std_resid)
    print("median resid: ", median_resid)

    metric_font_style = {"fontsize": 18, "transform": "ax.transAxes"}
    bbox = dict(boxstyle="round,pad=0.5,rounding_size=0.2", fc="white", ec ="lightgray")

    _x, _y = Y_TRUE,Y_PRED

    fig, ax = plt.subplots(nrows = 1, ncols = 1,figsize = (10, 10))
    label_text = f"CCC  : {Lin_concor:,.4f}\nMAE  : {mae_value:,.4f}\nRMSE: {rmse:,.4f}\nR2: {r2_scipy_value:,.4f}"
    ax.annotate(label_text, (0.05, 0.82), xycoords='axes fraction', **metric_font_style, bbox = bbox)
    
    ax.scatter(_x, _y, marker = "o", alpha = 0.2, s = 20)
    ax.plot([0, 25], [0, 25], color='gray', linestyle='--')
    ax.tick_params(axis = "both",size = 10, labelsize = 18)
    ax.set_xlabel("Ground truth onset year", fontsize = 20)
    ax.set_ylabel("Predicted onset year", fontsize = 20)
    plt.savefig(args.log_dir_png + str(epoch), dpi = 300, bbox_inches = "tight")
    plt.close()

    return {"rmse" : round(rmse, 4), 
            "mae_value" : round(mae_value, 4), 
            "abs_max_error_value" : round(abs_max_error_value, 4), 
            "median_abs_error_value" : round(median_abs_error_value, 4), 
            "r2_scipy_value" : round(r2_scipy_value, 4), 
            "r2_inverse" : round(r2_inverse, 4), 
            "max_resid" : round(max_resid, 4), 
            "mean_resid" : round(mean_resid, 4), 
            "std_resid" : round(std_resid, 4), 
            "median_resid" : round(median_resid, 4), 
            "r2_uncentered" : round(r2_uncentered, 4), 
            "Lin_concor" : round(Lin_concor, 4)}
    
def save_checkpoint(save_file_path, model):
    save_dict = {
        'state_dict': model.state_dict(),
    }
    torch.save(save_dict, save_file_path)
    