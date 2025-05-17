from sklearn.metrics import roc_curve, roc_auc_score
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import numpy as np

def calculate_eer(y_true, y_score):
    # Check if there are enough samples and both classes are present
    if len(y_true) < 2 or len(np.unique(y_true)) < 2:
        print("Warning: Not enough samples or only one class present. Returning EER=0.5")
        return 0.5

    try:
        fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=1)
        eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
        return eer
    except Exception as e:
        print(f"Error calculating EER: {e}")
        return 0.5

def calculate_auc(y_true, y_score):
    # Check if there are enough samples and both classes are present
    if len(y_true) < 2 or len(np.unique(y_true)) < 2:
        print("Warning: Not enough samples or only one class present. Returning AUC=0.5")
        return 0.5

    try:
        return roc_auc_score(y_true, y_score)
    except Exception as e:
        print(f"Error calculating AUC: {e}")
        return 0.5
