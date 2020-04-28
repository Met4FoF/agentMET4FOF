from sklearn.metrics import f1_score, mean_squared_error
import pandas as pd
import numpy as np

def avg_unc(y_true,y_pred,y_unc):
    return np.mean(y_unc)

def p_acc_unc(y_data,y_pred,y_unc,threshold=0.8, task="classification"):
    aggregate = pd.DataFrame.from_dict({"y_true":y_data,"y_pred":y_pred,"y_unc":y_unc})
    aggregate_filter = aggregate[aggregate["y_unc"]>=threshold]
    if task == "classification":
        result = f1_score(aggregate_filter["y_true"],aggregate_filter["y_pred"],average="micro")
    else:
        result = mean_squared_error(aggregate_filter["y_true"],aggregate_filter["y_pred"])
    return result


