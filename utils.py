import os
import numpy as np
import pickle
from sklearn.metrics import roc_auc_score, confusion_matrix, precision_recall_fscore_support, classification_report
from sklearn.utils.multiclass import unique_labels


def binarizey(y, c):
    ret = np.zeros_like(y)

    p = np.where(y==c)[0]

    ret[p]=1
    return ret

def savemetrics(path, predicts, groundtruth, target_names, meta):
    try:
        os.makedirs(os.path.dirname(path))
    except:
        pass

    avgcsr = np.mean(groundtruth == predicts)
    confmat = confusion_matrix(groundtruth, predicts)

    p, r, f1, _ = precision_recall_fscore_support(groundtruth, predicts,
                                                  labels= unique_labels(groundtruth, predicts),
                                                  average=None,
                                                  sample_weight=None)
    
    if len(target_names)>2:
        val_auc = np.zeros((len(target_names),))
        for i in range(len(target_names)):
            val_auc[i] = roc_auc_score(binarizey(groundtruth, i), binarizey(predicts, i))
    else:
        val_auc = roc_auc_score(groundtruth, predicts)

    with open(path + ".bin", "wb") as file:
        np.array(avgcsr).astype('float64').tofile(file)
        np.array(confmat).astype('float64').tofile(file)
        np.array(val_auc).astype('float64').tofile(file)
        np.array(r).astype('float64').tofile(file)
        np.array(p).astype('float64').tofile(file)
        np.array(f1).astype('float64').tofile(file)
        np.array(predicts).astype('int32').tofile(file)
        np.array(groundtruth).astype('int32').tofile(file)
      
    if meta is not None:
        with open(path+'.meta', "wb") as file:
            pickle.dump(meta, file)

    with open(path + '.txt', 'w') as txtfile:
        txtfile.write(str(avgcsr))
        txtfile.write('\n')
        txtfile.write(str(confmat))
        txtfile.write('\n')
        txtfile.write(str(val_auc))
        txtfile.write('\n')
        txtfile.write(classification_report(groundtruth, predicts, target_names=target_names, digits=3))
     
    return avgcsr