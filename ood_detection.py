import os
import numpy as np

root_path = "../data/OOD/"

id_sm = np.load(os.path.join(root_path, "ID_softmax.npy"))
id_pn = np.load(os.path.join(root_path, "ID_penultimate.npy"))
imagenet_sm = np.load(os.path.join(root_path, "Imagenet_softmax.npy"))
imagenet_pn = np.load(os.path.join(root_path, "Imagenet_penultimate.npy"))
train_pn = np.load(os.path.join(root_path, "train_penultimate.npy"))

def calc_auroc(binary_labels, uncertainties):
    from sklearn import metrics
    fpr, tpr, thresholds = metrics.roc_curve(binary_labels, uncertainties)
    auc = metrics.auc(fpr, tpr)
    return auc

def calc_paper_results(binary_labels, uncertainties):
    N = np.shape(binary_labels)[0]
    auroc = calc_auroc(binary_labels, uncertainties)

    order = np.argsort(uncertainties)
    uncertainties = uncertainties[order]
    binary_labels = binary_labels[order]
    print(N)

    fnrs = np.empty(N)
    fprs = np.empty(N)
    tprs = np.empty(N)
    for T in range(1, N+1):
        fp = np.sum(binary_labels[:T])
        tp = T - fp
        tn = np.sum(binary_labels[T:])
        fn = (N - T) - tn
        fnrs[T-1] = fn / (fn + tp)
        fprs[T-1] = fp / (fp + tn)
        tprs[T-1] = tp / (tp + fn)
    idx95TPR = np.argmin(np.where(tprs < 0.95, 100, tprs))
    tnr95tpr = 1.0 - fprs[idx95TPR]
    d_acc = 1.0 - np.amin(np.add(fprs, fnrs) * 0.5) # live:spoof = 50:50

    return tnr95tpr, d_acc, auroc

def softmax_ood_detector(id_sm, ood_sm):
    id_num = id_sm.shape[0]
    ood_num = ood_sm.shape[0]
    print("id_num = {}, ood_num = {}".format(id_num, ood_num))
    inv_ood_scores = np.concatenate([np.amax(id_sm, axis=1), np.amax(ood_sm, axis=1)])
    ood_scores = np.ones_like(inv_ood_scores) - inv_ood_scores
    true_labels = [0] * id_num
    true_labels.extend([1] * ood_num)
    tnr95, da, roc = calc_paper_results(np.array(true_labels), ood_scores) 
    print(ood_scores)
    print("tnr95: {}\nda: {}\nroc:{}".format(tnr95, da, roc))


if __name__=="__main__":
    softmax_ood_detector(id_sm, imagenet_sm)