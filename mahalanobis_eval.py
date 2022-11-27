import glob
import os
import sys
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
from albumentations import (CLAHE, Compose, HorizontalFlip, HueSaturationValue,
                            RandomBrightness, RandomContrast, RandomGamma,
                            ShiftScaleRotate, ToFloat, VerticalFlip)
from keras.models import load_model
import keras.backend as K
from keras import Model

from hyper_utils import hyper2rgb_batch, hyper2truecolor
from eval_utils import calc_score, calc_score_variance, calc_anomaly_scores, calc_ece, calc_mce, calc_auroc, calc_auprc
from v_models import build_class_model, build_seg_model, build_kernel_googlenet, build_pixel_class_model, build_pixel_mlp_class_model
from utils import imshow, mask2rgb, print_prob_rank, reset_tf
from v_VegetableSequence import VegetableDataset, VegetableSequence, C2HDataset
from temporal_random_seed import TemporalRandomSeed
import tqdm

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

FAKETYPE = "display" #'halo'
def evaluation(model_fn, isRGB,
               isGoogLeNet, isC2H, enableDropoutSampling=False):
    fakeAugs = 1 if isC2H else 8
    trueAugs = 1 if isC2H else 4
    includeFake = True
    numMCSamples = 10
    ds_info = C2HDataset(isTrain=False) if isC2H else VegetableDataset(isTrain=False)
    cats = ds_info.get_categories()
    if isC2H:
        fakefn = "m.rf.rgb.png" if isRGB else "m.rf.npy"
        # banana, apple, pakchoi, and petal
        fakedirs = list(range(1,121))
        fakefns = [os.path.join(ds_info.path(), "fakes", str(dn).zfill(3), fakefn) for dn in fakedirs]
    else:
        fakefn = "m.rf.rgb.png" if isRGB else "m.rf.npy"
        fakedir = '03/' if FAKETYPE == 'halo' else '05/' if FAKETYPE == 'lamp' else '06/' #06 display
        fakefns = [ds_info.DATASET_ROOT_PATH+str(i).zfill(2)+"/05/"+fakedir+fakefn for i in cats]
    is_segRGB = False

    max_value = 255
    color_limit = 0.001
    if isRGB:
        input_shape = (224, 224, 3)
        dim = 3
    else:
        dim = ds_info.hsi_dims
        input_shape = (dim, 224, 224) if isGoogLeNet else (224, 224, dim)
        if not isC2H:
            max_value = 1024
    if isC2H:
        #shift_limit = 0
        #scale_limit = 0
        shift_limit = 0.3
        scale_limit = 0.9
    else:
        shift_limit = 0.3
        scale_limit = 0.9

    AUGMENTATIONS_ALL = Compose([
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.2),
        RandomContrast(limit=color_limit, p=0.5),
        RandomBrightness(limit=color_limit, p=0.5),
        ShiftScaleRotate(
            shift_limit=shift_limit, scale_limit=scale_limit,
            rotate_limit=30, border_mode=cv2.BORDER_REFLECT_101, p=0.8),
        ToFloat(max_value=max_value)
    ])
    AUGMENTATIONS_SIMPLE = Compose([
        ToFloat(max_value=max_value)
    ])
    nb_classes = ds_info.ObjectCategories

    if isGoogLeNet:
        model = build_kernel_googlenet(nb_classes=nb_classes,
                              input_shape=input_shape,
                              dropoutSampling=enableDropoutSampling)
    else:
        model = build_class_model(nb_classes=nb_classes,
                                  weights=None, input_shape=input_shape,
                                  dropoutSampling=enableDropoutSampling)
    model.load_weights(model_fn)
    
    # mean and covariance from training set
    #for i in tqdm.trange(trueAugs, desc="true augumentations"): # 8
    penultimate_feat_extractor = Model(inputs=model.input,
                outputs=model.get_layer("penultimate").output)
    train_instances = list(range(1,101)) if isC2H else [1,2,3,4]
    train_samples = [-1] if isC2H else [1,2]
    train_raw_gen = VegetableSequence(dataset=ds_info, instance_ids=train_instances, sample_ids=train_samples, random_state=2021,
                                     augmentationRepeat=1, isSeg=False, isGoogLeNet=isGoogLeNet,
                                     batch_size=32, augmentations=AUGMENTATIONS_SIMPLE, isRGB=isRGB)
    feats = []
    train_labels = []
    for batch in tqdm.tqdm(train_raw_gen, desc="forward train"):
        xs, ys = batch
        feat = penultimate_feat_extractor.predict(
                        xs, batch_size=224*224*dim).reshape(-1, 32)
        if isGoogLeNet:
            ys = ys[0]

        assert len(ys) == len(feat)
        feats.extend(feat)
        train_labels.extend(np.argmax(ys, axis=1))

    # order
    #train_labels = np.array(train_labels)
    #feats = np.array(feats)
    #order = np.argsort(train_labels)
    #train_labels = train_labels[order]
    #feats = feats[order,:]
    #print(train_labels)
    #print(feats[:,0])
    
    class_means = np.zeros((nb_classes, 32))
    class_count = np.zeros(nb_classes)
    for i, label in enumerate(train_labels):
        class_means[label] += feats[i]
        class_count[label] += 1.0
    means = np.array([class_means[idx] / class_count[idx] for idx in range(nb_classes)])
    
    class_vars = np.zeros((nb_classes, 32, 32))
    for i, label in enumerate(train_labels):
        var = (feats[i] - means[label])[:,np.newaxis]
        cov = np.dot(var, np.transpose(var))
        class_vars[label] += cov
    covar = np.sum(class_vars, axis=0) / float(len(train_labels))

    predict_func = model.predict

    if isC2H:
        exp_instances = list(range(1, 31))
        exp_samples = [-1]
    else:
        exp_instances = [5]
        exp_samples = [1,2]

    test_aug_gen = VegetableSequence(dataset=ds_info, instance_ids=exp_instances, sample_ids=exp_samples, random_state=2021,
                                     augmentationRepeat=1, isSeg=False, isGoogLeNet=isGoogLeNet,
                                     batch_size=16, augmentations=AUGMENTATIONS_ALL, isRGB=isRGB)
    test_raw_gen = VegetableSequence(dataset=ds_info, instance_ids=exp_instances, sample_ids=exp_samples, random_state=2021,
                                     augmentationRepeat=1, isSeg=False, isGoogLeNet=isGoogLeNet,
                                     batch_size=16, augmentations=AUGMENTATIONS_SIMPLE, isRGB=isRGB)
    
    pred_f = []
    pred_y = []
    true_labels = []
    index = 0
    for i in tqdm.trange(trueAugs, desc="true augumentations"): # 8
        for batch in tqdm.tqdm(test_aug_gen, desc="true augumentations batch"): #nan batch
            xs, ys = batch
            pred_ys = predict_func(xs)
            pred_fs = penultimate_feat_extractor.predict(
                            xs, batch_size=224*224*dim).reshape(-1, 32)
            if isGoogLeNet:
                pred_ys = pred_ys[2]
                ys = ys[0]

            assert len(ys) == len(pred_ys)
            pred_y.extend(pred_ys)
            pred_f.extend(pred_fs)
            true_labels.extend(np.argmax(ys, axis=1))

    #for batch in tqdm.tqdm(test_raw_gen, desc="true raw batches"):
    #    xs, ys = batch
    #    pred_ys = predict_func(xs)
    #    pred_fs = penultimate_feat_extractor.predict(
    #                    xs, batch_size=224*224*dim).reshape(-1, 32)
    #    if isGoogLeNet:
    #        pred_ys = pred_ys[2]
    #        ys = ys[0]

    #    assert len(ys) == len(pred_ys)
    #    pred_y.extend(pred_ys)
    #    pred_f.extend(pred_fs)
    #    true_labels.extend(np.argmax(ys, axis=1))

    if includeFake:
        with TemporalRandomSeed(2019):
            for fn in tqdm.tqdm(fakefns, desc="fakes"):
                #cache_fn = fn[:-4] + ".224_224.npy"
                cache_fn = os.path.join(os.path.dirname(fn), "224_224." + os.path.basename(fn))
                if os.path.exists(cache_fn):
                    #x = np.load(cache_fn).astype(
                    #    "uint8" if isRGB else "uint16")
                    x = cv2.imread(cache_fn) if isRGB else np.load(cache_fn).astype("uint16")
                else:
                    if isRGB:
                        #x = np.load(fn).astype("uint8")
                        x = cv2.imread(fn).astype("uint8")
                    else:
                        x = np.load(fn).astype("uint16")
                    x = cv2.resize(x, (224, 224))
                    np.save(cache_fn, x)
                #xs_aug = np.array([AUGMENTATIONS_SIMPLE(image=x)["image"]
                if RESULT_CACHE_ROOT.find("replay") != -1:
                    x *= 2
                xs_aug = np.array([AUGMENTATIONS_ALL(image=x)["image"]
                          for i in range(fakeAugs)])
                if isGoogLeNet:
                    xs_aug = xs_aug.transpose((0, 3, 1, 2)) 
                pred_ys = predict_func(xs_aug)
                pred_fs = penultimate_feat_extractor.predict(
                                xs_aug, batch_size=224*224*dim).reshape(-1, 32)
                if isGoogLeNet:
                    pred_ys = pred_ys[2]
                assert fakeAugs == len(pred_ys)
                pred_y.extend(pred_ys)
                pred_f.extend(pred_fs)
                true_labels.extend([200] * fakeAugs)  # for fake

    pred_y = np.array(pred_y)
    pred_f = np.array(pred_f)
    true_labels = np.array(true_labels)
    frac, mean, uncertainties = mahalanobis_score(true_labels, pred_y, pred_f, means, covar)
    assert len(mean) == len(frac)
    print(uncertainties.shape)
    return frac, mean, pred_y, true_labels, uncertainties

#RESULT_CACHE_ROOT = "results_cache/"+FAKETYPE+'/fakeRaw15_realRaw30/'
RESULT_CACHE_ROOT = "results_cache/masked/replay_attack/"
#RESULT_CACHE_ROOT = "results_cache/masked/print_attack/"
#RESULT_CACHE_ROOT = "results_cache/ob4/" #if isPublicData else "results_cache/ob15/"

def mahalanobis_score(labels, pred_y, feats, means, covar):
    from sklearn.metrics import accuracy_score
    nb_classes = means.shape[0]
    N = labels.shape[0]
    uncertainties = np.empty(N)
    for f_i, feat in enumerate(feats):
        dists = []
        for c_i in range(nb_classes):
            sub = (feat - means[c_i])[:,np.newaxis]
            distance = np.dot(np.dot(np.transpose(sub), np.linalg.inv(covar)), sub)
            dists.append(distance[0][0])
        uncertainties[f_i] = np.amin(np.array(dists))
    uncertainties /= np.amax(uncertainties)

    fractions = [.1, .2, .3, .4, .5, .6, .7, .8, .9, 1]
    I_uncertainties = np.argsort(uncertainties)
    mean = np.empty_like(fractions)
    for i, frac in enumerate(fractions):
        # Keep only the %-frac of lowest uncertainties
        I = np.zeros(N, dtype=bool)
        I[I_uncertainties[:int(N * frac)]] = True
        mean[i] = accuracy_score(
            labels[I].astype("int8"),
            np.argmax(pred_y[I], axis=1).astype("int8"))
    binary_labels = np.where(labels > 100, 1, 0) if np.amax(labels) > 1 else np.copy(true_labels)
    print("results")
    print("auroc", calc_auroc(binary_labels, uncertainties))
    print("auprc", calc_auprc(binary_labels, uncertainties))
    print("ece", calc_ece(labels, pred_y, uncertainties))
    print("mce", calc_mce(labels, pred_y, uncertainties))
    return fractions, mean, uncertainties
        
        
    #for i in range(nb_classes):
    #uncertainty = 

    

def load_result_cache(expr_name):
    frac_fn = RESULT_CACHE_ROOT + expr_name + ".frac.npy"
    mean_fn = RESULT_CACHE_ROOT + expr_name + ".mean.npy"
    if os.path.exists(frac_fn) and os.path.exists(mean_fn):
        return np.load(frac_fn), np.load(mean_fn)
    return None, None

def save_result_cache(expr_name, frac, mean, preds, labels, uncertainties):
    dn = os.path.join(RESULT_CACHE_ROOT, expr_name)
    os.makedirs(dn, exist_ok=True)
    np.save(os.path.join(dn, "frac.npy"), frac)
    np.save(os.path.join(dn, "mean.npy"), mean)
    np.save(os.path.join(dn, "preds.npy"), preds)
    np.save(os.path.join(dn, "labels.npy"), labels)
    np.save(os.path.join(dn, "uncertainties.npy"), uncertainties)
    print("saved to " + dn)

def get_all_caches():
    caches = []
    for frac_fn in glob.glob(RESULT_CACHE_ROOT + "*.frac.npy"):
        mean_fn = frac_fn.replace("frac.npy", "mean.npy")
        pred_fn = frac_fn.replace("frac.npy", "preds.npy")
        label_fn = frac_fn.replace("frac.npy", "labels.npy")
        expr_name = frac_fn.replace(
            RESULT_CACHE_ROOT, "").replace(".frac.npy", "")
        caches.append((expr_name, np.load(frac_fn), np.load(mean_fn), np.load(pred_fn), np.load(label_fn)))
    return caches

def parentdirname(path):
    return os.path.basename(os.path.dirname(path))

def main(model_fns):
    for model_fn in model_fns:
        expr_name = parentdirname(model_fn)
        isRGB = model_fn.find("rgb") != -1
        isGoogLeNet = model_fn.find("googlenet") != -1
        isC2H = model_fn.find("C2Hdata") != -1
        if isGoogLeNet:
            K.set_image_data_format('channels_first')
        else:
            K.set_image_data_format('channels_last')

        local_expr_name = "mahalanobis" + expr_name
        print("------ ", local_expr_name, " --------")
        print("isRGB: ", isRGB)
        print("isGoogLeNet: ", isGoogLeNet)
        print("isC2Hdata: ", isC2H)

        frac, mean = load_result_cache(local_expr_name)
        if frac is None:
            frac, mean, preds, labels, uncertainties = evaluation(model_fn, isRGB,
                isGoogLeNet, isC2H)
            save_result_cache(local_expr_name, frac, mean, preds, labels, uncertainties)
        else:
            print("** results cache is used.")
        plt.plot(frac, mean, label=local_expr_name)
    plt.legend()
    plt.show()
    plt.savefig(RESULT_CACHE_ROOT+'results.png')
    save_all_to_csv()

def test():
    frac, mean = calc_score(np.array([0, 1, 2]), np.array(
        [[0.3, 0.4, 0.3], [0, 0.9, 0.1], [0, 0.2, 0.8]]))

    plt.plot(frac, mean, label="test")
    plt.legend()
    plt.show()
    exit()

def show_all_cache():
    res_anoscores = []
    for (expr_name, frac, mean, pred, label) in tqdm.tqdm(get_all_caches()):
        plt.plot(frac, mean, label=expr_name)
        print(expr_name)
        anoscores = calc_anomaly_scores(label, pred, ['auc', 'f1', 'ece'])
        anoscores['expr_name'] = expr_name
        res_anoscores.append(anoscores)
    plt.legend()
    #plt.show()
    plt.savefig("results.png")
    save_all_to_csv(res_anoscores)

def save_all_to_csv(all_anomaly_scores=None):
    import csv
    rows = [["frac."]]

    for (expr_name, frac, mean, pred, label) in tqdm.tqdm(get_all_caches()):
        rows[0].append(expr_name)
        if len(rows) == 1:
            for f in frac:
                rows.append([f])
        for i, m in enumerate(mean):
            rows[i + 1].append(m)
    # write
    with open(RESULT_CACHE_ROOT + 'results_graph.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    # anomaly scores
    if all_anomaly_scores:
        anorows = []
        for score_key in all_anomaly_scores[0].keys():
            anorows.append([score_key])
        for anoscores in all_anomaly_scores:
            for i, svalue in enumerate(anoscores.values()):
                anorows[i].append(svalue)
        with open(RESULT_CACHE_ROOT + 'anomaly_scores.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerows(anorows)

if __name__ == "__main__":
    reset_tf(visible_device="0")
    if sys.argv[1] == "all":
        show_all_cache()
        exit()
    model_fns = sys.argv[1:]
    main(model_fns)
