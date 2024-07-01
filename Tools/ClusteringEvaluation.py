import numpy as np
from sklearn import metrics

'''
Cluster evaluation indicators:
ACC (Accuracy of Clustering)
FMI（Fowlkes–Mallows index）
ARI (Adjusted Rand Index)
AMI (Adjusted Mutual Information)
NMI (Normalized Mutual Information)
'''

def getPurity(realLabels, preLabels):
    '''
    Get purity of clustering
    :param realLabels:  real labels
    :param preLabels:   predictive labels
    :return:
    '''
    # Dividing sub clusters based on predictive labels
    # Data size
    DatSize = len(preLabels)
    # Label catalogue
    Labels = np.unique(preLabels)
    LN = len(Labels)
    # Correctly clustered samples
    CCS = 0
    for label in Labels:
        # Classification vector
        CV = np.zeros(LN+1, dtype=int)
        # Statistics on the clustering situation in the current cluster
        indeces = np.arange(DatSize)[preLabels == label]
        for index in indeces:
            CV[realLabels[index]] += 1
        # Count the number of samples with the highest number of correct clusters
        CCS += max(CV)
    return CCS / DatSize


def getACC(realLabels, preLabels):
    '''
    Get accuracy of clustering
    :param realLabels:  real labels
    :param preLabels:   predictive labels
    :return:
    '''
    return metrics.accuracy_score(realLabels, preLabels)


def getFMI(realLabels, preLabels):
    '''
    Get Fowlkes–Mallows index
    :param realLabels:  real labels
    :param preLabels:   predictive labels
    :return:
    '''
    return metrics.fowlkes_mallows_score(realLabels, preLabels)


def getARI(realLabels, preLabels, otherIndex = False, beta=1):
    '''
    Get Adjusted Rand Index
    :param realLabels:  real labels
    :param preLabels:   predictive labels
    :return:
    '''
    # 计算 FN 矩阵
    (tn, fp), (fn, tp) = metrics.pair_confusion_matrix(realLabels, preLabels)
    ri = (tp + tn) / (tp + tn + fp + fn)
    ari = 2. * (tp * tn - fn * fp) / ((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn))
    p, r = tp / (tp + fp), tp / (tp + fn)
    f_beta = (1 + beta ** 2) * (p * r / ((beta ** 2) * p + r))
    # 反馈结果
    if not otherIndex:
        return ari
    else:
        return ari, ri, f_beta


def getAMI(realLabels, preLabels):
    '''
    Get Adjusted Mutual Information
    :param realLabels:  real labels
    :param preLabels:   predictive labels
    :return:
    '''
    return metrics.adjusted_mutual_info_score(realLabels, preLabels)


def getNMI(realLabels, preLabels):
    '''
    Get Normalized Mutual Information
    :param realLabels:  real labels
    :param preLabels:   predictive labels
    :return:
    '''
    return metrics.normalized_mutual_info_score(realLabels, preLabels)


def getClusteringEvaluation(realLabels, preLabels, Dataset = "Null", Algorithm = "Null", saveFlag = False):
    if Algorithm == "Null" and Dataset == "Null":
        s = "Clustering evaluation"
    elif Algorithm != "Null" and Dataset != "Null":
        s = "Clustering evaluation on %s based on %s" % (Dataset, Algorithm)
    elif Dataset != "Null":
        s = "Clustering evaluation on %s" % Dataset
    else:
        s = "Clustering evaluation based on %s" % Algorithm
    evaluation = ""
    evaluation += "Purity：%.4f\n" % getPurity(realLabels, preLabels)
    evaluation += "ACC：\t%.4f\n" % getACC(realLabels, preLabels)
    evaluation += "FMI：\t%.4f\n" % getFMI(realLabels, preLabels)
    evaluation += "ARI: \t%.4f\n" % getARI(realLabels, preLabels)
    evaluation += "AMI: \t%.4f\n" % getAMI(realLabels, preLabels)
    evaluation += "NMI: \t%.4f" % getNMI(realLabels, preLabels)
    print("\n\n-----------------------------------------------------")
    print(s)
    print(evaluation)
    print("------------------------ END ------------------------\n\n")
    if saveFlag:
        with open("../Texts/ClusteringEvaluation_" + Dataset + "_" + Algorithm + ".txt", "w") as f:
            f.write(evaluation)
        f.close()