from Tools.Dataload import Dataload
import Tools.ClusteringShow as CS
import Tools.ClusteringEvaluation as CE
from StandardLibrary.FKNNDPC import FKNNDPC

def getDataset(FI):
    if FI == 0:
        C, K = 3, 4
        FileName = "MST_1"
    if FI == 1:
        C, K = 2, 3
        FileName = "Flame"
    if FI == 2:
        C, K = 3, 6
        FileName = "Zelnik1"
    if FI == 3:
        C, K = 3, 6
        FileName = "Zelnik3"
    if FI == 4:
        C, K = 3, 5
        FileName = "Path-based"
    if FI == 5:
        C, K = 3, 3
        FileName = "Spiral"
    if FI == 6:
        C, K = 2, 11
        FileName = "Jain"
    if FI == 7:
        C, K = 6, 5
        FileName = "Compound"
    if FI == 8:
        C, K = 15, 7
        FileName = "R15"
    if FI == 9:
        C, K = 7, 5
        FileName = "Aggregation"
    if FI == 10:
        C, K = 31, 5
        FileName = "D31"
    if FI == 11:
        C, K = 15, 5
        FileName = "S2"
    if FI == 12:
        C, K = 8, 3
        FileName = "Unbalance"
    return FileName, C, K

for i in range(4,5):
    FileName, C, K = getDataset(i)
    DatasetPath = "../Dataset/" + FileName + ".csv"
    # Paras: path, label, feature start, feature end
    dataObject = Dataload(DatasetPath, 0, 1, 2)
    print("real labels:")
    print(dataObject.y)
    # Paras: feature matrix, labels, CN, threshold, K value of KNN
    cls = FKNNDPC(dataObject.X, dataObject.y, C, dataObject.thr, K)
    cls.fit()
    print("predict labels:")
    print(cls.Labels)
    CS.ShowClustering(cls.X, cls.Labels, cls.ClusterCenters, Algorithm="FKNN-DPC")
    CE.getClusteringEvaluation(cls.y, cls.Labels, Dataset=FileName, Algorithm="FKNN-DPC", saveFlag=True)
