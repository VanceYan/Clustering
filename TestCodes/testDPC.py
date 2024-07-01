from Tools.Dataload import Dataload
import Tools.ClusteringShow as CS
import Tools.ClusteringEvaluation as CE
from StandardLibrary.DPC import DPC

def getDataset(FI):
    if FI == 0:
        C = 3
        FileName = "MST_1"
    if FI == 1:
        C = 2
        FileName = "Flame"
    if FI == 2:
        C = 3
        FileName = "Zelnik1"
    if FI == 3:
        C = 3
        FileName = "Zelnik3"
    if FI == 4:
        C = 3
        FileName = "Path-based"
    if FI == 5:
        C = 3
        FileName = "Spiral"
    if FI == 6:
        C = 2
        FileName = "Jain"
    if FI == 7:
        C = 6
        FileName = "Compound"
    if FI == 8:
        C = 15
        FileName = "R15"
    if FI == 9:
        C = 7
        FileName = "Aggregation"
    if FI == 10:
        C = 31
        FileName = "D31"
    if FI == 11:
        C = 15
        FileName = "S2"
    if FI == 12:
        C = 8
        FileName = "Unbalance"
    return FileName, C

for i in range(4,5):
    FileName, C = getDataset(i)
    DatasetPath = "../Dataset/" + FileName + ".csv"
    # Paras: path, label, feature start, feature end
    dataObject = Dataload(DatasetPath, 0, 1, 2)
    print("real labels:")
    print(dataObject.y)
    # Paras: feature matrix, labels, CN, threshold
    cls = DPC(dataObject.X, dataObject.y, C, dataObject.thr)
    cls.fit()
    print("predict labels:")
    print(cls.Labels)
    CS.ShowClustering(cls.X, cls.Labels, cls.ClusterCenters, Algorithm="DPC")
    CE.getClusteringEvaluation(cls.y, cls.Labels, Dataset="Dataset", Algorithm="DPC", saveFlag=True)