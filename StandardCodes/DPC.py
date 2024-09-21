import math
import numpy as np
from sklearn.preprocessing import MinMaxScaler

'''
Density peak clustering
'''


class DPC:
    def __init__(self, X, y, k, dcRatio=1.5, kernel="gaussian"):
        '''
        Constructor
        :param X:       The feature matrix of dataset
        :param y:       Label of dataset
        :param k:       The number of sub clusters in the dataset
        :param dcRatio: The radius ratio (this value is a percentage, which means entering 1 represents 1% of the nearest neighbors),
                        and the recommended range for this value is between 1 and 2
        :param kernel:  Kernel function of density , including "cutoff-kernel" "gaussian-kernel"
        '''
        # Feature matrix
        self.X = X
        # Label vector
        self.y = y
        # The number of clusters to be clustered
        self.NC = k
        # The radius ratio of the dataset
        self.__dcRatio = dcRatio
        # Density kernel type
        self.__kernel = kernel
        # The number of samples included in the dataset
        self.Size = len(self.y)
        # The number of features in the dataset
        self.FN = X.shape[1]
        # Distance matrix
        self.DistanceMatrix = []
        # Cutoff distance
        self.Dc = 0
        # Local density vector
        self.Density = []
        # The object sample vector of a sample
        # indicating the sample with a higher density and closest distance to it when calculating relative distance
        self.Object = np.zeros(self.Size, dtype=int)
        # Relative distance 𝛿
        self.RelativeDistance = np.zeros(self.Size, dtype=float)
        # Decision value
        self.DecisionValue = np.zeros(self.Size, dtype=float)
        # Cluster centers
        self.ClusterCenters = []
        # Predictive labels
        self.Labels = np.zeros(self.Size, dtype=int)


    def __Normalization(self, X):
        '''
        Standardize the feature matrix
        :param X: feature matrix
        :return: Standardized feature matrix
        '''
        # 1. Instantiation converter (feature_range is the normalized range, i.e. minimum maximum)
        transfer = MinMaxScaler(feature_range=(0, 1))
        # 2. Call fit_transform (only needs to handle features)
        return transfer.fit_transform(X)


    def __getDistanceMatrix(self):
        '''
        Calculate the distance matrix of the sample (if there are discrete attributes, calculate the mixed distance)
        :return:
        '''
        # Initialize a distance matrix
        self.DistanceMatrix = np.zeros((self.Size, self.Size), dtype=float)
        # Normalize the input data
        self.X = self.__Normalization(self.X)
        # Calculate distance matrix (based on Euclidean distance)
        for i in range(self.Size - 1):
            for j in range(i + 1, self.Size):
                dis = np.sum(np.power(self.X[i] - self.X[j], 2)) ** 0.5
                self.DistanceMatrix[i][j] = self.DistanceMatrix[j][i] = dis


    def __getCutoffDistance(self):
        '''
        Calculate cutoff distance Dc
        The original author of DPC suggested that the value of Dc should be such that the total number of neighbors
        for each data point is between 1-2% of the total number of points
        :return:
        '''
        # Calculating the number of neighbors based on Dc
        DcNN = math.ceil(self.Size*self.__dcRatio/100)
        # Calculate the nearest neighbor matrix of data points
        NeighborsMatrix = np.sort(self.DistanceMatrix)[:, 1: ]
        # Take the average distance from all samples to the corresponding DcNN-th nearest neighbor as the cutoff distance Dc
        self.Dc = np.mean(NeighborsMatrix[:,DcNN-1])


    def __getLocalDensity(self):
        '''
        Calculate the local density of each sample
        '''
        if(self.__kernel == "gaussian"):
            self.Density = np.sum(np.exp(-np.power(self.DistanceMatrix / self.Dc, 2)), axis=1) - 1
        if(self.__kernel == "cutoff"):
            self.Density = np.sum((self.DistanceMatrix - self.Dc)<0, axis=1)


    def __getRelativeDistance(self):
        '''
        Calculate the relative distance between each sample
        '''
        # Sort density in descending order
        sortDensityIndex = np.argsort(self.Density)[::-1]
        # Set the relative distance of the sample with the highest density (which must be the cluster center)
        self.RelativeDistance[sortDensityIndex[0]] = np.max(self.DistanceMatrix)
        # Calculate relative distance and object sample
        for i in range(1, self.Size):
            tempIndex = sortDensityIndex[i]
            self.Object[tempIndex] = sortDensityIndex[
                np.argmin(self.DistanceMatrix[tempIndex][sortDensityIndex[:i]])]
            self.RelativeDistance[tempIndex] = self.DistanceMatrix[tempIndex][self.Object[tempIndex]]


    def __getClusterCenter(self):
        '''
        Calculate decision values for each sample
        '''
        self.DecisionValue = np.multiply(self.Density, self.RelativeDistance)
        # Initialize cluster center:
        # Select n samples with the highest decision value as cluster centers
        self.ClusterCenters = np.argsort(self.DecisionValue)[::-1][:self.NC]
        # Assign different label values to the selected initial cluster center: 1, 2, 3,... (positive integer)
        for i in range(self.NC):
            self.Labels[self.ClusterCenters[i]] = i + 1


    def __LabelAllocation(self):
        '''
        Assign labels (clustering)
        :return:
        '''
        # Clustering by searching for density peaks
        for i in range(self.Size):
            # Samples that have not been assigned labels need to continuously search upwards based on the object vector
            # for samples that have been assigned labels
            index = i
            # Record the sample index of the label to be assigned
            allocatedChain = []
            while self.Labels[index] == 0:
                allocatedChain.append(index)
                # Get object sample
                index = self.Object[index]
            # Label all samples in the allocatedChain
            for sample in allocatedChain:
                self.Labels[sample] = self.Labels[index]


    def fit(self):
        self.__getDistanceMatrix()
        self.__getCutoffDistance()
        self.__getLocalDensity()
        self.__getRelativeDistance()
        self.__getClusterCenter()
        self.__LabelAllocation()