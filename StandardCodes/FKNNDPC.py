import numpy as np
from sklearn.preprocessing import MinMaxScaler
from queue import Queue

'''
Density peak clustering based fuzzy weighted K Nearest Neighbours
'''


class FKNNDPC:
    def __init__(self, X, y, k, K):
        '''
        Constructor
        :param X:       The feature matrix of dataset
        :param y:       Label of dataset
        :param k:       The number of sub clusters in the dataset
        :param K:       The number of nearest neighbors for each sample
        '''
        # Feature matrix
        self.X = X
        # Label vector
        self.y = y
        # The number of clusters to be clustered
        self.NC = k
        # The number of nearest neighbors for each sample
        self.K = K
        # The number of samples included in the dataset
        self.Size = len(self.y)
        # The number of features in the dataset
        self.FN = X.shape[1]
        # Distance matrix
        self.DistanceMatrix = []
        # K-nearest neighbor matrix of the sample
        self.KNNMatrix = []
        # The K-nearest neighbor distance matrix of the sample
        self.KNNDMatrix = []
        # Local density vector
        self.Density = []
        # The object sample vector of a sample
        self.Object = np.zeros(self.Size, dtype=int)
        # Relative distance 𝛿
        self.RelativeDistance = np.zeros(self.Size, dtype=float)
        # Decision value
        self.DecisionValue = np.zeros(self.Size, dtype=float)
        # Cluster centers
        self.ClusterCenters = []
        # Get outliers
        self.Outlier = []
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


    def __getKNNInfomation(self):
        '''
       Calculate the K-nearest neighbor information of the sample
        :return:
        '''
        # Calculate the K-nearest neighbor matrix of the sample
        self.KNNMatrix = np.argsort(self.DistanceMatrix)[:, 1: self.K + 1]
        # Calculate the K-nearest neighbor distance matrix of the sample
        self.KNNDMatrix = np.sort(self.DistanceMatrix)[:, 1: self.K + 1]


    def __getLocalDensity(self):
        '''
        Calculate the local density of each sample
        '''
        self.Density = np.sum(np.exp(-self.KNNDMatrix), axis=1)


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
        Calculate decision values for each sample and get the cluster centers
        '''
        self.DecisionValue = np.multiply(self.Density, self.RelativeDistance)
        # Initialize cluster center:
        # Select n samples with the highest decision value as cluster centers
        self.ClusterCenters = np.argsort(self.DecisionValue)[-self.NC:][::-1]
        # Assign different label values to the selected initial cluster center: 1, 2, 3,... (positive integer)
        for i in range(self.NC):
            self.Labels[self.ClusterCenters[i]] = i + 1


    def __getOutlier(self):
        '''
        Calculate the set of outliers
        :return:
        '''
        # Calculate the distance of the K-th nearest neighbor for each sample
        KthNNDistance = self.KNNDMatrix[:,self.K-1]
        # Calculate outlier threshold
        OutlierThreshold = np.mean(KthNNDistance)
        # Get outliers
        self.Outlier = np.arange(self.Size)[KthNNDistance > OutlierThreshold]


    def __LabelAllocation1(self):
        '''
        Label allocation strategy 1: Assign to non outlier points
        :return:
        '''
        q = Queue()
        # Traverse cluster centers
        for center in self.ClusterCenters:
            # Push all K-nearest neighbor samples at the center of the current cluster into queue q
            # (and classify them into the cluster to which this sample belongs)
            for neighbor in self.KNNMatrix[center]:
                self.Labels[neighbor] = self.Labels[center]
                q.put(neighbor)
            # The requirement for determining whether there are still samples that can be classified in each nearest neighbor is:
            # 1 Unassigned
            # 2 Non outlier points
            # 3 The distance between the current sample r and a certain point p in its K-nearest neighbor sample is
            # less than the mean distance between the sample point r and its K-nearest neighbor
            while not q.empty():
                r = q.get()
                Radius = np.mean(self.KNNDMatrix[r])
                for p in self.KNNMatrix[r]:
                    if (self.Labels[p] == 0) and (p not in self.Outlier) and (self.DistanceMatrix[r][p] <= Radius):
                        self.Labels[p] = self.Labels[r]
                        q.put(p)


    def __LabelAllocation2(self):
        '''
        Label allocation strategy 2: Assign to outliers and non outliers that have not been assigned after strategy 1
        :return:
        '''
        # Calculate sample similarity matrix
        self.SimilarMatrix = 1 / (1 + self.DistanceMatrix)
        # Calculate the sum of K-nearest neighbor similarity for each sample (used to calculate weighted membership)
        KNNSimilarSumVector = np.sum(1 / (1 + self.KNNDMatrix), 1)
        #Define the recognition matrix
        # (this structure is a dictionary, where the key corresponds to the sample number, and the value corresponds to
        # the K-nearest neighbor classification of the sample, which is a list)
        identifyMatrix = {}
        # Initialize recognition matrix (h * C):
        # Calculate the membership degree of each unassigned sample belonging to the cluster Ci
        for index in range(self.Size):
            # Skip allocated samples
            if self.Labels[index] > 0:
                continue
            # Defining K-nearest neighbor classification vectors for samples
            # Note: Due to the fact that the identification of sample clusters starts from 1 and the array index
            # starts from 0, the number of columns in the initial construction of the recognition matrix should be +1
            neighborClassVector = np.zeros(self.NC + 1, dtype=float)
            # Calculate the membership degree of the current sample on various clusters (requiring traversal of its neighbors)
            for neighbor in self.KNNMatrix[index]:
                if self.Labels[neighbor] > 0:
                    neighborClassVector[self.Labels[neighbor]] += self.SimilarMatrix[index][neighbor]**2 / KNNSimilarSumVector[neighbor]
            # Add statistical results to the identification matrix
            identifyMatrix[index] = neighborClassVector
        # Constructing recognition vectors: selecting the maximum membership degree of each sample from the recognition matrix
        # This vector has a dictionary structure: key: sample index, value: [membership degree, corresponding class cluster number]
        identifyVector = {}
        for index in identifyMatrix:
            # Extract the K-nearest neighbor classification vector of the specified sample from the dictionary
            npary = np.array(identifyMatrix[index])
            # Calculate the maximum membership degree and corresponding cluster number from the K-nearest neighbor classification vector
            identifyVector[index] = [np.max(npary), np.argmax(npary)]
        # Assign labels based on recognition vectors
        while len(identifyVector) > 0:
            # Select a sample with the highest membership degree from it
            object = max(identifyVector, key=identifyVector.get)
            # If the nearest neighbors of the current extracted sample have not been classified, it indicates that
            # the sample is in an extremely isolated small cluster, and the loop is exited
            if identifyVector[object][0] == 0:
                break
            # Classify the sample into the cluster indicated by its maximum membership degree
            self.Labels[object] = identifyVector[object][1]
            # Determine if there are any unassigned samples in the K-nearest neighbors of this sample,
            # and update the membership degree of the unassigned samples corresponding to this sample
            # Note: The original author of FKNN_DPC made this processing based on the assumption that if sample x is
            # the nearest neighbor of sample y, then y is also the nearest neighbor of x
            for neighboor in self.KNNMatrix[object]:
                if self.Labels[neighboor] == 0:
                    # Update recognition matrix
                    identifyMatrix[neighboor][identifyVector[object][1]] += self.SimilarMatrix[neighboor][object]**2 / KNNSimilarSumVector[neighboor]
                    # Update the recognition vector (note: do not continue updating data that is no longer
                    # in the recognition vector, otherwise it may lead to falling into a dead cycle)
                    identifyVector[neighboor] = [np.max(identifyMatrix[neighboor]), np.argmax(identifyMatrix[neighboor])]
            # Pop up the current sample from the recognition matrix and vector
            identifyMatrix.pop(object)
            identifyVector.pop(object)


    def __LabelAllocation3(self):
        '''
        For extremely isolated samples, there may be a situation where mutual exclusion cannot be broken,
        so it is necessary to cluster them separately
        Cluster criterion: Merge unclassified samples into the cluster of the closest assigned sample
        :return:
        '''
        for index in range(self.Size):
            if self.Labels[index] == 0:
                # Calculate the nearest neighbor vector of the current sample
                NearstNeighbors = np.argsort(self.DistanceMatrix[index])[1:]
                # Traverse neighbors to classify the current sample into the cluster where the most recently allocated sample is located
                for neighbor in NearstNeighbors:
                    if self.Labels[neighbor] > 0:
                        self.Labels[index] = self.Labels[neighbor]
                        break


    def fit(self):
        self.__getDistanceMatrix()
        self.__getKNNInfomation()
        self.__getLocalDensity()
        self.__getRelativeDistance()
        self.__getClusterCenter()
        self.__getOutlier()
        self.__LabelAllocation1()
        self.__LabelAllocation2()
        self.__LabelAllocation3()