import os
import pandas as pd
import numpy as np


class Dataload:
    '''
    Data loading class (preprocessing)
    This class is used to calculate the following three parameters:
    1. Feature matrix X
    2. True label y
    3. If there are discrete attributes, a value Thr will be returned.
        At this point, the attribute of [0 thr] in the feature matrix is a continuous value,
        and the attribute of [Thr, -1] is a discrete value
    '''
    def __init__(self, dataPath, labelIndex, featureStartIndex, featureEndIndex, header=False):
        '''
        :param dataPath:  Dataset path
        :param labelIndex: Label index
        :param featureStartIndex: Attribute starting index
        :param featureEndIndex: Attribute end index
        :param header: Is there a header present
        '''
        # Check if the file exists
        if not os.path.exists(dataPath):
            print("\n*************************")
            print("*  File is not exists!  *")
            print("*************************\n")
            self.X = None
            self.y = None
            self.dataSet = None
            return
        # Get the original dataset according to the specified path
        if header:
            self.dataSet = pd.read_csv(dataPath)
        else:
            self.dataSet = pd.read_csv(dataPath, header=None)
        # Get preliminary feature matrix
        self.__X = self.dataSet.to_numpy()[:, featureStartIndex:featureEndIndex + 1]
        # Get tags
        self.y = self.dataSet.to_numpy()[:, labelIndex]
        # Preprocessing the feature matrix
        self.__getX()
        # Map the values in the label vector to integer values
        self.__getY()


    def __getX(self):
        # Get columns that can be converted to numeric type
        numeric_col = []
        row = 0
        ROW, COL = len(self.__X), len(self.__X[0])
        while True:
            for col in range(COL):
                if self.__is_number(self.__X[row][col]):
                    numeric_col.append(col)
                if self.__X[row][col] == "?":
                    row += 1
                    numeric_col = []
                    break
            if col == COL - 1:
                break
        # Get discrete columns
        dispersed_col = [x for x in range(COL) if x not in numeric_col]
        # Convert numeric attributes to float (and handle missing values)
        # Principle for handling missing values: take the mean
        for col in numeric_col:
            numberList = []
            missValueRow = []
            for row in range(ROW):
                if self.__is_number(self.__X[row][col]):
                    self.__X[row][col] = float(self.__X[row][col])
                    numberList.append(self.__X[row][col])
                else:
                    missValueRow.append(row)
            meanValue = np.mean(numberList)
            for row in missValueRow:
                self.__X[row][col] = meanValue
        # If there are no discrete columns in the entire dataset, return the current feature matrix directly
        if len(dispersed_col) == 0:
            self.X = np.array(self.__X)
            self.thr = -1
        # Otherwise, it is necessary to reconstruct the feature matrix
        else:
            self.X = []
            for col in numeric_col:
                self.X.append(self.__X[:,col])
            for col in dispersed_col:
                self.X.append(self.__X[:, col])
            self.X = np.array(self.X).T
            self.thr = len(numeric_col)


    def __getY(self):
        # Establishing a mapping relationship between labels and integer data
        labelCodeCnt = 1
        labelCodeDict = {}
        for labelName in np.unique(self.y):
            labelCodeDict[labelName] = labelCodeCnt
            labelCodeCnt += 1
        # According to the mapping rules obtained earlier, convert the original data labels to numeric types
        Labels = []
        for var in self.y:
            Labels.append(labelCodeDict[var])
        self.y = Labels


    def __is_number(self, s):
        '''
        Determine whether a character can be converted into numeric data
        :param s: String to be tested
        :return:
        '''
        try:
            float(s)
            return True
        except ValueError:
            pass
        try:
            import unicodedata
            unicodedata.numeric(s)
            return True
        except (TypeError, ValueError):
            pass
        return False