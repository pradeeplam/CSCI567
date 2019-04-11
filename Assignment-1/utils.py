from typing import List

import numpy as np


def mean_squared_error(y_true: List[float], y_pred: List[float]) -> float:
    assert len(y_true) == len(y_pred)

    return np.sum(np.square(np.matrix(y_true) - np.matrix(y_pred)))/ len(y_true)


def f1_score(real_labels: List[int], predicted_labels: List[int]) -> float:
    """
    f1 score: https://en.wikipedia.org/wiki/F1_score
    """
    assert len(real_labels) == len(predicted_labels)

    # Piazza: 1 is + and 0 is -
    
    # All correct positive results
    posCorr = float(sum([1 for i in range(0,len(real_labels)) if(real_labels[i]*predicted_labels[i] == 1)]))
    
    # All results that are positive
    posPred = float(sum([1 for i in range(0,len(predicted_labels)) if(predicted_labels[i] == 1)]))
    
    # All samples that should be positive
    posReal = float(sum([1 for i in range(0,len(real_labels)) if(real_labels[i] == 1)]))

    p = posCorr/posPred
    r = posCorr/posReal

    if p == 0.0 and r == 0.0:
        return 0.0

    return 2.0*(p*r)/(p+r)

def polynomial_features(features: List[List[float]], k: int) -> List[List[float]]:
    poly_features = []
    
    for f in features:
        oneElement = []
        for n in range(1,k):
            oneElement += [ e**n for e in f ]
        poly_features.append(oneElement)

    return poly_features


def euclidean_distance(point1: List[float], point2: List[float]) -> float:
    return np.sum(np.square(np.matrix(point1) - np.matrix(point2)))**0.5


def inner_product_distance(point1: List[float], point2: List[float]) -> float:
    return np.asscalar(np.matrix(point1)*np.matrix(point2).T)


def gaussian_kernel_distance(point1: List[float], point2: List[float]) -> float:
    return -np.exp((-0.5)*np.sum(np.square(np.matrix(point1) - np.matrix(point2))))


class NormalizationScaler:
    def __init__(self):
        pass

    def __call__(self, features: List[List[float]]) -> List[List[float]]:
        """
        normalize the feature vector for each sample . For example,
        if the input features = [[3, 4], [1, -1], [0, 0]],
        the output should be [[0.6, 0.8], [0.707107, -0.707107], [0, 0]]
        """
        normal = []
        for x in features:
            if x == [float(0)]*len(x):
                normal.append(x)
            else:
                normal.append((np.matrix(x)/(np.asscalar(np.matrix(x)*np.matrix(x).T)**0.5)).tolist()[0])

        return normal



class MinMaxScaler:
    """
    You should keep some states inside the object.
    You can assume that the parameter of the first __call__
        must be the training set.

    Note:
        1. you may assume the parameters are valid when __call__
            is being called the first time (you can find min and max).

    Example:
        train_features = [[0, 10], [2, 0]]
        test_features = [[20, 1]]

        scaler = MinMaxScale()
        train_features_scaled = scaler(train_features)
        # now train_features_scaled should be [[0, 1], [1, 0]]

        test_features_sacled = scaler(test_features)
        # now test_features_scaled should be [[10, 0.1]]

        new_scaler = MinMaxScale() # creating a new scaler
        _ = new_scaler([[1, 1], [0, 0]]) # new trainfeatures
        test_features_scaled = new_scaler(test_features)
        # now test_features_scaled should be [[20, 1]]

    """

    def __init__(self):
        self.isTrain = True
        self.min = []
        self.max = []

    def __call__(self, features: List[List[float]]) -> List[List[float]]:
        """
        normalize the feature vector for each sample . For example,
        if the input features = [[2, -1], [-1, 5], [0, 0]],
        the output should be [[1, 0], [0, 1], [0.333333, 0.16667]]
        """
        if self.isTrain == True:
            for d in range(0,len(features[0])): 
                sFeatures = sorted(features, key=lambda x: x[d])
                # Get min
                self.min.append(sFeatures[0][d])
                # Get max 
                self.max.append(sFeatures[len(features)-1][d])
            self.isTrain = False


        # Normalize each element
        normalized = []
        for x in features:
            point = []

            for i in range(0,len(x)):
                point.append((x[i]-self.min[i])/(self.max[i]-self.min[i]))

            normalized.append(point)

        return normalized