from __future__ import division, print_function

from typing import List

import numpy
import scipy


############################################################################
# DO NOT MODIFY ABOVE CODES
############################################################################

class LinearRegression:
    def __init__(self, nb_features: int):
        self.nb_features = nb_features
        self.model = None

    def train(self, features: List[List[float]], values: List[float]):
        np_X = numpy.matrix([ [1.0] + xi for xi in features])
        np_Y = numpy.matrix(values).T
     
        self.model = ((np_X.T*np_X).I)*np_X.T*np_Y

     

    def predict(self, features: List[List[float]]) -> List[float]:
  
        return [numpy.asscalar(self.model.T*(numpy.matrix([1.0] + f).T)) for f in features]
        


    def get_weights(self) -> List[float]:
        """
        for a model y = 1 + 3 * x_0 - 2 * x_1,
        the return value should be [1, 3, -2].
        """
        return [e for e in self.model.flatten().tolist()[0]]


class LinearRegressionWithL2Loss:
    '''Use L2 loss for weight regularization'''
    def __init__(self, nb_features: int, alpha: float):
        self.alpha = alpha
        self.nb_features = nb_features
        self.model = None

    def train(self, features: List[List[float]], values: List[float]):
        np_X = numpy.matrix([ [1.0] + xi for xi in features])
        np_Y = numpy.matrix(values).T
        aI = self.alpha*numpy.identity(np_X.shape[1])

        self.model = ((np_X.T*np_X + aI).I)*np_X.T*np_Y


    def predict(self, features: List[List[float]]) -> List[float]:

        return [numpy.asscalar(self.model.T*(numpy.matrix([1.0] + f).T)) for f in features]

    def get_weights(self) -> List[float]:
        """
        for a model y = 1 + 3 * x_0 - 2 * x_1,
        the return value should be [1, 3, -2].
        """
        return [e for e in self.model.flatten().tolist()[0]]

if __name__ == '__main__':
    print(numpy.__version__)
    print(scipy.__version__)
