from __future__ import division, print_function

from typing import List, Callable

import numpy
import scipy


############################################################################
# DO NOT MODIFY ABOVE CODES
############################################################################

class KNN:

    def __init__(self, k: int, distance_function) -> float:
        self.k = k
        self.distance_function = distance_function
        self.idenified = []

    def train(self, features: List[List[float]], labels: List[int]):
    	count = 0
    	
    	for val in features:
    		self.idenified.append((val,labels[count]))
    		count+=1 

    # assistance from Giorgio Pizzorni
    # Sorta..
    # Initially, prediction test case on journal was taking about 20 minutes
    # to complete. I asked Giorgio if his was taking that long
    # , and through that conversation, I came to realize I could just sort the 
    # features array once and not have to keep reiterating and finding 
    # the max element not already found. Don't know why I was doing that..

    def predict(self, features: List[List[float]]) -> List[int]:
    	predictions = []

    	for f in features:
    		
    		toSort = []# List of tuples (Distances, Class)
    		
    		for i in range(0,len(self.idenified)):
    			toSort.append((self.distance_function(f,self.idenified[i][0]),self.idenified[i][1]))

    		toSort.sort(key=lambda x: x[0])

    		toVote = [toSort[i][1] for i in range(0,self.k)]
            
    		predictions.append(max(set(toVote), key=toVote.count))


    	return predictions

if __name__ == '__main__':
    print(numpy.__version__)
    print(scipy.__version__)
