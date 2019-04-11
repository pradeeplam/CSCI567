import numpy as np


class KMeans():

    '''
        Class KMeans:
        Attr:
            n_cluster - Number of cluster for kmeans clustering
            max_iter - maximum updates for kmeans clustering
            e - error tolerance
    '''

    def __init__(self, n_cluster, max_iter=100, e=0.0001):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e

    def fit(self, x):
        '''
            Finds n_cluster in the data x
            params:
                x - N X D numpy array
            returns:
                A tuple
                (centroids or means, membership, number_of_updates )
            Note: Number of iterations is the number of time you update means other than initialization
        '''
        assert len(x.shape) == 2, "fit function takes 2-D numpy arrays as input"
        np.random.seed(42)
        N, D = x.shape

        # TODO:
        # - comment/remove the exception.
        # - Initialize means by picking self.n_cluster from N data points
        # - Update means and membership untill convergence or untill you have made self.max_iter updates.
        # - return (means, membership, number_of_updates)

        # DONOT CHANGE CODE ABOVE THIS LINE
        # raise Exception('Implement fit function in KMeans class (filename: kmeans.py')
        
        # Init means of each cluster by choosing at random from data (Points *must* be unique)
        # The assumption is made that more unique data points than clusters
        
        # Only want unique x values
        uX = np.unique(x, axis=0) 
        uN, _ = uX.shape

        # Can't implement using method described in instructions if the case
        if self.n_cluster > uN:
            raise Exception('Cannot use method w/ fewer clusters than unique data!')

        # np.arange(uN) -> Array, Range of indexes
        # np.random.shuffle(...) -> Shuffles this range
        # Get index for locations of init mean values
        meanLoc = np.arange(uN)
        np.random.shuffle(meanLoc)
        meanLoc = meanLoc[:self.n_cluster]
        
        # Extract from unique x the chosen x vectors (Rows)
        # to serve as initial means 
        u = np.take(uX,meanLoc,axis=0)

        r = np.zeros((N,self.n_cluster))

        jOld = 0

        c = 0
        while c < self.max_iter:

            # Compute r
            a1 = ((x[:,np.newaxis]-u)**2)
            a2 = np.sum(a1,axis=2)
            a3 = np.argmin(a2,axis=1)
            r[:] = 0.0 # Reset
            r[np.arange(a3.size),a3] = 1

            # Compute distortion
            jNew = np.sum(a2*r)/N

            # Break on convergence
            #print(np.absolute(jNew-jOld))
            
            if np.absolute(jNew-jOld) <= self.e:
                break

            jOld = jNew

            # Compute mean
            gbyk = r.T[:,:,None]*x # Group by k (Make others 0 ) 
            numk = np.sum(r.T,axis=1) # Num of dp in each group
            top = np.sum(gbyk,axis=1)
            bot = numk[:,None]

            # Have to divide by 1.0 to ensure u is float
            u = np.divide(top,bot,out=u/1.0,where=(bot!=0))

            c += 1
        
        r = np.argmax(r, axis=1) # Convert from one hot to index

        return (u,r,c)
        # DONOT CHANGE CODE BELOW THIS LINE


class KMeansClassifier():

    '''
        Class KMeansClassifier:
        Attr:
            n_cluster - Number of cluster for kmeans clustering
            max_iter - maximum updates for kmeans clustering
            e - error tolerance
    '''

    def __init__(self, n_cluster, max_iter=100, e=1e-6):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e

    def fit(self, x, y):
        '''
            Train the classifier
            params:
                x - N X D size  numpy array
                y - N size numpy array of labels
            returns:
                None
            Stores following attributes:
                self.centroids : centroids obtained by kmeans clustering
                self.centroid_labels : labels of each centroid obtained by
                    majority voting
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"
        assert len(y.shape) == 1, "y should be a 1-D numpy array"
        assert y.shape[0] == x.shape[0], "y and x should have same rows"

        np.random.seed(42)
        N, D = x.shape
        # TODO:
        # - comment/remove the exception.
        # - Implement the classifier
        # - assign means to centroids
        # - assign labels to centroid_labels

        # DONOT CHANGE CODE ABOVE THIS LINE
        #raise Exception(
        #    'Implement fit function in KMeansClassifier class (filename: kmeans.py')

        # Run k-means clustering to find centroids and membership
        km = KMeans(self.n_cluster,self.max_iter,self.e)
        centroids,labels,_ = km.fit(x)

        # Predict the same label as the nearest centroid (Based on votes)
        centroid_labels = np.zeros((N,self.n_cluster,np.max(y)+1))
        centroid_labels[np.arange(N),labels,y] = 1 # Hacky as hell but im fking brilliant
        centroid_labels = np.sum(centroid_labels,axis=0)
        centroid_labels = np.argmax(centroid_labels,axis=1) # For all 0 -> "First occurance" is 0

        # DONOT CHANGE CODE BELOW THIS LINE

        self.centroid_labels = centroid_labels
        self.centroids = centroids

        assert self.centroid_labels.shape == (self.n_cluster,), 'centroid_labels should be a vector of shape {}'.format(
            self.n_cluster)

        assert self.centroids.shape == (self.n_cluster, D), 'centroid should be a numpy array of shape {} X {}'.format(
            self.n_cluster, D)

    def predict(self, x):
        '''
            Predict function

            params:
                x - N X D size  numpy array
            returns:
                predicted labels - numpy array of size (N,)
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"

        np.random.seed(42)
        N, D = x.shape
        # TODO:
        # - comment/remove the exception.
        # - Implement the prediction algorithm
        # - return labels


        # DONOT CHANGE CODE ABOVE THIS LINE
        #raise Exception(
        #    'Implement predict function in KMeansClassifier class (filename: kmeans.py')

        # match x w/ self.centroids and self.centroid_labels
        # Same stuff as done in kmeans

        a1 = ((x[:,None]-self.centroids)**2)
        a2 = np.sum(a1,axis=2)
        a3 = np.argmin(a2,axis=1) # Know which centroid every point is
        
        return np.take(self.centroid_labels,a3)

        # DONOT CHANGE CODE BELOW THIS LINE
