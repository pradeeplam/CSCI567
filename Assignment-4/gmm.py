import numpy as np
from kmeans import KMeans


class GMM():
    '''
        Fits a Gausian Mixture model to the data.

        attrs:
            n_cluster : Number of mixtures
            e : error tolerance
            max_iter : maximum number of updates
            init : initialization of means and variance
                Can be 'random' or 'kmeans'
            means : means of gaussian mixtures
            variances : variance of gaussian mixtures
            pi_k : mixture probabilities of different component
    '''

    def __init__(self, n_cluster, init='k_means', max_iter=100, e=0.0001):
        self.n_cluster = n_cluster
        self.e = e
        self.max_iter = max_iter
        self.init = init
        self.means = None
        self.variances = None
        self.pi_k = None

    # Helper fun: N(x|u,cov)
    def calc_n(self,x):
        # Invert self.variances matrix
        inverted = np.zeros(self.variances.shape)
        for i in range(self.n_cluster):
            while True:
                try:
                    inverted[i] = np.linalg.inv(self.variances[i])
                    break
                except:
                    self.variances[i] += np.identity(self.means[0].size)*(10**(-3)) 

        dPart = np.linalg.det(2*np.pi*self.variances)**(-0.5)

        xminu = (x-self.means[:,None])
        ePart = np.exp(-0.5*np.sum((xminu@inverted)*xminu,axis=2))

        # Row->which k
        # Col->which x
        # Specific pos p(xi|z=k)
        normal = dPart[:,None]*ePart

        return normal

    # Helper fun: gamma
    def calc_g(self,normal):
        top = self.pi_k[:,None]*normal
        bot = np.sum(self.pi_k[:,None]*normal,axis=0)

        gamma_k = top/bot

        return gamma_k

    # Helper fun: mean
    # What if n_k = 0?? -> Perhaps the prob of this is very very low
    def calc_u(self,gamma_k,x,n_k):
        self.means = np.einsum('ij,jk->ik',gamma_k,x)
        self.means /= n_k[:,None] 


    # Helper fun: variance
    def calc_v(self,gamma_k,x,n_k):
        uminx = (self.means[:,None]-x) # k grouped u-x

        # Broadcast for extra dimensions
        trans = np.einsum('...i,...j->...ij',uminx,uminx) # k grouped (u-x)*(u-x)T

        # Broadcast for extra dimensions
        corrected = np.einsum('ij,ij...->ij...',gamma_k,trans)
        summed = np.sum(corrected,axis=1)

        self.variances = np.zeros(summed.shape)
        self.variances = (np.divide(summed.T,n_k,where=(n_k!=0))).T
    
    def fit(self, x):
        '''
            Fits a GMM to x.

            x: is a NXD size numpy array
            updates:
                self.means
                self.variances
                self.pi_k
        '''
        assert len(x.shape) == 2, 'x can only be 2 dimensional'

        np.random.seed(42)
        N, D = x.shape

        if (self.init == 'k_means'):
            # TODO
            # - comment/remove the exception
            # - initialize means using k-means clustering
            # - compute variance and pi_k

            # DONOT MODIFY CODE ABOVE THIS LINE
            #raise Exception(
            #    'Implement initialization of variances, means, pi_k using k-means')
            
            kmObject = KMeans(n_cluster=self.n_cluster, max_iter=self.max_iter, e=self.e)
            self.means,r,_ = kmObject.fit(x)
            
            # Convert r back to one-hot
            rHot = np.zeros((N,self.n_cluster))
            rHot[np.arange(r.size),r] = 1
            gamma_k = rHot.T

            # Compute nk
            n_k = np.sum(gamma_k,axis=1)

            self.calc_v(gamma_k,x,n_k)

            # Computer pi_k
            self.pi_k = n_k/N

            # DONOT MODIFY CODE BELOW THIS LINE

        elif (self.init == 'random'):
            # TODO
            # - comment/remove the exception
            # - initialize means randomly
            # - compute variance and pi_k

            # DONOT MODIFY CODE ABOVE THIS LINE
            #raise Exception(
            #    'Implement initialization of variances, means, pi_k randomly')

            # Very unlikely to crash..
            # Ya..Ya. Somewhat biased because once you use a num cant use exact num again
            self.means = np.random.choice(np.random.uniform(0,1,self.n_cluster*D),replace=False,size=(self.n_cluster,D))

            self.variances = np.stack([np.identity(D)]*self.n_cluster,axis=0)

            self.pi_k = np.array([1/self.n_cluster]*self.n_cluster)

            # DONOT MODIFY CODE BELOW THIS LINE

        else:
            raise Exception('Invalid initialization provided')

        # TODO
        # - comment/remove the exception
        # - find the optimal means, variances, and pi_k and assign it to self
        # - return number of updates done to reach the optimal values.
        # Hint: Try to seperate E & M step for clarity

        # DONOT MODIFY CODE ABOVE THIS LINE
        #raise Exception('Implement fit function (filename: gmm.py)')
        c = 0
        
        lOld = self.compute_log_likelihood(x)

        while c < self.max_iter:

            # E step
            gamma_k = self.calc_g(self.calc_n(x))

            # M step
            n_k = np.sum(gamma_k,axis=1)

            self.calc_u(gamma_k,x,n_k) 
            self.calc_v(gamma_k,x,n_k)
            self.pi_k = n_k/N  

            lNew = self.compute_log_likelihood(x)

            if np.absolute(lNew-lOld) <= self.e:
                break

            lOld = lNew

            c += 1

        return c

        # DONOT MODIFY CODE BELOW THIS LINE
    
    def sample(self, N):
        '''
        sample from the GMM model

        N is a positive integer
        return : NXD array of samples

        '''
        assert type(N) == int and N > 0, 'N should be a positive integer'
        np.random.seed(42)
        if (self.means is None):
            raise Exception('Train GMM before sampling')

        # TODO
        # - comment/remove the exception
        # - generate samples from the GMM
        # - return the samples

        # DONOT MODIFY CODE ABOVE THIS LINE
        #raise Exception('Implement sample function in gmm.py')
        which = np.random.choice(self.n_cluster, size=N, p=self.pi_k)

        toReturn = np.zeros((N,self.means[0].size))
        for i in range(0,which.size):
            index = which[i]
            toReturn[i] = np.random.multivariate_normal(self.means[index],self.variances[index])

        return toReturn
        # DONOT MODIFY CODE BELOW THIS LINE


    def compute_log_likelihood(self, x):
        '''
            Return log-likelihood for the data

            x is a NXD matrix
            return : a float number which is the log-likelihood of data
        '''
        assert len(x.shape) == 2,  'x can only be 2 dimensional'
        # TODO
        # - comment/remove the exception
        # - calculate log-likelihood using means, variances and pi_k attr in self
        # - return the log-likelihood
        # Note: you can call this function in fit function (if required)
        
        # DONOT MODIFY CODE ABOVE THIS LINE
        #raise Exception('Implement compute_log_likelihood function in gmm.py')

        # Calculate normal
        normal = self.calc_n(x)

        return np.asscalar(np.sum(np.log(np.sum(self.pi_k[:,None]*normal,axis=0))))

        # DONOT MODIFY CODE BELOW THIS LINE
