# -*- coding: utf-8 -*-

"""
Created on Tue Oct 1 14:19:07 2022

Koushik A Manjunatha
"""


import numpy as np
import numexpr as ne
import math
import cvxopt
from cvxopt import matrix, spmatrix # need version 1.2.6
cvxopt.solvers.options['show_progress'] = False # True #
import pandas as pd

np.random.seed(seed=10)

from Visualization import visualize
from plotly.offline import plot 

# In[]

''' kernels '''

""" Linear kernel """
class linear():
    """ Takes two vectors x1 and x2 and calculates dot pro """
    def __init__(self):
        pass

    def __repr__(self):
        return "Linear"

    def __call__(self,x1, x2):
        if len(x2):
            return np.einsum('ijk,nk->in',x1,x2)
        else:# for gram matrix calculation
            return x1.dot(x1.T)
        
""" Polynomial kernel"""
class polynomial():
    '''
    p: the polynomial degree
    
    '''
    def __init__(self,p=None):
        self.p = p

    def __repr__(self):
        return "Poly"

    def __call__(self,x1, x2):
        # Implementation of polynomial equation (x.Ty+c)^p
        return (1 + np.dot(x1, x2)) ** self.p

''' RBF Kernel '''
class rbf():
    '''
    Radial Basis Function
    
    gamma: equation parameter defined by 1/2*sigma^2
    
    '''
    def __init__(self,gamma=0.01):
        self.gamma = gamma

    def __repr__(self):
        return "RBF"

    def __call__(self,u,v):

        if len(v): # if v vector is present
            w = v - u
            # implementation of vector dot product using Einsten summation approach
            return np.exp(-self.gamma * np.einsum("ijk,ijk->ij",w,w)) 
        else: # for gram matrix calculation
            X_norm = np.sum(u ** 2, axis = -1)
            K = ne.evaluate('exp(-g * (A + B - 2 * C))', {
                    'A' : X_norm[:,None],
                    'B' : X_norm[None,:],
                    'C' : np.einsum('ij,kj->ik',u,u),
                    'g' : self.gamma,
            })
            return K

''' Sigmoid kernel: not used in the model '''
class sigmoid():
    def __init__(self,gamma, c):
        self.gamma = gamma
        self.c = c

    def __repr__(self):
        return "Sigmoid"

    def __call__(self,x1, x2):
        return math.tanh(self.gamma * np.dot(x1, x2) + self.c)


def gram_matrix(kernel,x):
    return kernel(x,[]) # here we pre-computed gram matrix without feeding into kernel. 

# In[]
class Multi_Kernel():
    '''
    This class method performs the calculation of multiple kernels and a linear combination of such kernels with a weight parameter beta_f
    '''
    def __init__(self,X = None,kernels=None,gamma=None,p=None,beta_f=None,gram_Matrix =None):
        '''
        

        Parameters
        ----------
        X : float, optional
            Data/Sample matrix. The default is None.
        kernels : string, optional
            kernel types. The default is None.
        gamma : float, optional
            gamma value for rbf kernel. The default is None.
        p : integer, optional
            Polyomial kernel degree. The default is None.
        beta_f : beta_f : float vector, optional
            Kernel weight parameter for each kernel. The default is None.
        gram_Matrix : float, optional
            precomputed kernel matrix received from another SVM model or from a central server for a FL update. The default is None.

        Returns
        -------
        linear combination of such kernels with a weight parameter beta_f

        '''
        self.kernels = []#list( map(kernel_list.get,kernels) )
        for i,kernel in enumerate(kernels):
            if kernel=='rbf':
                self.kernels.append(rbf(gamma=gamma[i]))
            elif kernel=='linear':
                self.kernels.append(linear())
            elif kernel=='polynomial':
                self.kernels.append(polynomial(p[i]))
            elif kernel=='sigmoid':
                self.kernels.append(sigmoid(gamma[i],p[i]))
        self.X = X

        ''' Currently beta is pre-assigned or assigned equal weights based on number of kernels.
        We need to write a seperate optimization algorithm and integrate with MK-SVM to update and tune beta'''
        if beta_f is None:
            self.beta_f = (1.0 / len(kernels)) * np.ones(len(kernels)) # equal importance for each kernel
        else:
            self.beta_f = beta_f 

        if X is not None: # Calculate the Kernel/Gram matrix for each feature group samples of size Nxm1
            self.grams = np.array([gram_matrix(kernel, X[i]) for i,kernel in enumerate(self.kernels)])
        else:
            self.grams = gram_Matrix; # Use gram matrix from FL or TL

    def __call__(self):
        ''' generate linear combination of all the kernels '''
        return np.tensordot(self.beta_f, self.grams, axes=(0,0))

# In[]
class MK_SVM(object):

    def __init__(self, kernel="linear", C=None,gamma=None,p=None,max_iter = None,c=None,
                 beta_f = None,kernel_grams=None,model_aux=None):

        '''

        Parameters
        ----------
        kernel : string, optional
            Kernel type: 'rbf','linear',and 'polynomial', The default is "linear".
        C : float, optional
            Regularization parameter. The default is None.
        gamma : float, optional
            The gamma parameter for RBF kernel. The default is None.
        p : integer, optional
            Polyomial kernel degree. The default is None.
        max_iter : integer, optional
            Number of iteration for optimization. The default is None.
        beta_f : float vector, optional
            Kernel weight parameter for each kernel. The default is None.
        kernel_grams : float, optional
            precomputed kernel matrix received from another SVM model or from a central server for a FL update. The default is None.
        model_aux : complex, optional
            A transferred model or a pretrained model. The model can be SVM, NN, Random Forest etc.. The auxilary model for transfer learning

        '''

        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.p = p
        self.max_iter = max_iter
        self.c = c
        self.beta_f = beta_f
        self.kernel_grams = kernel_grams
        self.fa = model_aux


        ''' Final estimated parameters '''
        self.dual_coef = [] # Dual coefficients of the support vector in the decision function (see Mathematical formulation), multiplied by their targets. For multiclass, coefficient for all 1-vs-1 classifiers. The layout of the coefficients in the multiclass case is somewhat non-trivial. See the multi-class section of the User Guide for details.
        self.support_vectors_=[] # support vector
        self.support_vectors_y=[] # class label of the support vector (SV)
        self.n_support_=[] # number of SV
        self.class_seq=[] # sequence of class labels in multi-class classification
        self.b = [] # intercept in each binary classification
        self.w = [] # weight vector (if only linear classification is used)
        self.ind = [] # indexes of SVs

        if self.C is not None: self.C = float(self.C)

    ''' strip m, n for each node of classification for multiclass classification '''
    def strip_m_n(self,X,Y,m,n):
        '''
        Get data for a binary classification by selecting two class labels m and n out of M labels.
        
        class label n will be labelled as -1 and class label m is labelled as +1
        
        Parameters
        ----------
        X : float
            Data matrix.
        Y : Integer
            Class labels .
        m : Integer
            Class label m.
        n : Integer
            Class label n.

        Returns
        -------
        X_bc : float 
            data matrix for class label m and n.
        Y_bc : TYPE
            Class label with -1 for label n and +1 for label m.

        '''
        m_idx = np.where(Y==m)
        n_idx = np.where(Y==n)
        X_bc = [x[np.append(m_idx,n_idx)] for x in X]
        Y_bc = Y[np.append(m_idx,n_idx)]
        Y_bc[Y_bc==n]= -1
        Y_bc[Y_bc==m]= 1

        return X_bc,Y_bc


    ''' Optimization using quadratic progamming '''
    def optimize(self,K,y,N,Tau=0):
        '''
        Quadratic optimization algorithm
        
        Also implements transfer learning by incorporating prediction f_s(x) from a pretrained model
        
        f(x) =  f_s(x)+Delta_f(x) 
        
        where f(x) is the final prediction as a combination of predictionf from the pretrained model and the target /current model Delta_f(x)
        
        Parameters
        ----------
        K : float
            Kernel matrix. If it is a multi-kernel SVM, then its a weighted linear combination of multiple kernels
        y : Integer
            Class label.
        N : Integer
            Number of samples/ #rows in a kernel matrix.
        Tau : float
            prediction result f_s(x) from the source/pretrained model.

        Returns
        -------
        TYPE
            returns optimizer results, alpha values used in final prediction algorithm 

        '''
        if self.max_iter:

            cvxopt.solvers.options['maxiters']=self.max_iter

        ''' quadratic programming '''
        P = cvxopt.matrix(np.outer(y,y) * K)
        q = cvxopt.matrix(Tau-np.ones(N)) # -(1-Tau)
        A = cvxopt.matrix(y.astype('float'), (1,N))
        b = cvxopt.matrix(0.0)

        G = spmatrix(N*[-1.0]+N*[1.0], range(2*N), 2*list(range(N)))
        h = matrix(N*[0.0] + N*[self.C])

        return cvxopt.solvers.qp(P, q, G, h, A, b,kktsolver="chol2")

    ''' tune and get SVM parameters '''
    def tune_params(self,X,y):
        '''
        

        Parameters
        ----------
        X : float
            Data matrix.
        Y : Integer
            Class labels .

        Returns
        -------
        ind : Integer
            indexes of the support vectors in the data matrix.
        a : float
            tuned lagrange multipliers also termed as alpha as per equations of SVM
        sv : float vector
            Support vectors.
        sv_y : Integer
            Class label associated with the support vector.
        n_sv : Integer
            Number of support vectors.
        b : float
            Intercept of the decision function. Available only when linear kernel is used
        w : float
            weights associated with linear decision function. Available only when linear kernel is used

        '''
        if self.kernel_grams is None: # if the precomputed kernel/gram matrix is not available

            self.Ks = Multi_Kernel(X=X,kernels=self.kernel,gamma=self.gamma,p=self.p,beta_f=self.beta_f) # Calculate kernel matrix from the data

        else: # if the precomputed kernel/gram matrix is available from another SVM model or from a central server, incorporate in the main SVM 
            self.Ks = Multi_Kernel(X=None,kernels=self.kernel,gamma=self.gamma,p=self.p,beta_f=self.beta_f,gram_Matrix = self.kernel_grams)

        K = self.Ks() # get kernel matrix as the linear combination of multiple kernels
        n_samples, n_features = np.hstack(X).shape
        self.beta_f = self.Ks.beta_f # get weight vector which combines all the kernels

        """ check auxilary model """
        Tau = 0 # initialize the output of the auxilary/pretrained model for TL is zero
        if self.fa is not None: # if a pretrained/auxilary model is available then calculate its projecttion/prediction on the given sample/samples
            Tau = y*self.fa.predict(X,return_projection=True)


        # solve QP problem
        solution = self.optimize(K, y, n_samples,Tau)
        
        print(solution['primal objective'])

        # Lagrange multipliers
        a = np.ravel(solution['x'])

        # Support vectors have non zero lagrange multipliers
        threshold = a > 1e-5
        ind = np.arange(len(a))[threshold]
        a = a[threshold]
        sv = [Xf[threshold,:] for Xf in X] #X[threshold]
        sv_y = y[threshold]
        n_sv = np.unique(sv_y, return_counts=True)[1]

        # Intercept
        b = np.mean(sv_y - np.sum(a * sv_y * K[ind][:,ind], axis=1))

        # Weight vector
        if self.kernel == 'linear':
            w = np.zeros(n_features)
            for n in range(len(a)):
                w += a[n] * sv_y[n] * sv[n]
        else:
            w = None

        return ind,a,sv,sv_y,n_sv,b,w

    def fit(self, X, y):
        '''
        For the given input data, X with true labels, y fit/train the SVM model. 
        It takes care of multi-class function by performing 1 v/s 1 approach

        Parameters
        ----------
        X : float
            Data matrix.
        y : Integer
            Class labels .

        Returns
        -------
        None

        '''

        if len(np.unique(y))>1:
            self.labels = np.unique(y)

            ''' Directed Acyclic Graph (DAG) implementation for multi-class SVM '''
            # Initialization: votes are initially all 1s, each iteration we take out one guess by marking it zero

            #  Train classifier_m_n iteratively to update predictions vector
            for depth in np.arange(1,len(self.labels)):# tree depth, depth i has i nodes
                for m in np.arange(0 , depth):  #% ASSUMING m < n
                    n = m + (len(self.labels) - depth); #% At m_n node, we train, m_n classifier
                    X_train, y_train = self.strip_m_n(X,y,m,n); # function to get labels
                    ''' tune SVM parameters '''
                    idx,a,sv,sv_y,n_sv,b_temp,w_temp = self.tune_params(X_train, y_train)

               #Optimized parameters once svm model is trained
                    self.dual_coef.append(a)
                    self.support_vectors_.append(sv)
                    self.support_vectors_y.append(sv_y)
                    self.n_support_.append(n_sv.sum())
                    self.class_seq.append(str(m)+str(n))
                    self.b.append(b_temp)
                    self.w.append(w_temp)
                    self.ind.append(idx)

        self.kernel_grams = self.Ks.grams


    def project_class(self, X=[],feat_set=None,classifier=0):
        '''        
        Parameters
        ----------
        X : float
            Data matrix. The default is [].
        feat_set : Integer, optional
            The prediction will be on the specified feature groups or on a single group. The default is None.
        classifier : TYPE, optional
            DESCRIPTION. The default is 0.

        Returns
        -------
        float
            project f(x) as a combination of auxilary/pretrained model prediction plust the current model.

        '''

        K = self.Ks
        a = self.dual_coef[classifier]
        sv = self.support_vectors_[classifier]
        sv_y = self.support_vectors_y[classifier]
        b = self.b[classifier]

        ''' Projection from auxilary/pretrained model '''
        yh = 0
        if self.fa:
            yh = self.fa.predict(X,return_projection=True)
        ''' return final projection '''
        return yh + np.dot(a*sv_y,\
                       np.array([K.beta_f[j]*K.kernels[j](X[j][:,None],sv[j])\
                                 for j  in feat_set]).sum(axis=0).T) + b

    def predict(self, X,feat_set=None,return_projection=False):
        '''
        Function performs prediction of the given data to a class, by calling projection function and mapping 
        it to the respective class label for a multiclass classification
        
        Parameters
        ----------
        X : float
            Data matrix or a single data vector.
        feat_set : Integer, optional
            Number. The default is None.
        return_projection : Boolean, optional
            Returns just projecttion, f(x) for a regression model if true 
            else returns the class label for classficiation. The default is False.

        Returns
        -------
        Integer (for classification)/ float (for regression)
            Returns class labels for a classification problem else projection value f(x) for regression.

        '''

        if type(X)!=list:
           N,M = X.shape
           X  = [X]
           if feat_set is None:
               feat_set = [0]
        elif type(X)==list:
           N,M = X[0].shape
           if feat_set is None:
               feat_set = np.arange(len(X))

        votes = np.ones((N,len(self.labels)))

        # votes are initially all 1s, each iteration we take out one guess by marking it zero
        # Train classifier_m_n iteratively to update predictions vector
        for idx,seq in enumerate(self.class_seq):# seq='mn', m corresponds to label 1, and n corresponds to label -1

          projections = self.project_class(X,feat_set=feat_set,classifier=idx)

          if return_projection: # needed for adaptive learning
              return projections

          m_class = projections>0# np.sign(projections)
          votes[m_class,int(seq[1])] = 0; # this corresponds to keep class label n (seq[1]=n) as 1 as projections are less than 0
          votes[~m_class,int(seq[0])] = 0; # this corresponds to keep class label m(seq[0]) as 1 as projections are more than 0


        return np.argmax(votes,1)

# In[]
if __name__ == "__main__":

    from sklearn.metrics import mean_squared_error, make_scorer
    from sklearn.model_selection import GridSearchCV    
    import matplotlib.pyplot as plt

    """ Run test """
    from sklearn import datasets

    #Load dataset
    from sklearn.datasets import load_iris
    from sklearn import datasets
    from seaborn import heatmap
    iris = load_iris()
    # X, y = iris.data, iris.target
    # X = pd.DataFrame(X).apply(lambda x: (x-np.mean(x))/np.std(x)).to_numpy()


    ''' auxilary model '''
    #Load dataset
    cancer = datasets.load_breast_cancer()
    X,y = np.array(cancer['data'][:200,0:4]), np.array(cancer['target'])[:200]# np.expand_dims(np.array(cancer['target']),axis=1))

    X = pd.DataFrame(X).apply(lambda x: (x-np.mean(x))/np.std(x)).to_numpy()


    N, d = X.shape

    train_N = int(0.8 * N)

    # split the data into trainint and testing sets
    sel_idx = np.random.choice(np.arange(N), train_N, replace=False)
    selection = np.full((N,), False, dtype=bool)
    selection[sel_idx] = True

# Training Data Sets
    X_train = X[selection,:]
    y_train = y[selection]
    X_train1 = [X_train[:,[0,3]],X_train[:,[1,2]],X_train[:,[0,1,2]]]   #<------------------- Adjust Features

# Test Data Sets
    X_test = X[np.invert(selection),:]
    y_test = y[np.invert(selection)]
    X_test1 = [X_test[:,[0,3]],X_test[:,[1,2]],X_test[:,[0,1,2]]]       # <------------------ Adjust Features

    dm = 25  # Dimension for parameters
# MSE function for optmizing 
    '''
    def MSE(y_true, y_pred):
        mse = mean_squared_error(y_true,y_pred)
        return mse
    mse = make_scorer(MSE, greater_is_better= False)

# Paramaters to be optimized
    C_ps = np.logspace(-2, 2, dm)
    g_ps = np.logspace(-4, 1, dm)
   
    mses = []
    params =[]
    
#Grid Search for mk_svm
    for i, C_p in enumerate(C_ps):
        for j, g_p in enumerate(g_ps):
            clf1= MK_SVM(C=C_p, gamma=[g_p,g_p,g_p],kernel=['rbf','rbf','rbf'],max_iter=100,beta_f=[0.33,0.33,0.33])
            clf1.fit(X_train1, y_train)
            y_predict = clf1.predict(X_test1,feat_set=[0,1,2])
            mse_svm = MSE(y_test, y_predict)
            mses.append(mse_svm)    
            params.append([C_p, g_p])
                
    mses = np.array(mses)   
    min_mse = np.min(mses[np.nonzero(mses)]) 
    param_a = np.array(params)
    grid = np.reshape(mses,(dm,dm))
    mse_a = np.array(mses)
    df_mse = pd.DataFrame(grid,index=C_ps,columns=g_ps)
    opt_param = df_mse.stack().idxmin()
    # heatmap(df_mse)
    # plt.show()    
    '''
    g_op = 0.3480700588428413 #opt_param[1]  #Optimized gamma
    C_op = 0.06812920690579612 #opt_param[0]  #Optimized C
    
    # Retrain model with optimized parameters
    clf_op = MK_SVM(C=C_op, gamma=[g_op,g_op,g_op],kernel=['rbf','rbf','rbf'],max_iter=100,beta_f=[0.33,0.33,0.33])


    clf_op.fit(X_train1, y_train)
    y_predict = clf_op.predict(X_train1,feat_set=[0,1,2])#cProfile.run("clf.predict(X_train1)")
     
    correct = np.sum(y_predict == y_train)
    print("%d out of %d predictions correct" % (correct, len(y_predict)))
    
    # y_predict = clf.predict(X_test,feat_set=0)
    y_predict = clf_op.predict(X_test1,feat_set=[0,2,1])


    correct = np.sum(y_predict == y_test)
    print(f"Number of support vectors {clf_op.n_support_[0]}")
    print("%d out of %d predictions correct" % (correct, len(y_predict)))
    print(f'Accuracy: {100*correct/ len(y_predict)}')


    # feat_set=2
    # y_predict = clf1.predict(X_test1[feat_set],feat_set=feat_set)


    # correct = np.sum(y_predict == y_test)
    # print("2. %d out of %d predictions correct" % (correct, len(y_predict)))
    # print(f'2. Accuracy: {100*correct/ len(y_predict)}')


#%%

    ######################################################################
    ''' Target Model '''
    X,y = np.array(cancer['data'][200:700,0:4]), np.array(cancer['target'])[200:700]

    X = pd.DataFrame(X).apply(lambda x: (x-np.mean(x))/np.std(x)).to_numpy()


    N, d = X.shape

    train_N = int(0.1 * N)

    # split the data into trainint and testing sets
    sel_idx = np.random.choice(np.arange(N), train_N, replace=False)
    selection = np.full((N,), False, dtype=bool)
    selection[sel_idx] = True

    X_train = X[selection,:]
    y_train = y[selection]
    X_train1 = [X_train[:,[0,3]],X_train[:,[1,2]],X_train[:,[0,1,2]]]


    X_test = X[np.invert(selection),:]
    y_test = y[np.invert(selection)]
    X_test1 = [X_test[:,[0,3]],X_test[:,[1,2]],X_test[:,[0,1,2]]]


    clf = MK_SVM(C=0.0001,gamma=[0.05,0.05,0.05],kernel=['rbf','rbf','rbf'],
                 max_iter=1,beta_f=[0.33,0.33,0.33],model_aux=clf_op) #MK_SVM(C=10.1,kernel=['rbf','rbf'],max_iter=100)


    # clf.fit([X_train], y_train)
    clf.fit(X_train1, y_train)
    y_predict = clf.predict(X_train1,feat_set=[0,2,1])#cProfile.run("clf.predict(X_train1)")
    correct = np.sum(y_predict == y_train)
    print("%d out of %d predictions correct" % (correct, len(y_predict)))

    # y_predict = clf.predict(X_test,feat_set=0)
    y_predict = clf.predict(X_test1,feat_set=[0,2,1])


    correct = np.sum(y_predict == y_test)
    print(f"Number of support vectors {clf.n_support_[0]}")
    print("%d out of %d predictions correct" % (correct, len(y_predict)))
    print(f'Accuracy: {100*correct/ len(y_predict)}')

    print("\nUsing SKLEARN")
    from sklearn.svm import SVC
    clf2 = SVC(C=100,gamma=0.01,kernel='rbf',tol=1e-5)

    clf2.fit(X_train,y_train)
    y_predict = clf2.predict(X_train)
    correct = np.sum(y_predict == y_train)
    print("%d out of %d predictions correct" % (correct, len(y_predict)))

    preds = clf2.predict(X_test)

    correct = np.sum(preds == y_test)
    print(f"Number of support vectors {clf2.n_support_}")
    print("%d out of %d predictions correct" % (correct, len(y_test)))
    print(f'Accuracy: {100*correct/ len(y_test)}')

# In[] # Transfer Learning Implementation
    iris = load_iris()
    X, y = iris.data, iris.target
    X = pd.DataFrame(X).apply(lambda x: (x-np.mean(x))/np.std(x)).to_numpy()
    N, d = X.shape

    train_N = int(0.8 * N)

    # split the data into trainint and testing sets
    sel_idx = np.random.choice(np.arange(N), train_N, replace=False)
    selection = np.full((N,), False, dtype=bool)
    selection[sel_idx] = True

    X_train = X[selection,:]
    y_train = y[selection]
    X_train1 = [X_train[:,[0,3]],X_train[:,[1,2]],X_train[:,[0,1,2]]]


    X_test = X[np.invert(selection),:]
    y_test = y[np.invert(selection)]
    X_test1 = [X_test[:,[0,3]],X_test[:,[1,2]],X_test[:,[0,1,2]]]

    clf = MK_SVM(C=0.0001,gamma=[0.05],kernel=['rbf'],
                     max_iter=1,beta_f=[0.33]) #MK_SVM(C=10.1,kernel=['rbf','rbf'],max_iter=100)
    # clf.fit([X_train], y_train)

    clf.fit([X_train1[0]], y_train)

    projections = clf.predict(X_train1[0])

    fig = visualize(colors=['darkred','darkblue','lime']).plotly_decision_boundary(model=clf,X=X_train1[0],
                                               y= y_train,X_test=None,y_test=None,Categories=[0,1,2])
    plot(fig,filename=f'boundary.html')
    
    # fig.write_image(f'../SVM_Bound.png',height=500, width=1550,engine='orca')
    # fig.show()

