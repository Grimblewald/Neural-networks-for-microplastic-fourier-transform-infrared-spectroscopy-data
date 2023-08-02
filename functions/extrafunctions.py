# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 15:47:07 2023

@author: fritz
"""

import re
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K 

# Sparse matrix libraries
from scipy.sparse import csc_matrix, eye, diags # for creating sparse matrices
from scipy.sparse.linalg import spsolve # for solving sparse linear systems

def get_file_extension(file_path):
    pattern = r"\.([^.\\/:*?\"<>|\r\n]+)$"
    match = re.search(pattern, file_path)
    if match:
        return match.group(1)
    else:
        return None

#helper
def funcyDics(params,func):
    return func(**params)

#loss

def categoricalFocalLoss(name='categorical_focal_loss', gamma=2.0):
    """
    Focal loss for multi-classification problem.
    
    This might look gross and likely needs fixing, but loss functions passed
    to keras should only have 2 inputs, true and predicted, but i want to set
    up my loss function with a variable gamma. So this function is meant to 
    make that happen?
    
    :param gamma: float, the focusing parameter gamma.
    :param alpha: float, the class balance parameter alpha.
    :return: A loss function object that can be used with TensorFlow model.
    """
    
    def focalLoss(y_true, y_pred):
        """
        Compute the focal loss given the ground truth labels (y_true) and predicted labels (y_pred).
        
        :param y_true: tensor of true labels.
        :param y_pred: tensor of predicted labels.
        :return: scalar tensor representing the focal loss value.
        """
        
        # Clip predictions to prevent log(0)
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        
        # Compute cross entropy
        cross_entropy = -float(y_true) * K.log(y_pred)
        
        # Calculate the class weights required to balance the output
        #this is the equivalent to alpha
        ClassRepresentation = tf.math.reduce_sum(y_true, axis=0, keepdims=True)
        
        alpha = tf.math.divide_no_nan(tf.cast(tf.math.reduce_max(ClassRepresentation),dtype=tf.float32), 
                                      tf.cast(ClassRepresentation, dtype=tf.float32)) 
        
        # Compute weights based on class balance and gamma
        weights = alpha * tf.pow(1 - y_pred, gamma)
        
        # Compute final categorical focal loss
        loss = K.sum(weights * cross_entropy, axis=-1)
        
        return loss
    
    return focalLoss

# Smoothing baselines

# Define a function called 'WhittakerSmooth' that takes in an array x, a weight array w, a smoothing parameter lambda_, and an optional parameter called differences which defaults to 1. 
def whittakerSmooth(x,w,lambda_,differences=1):

    # Convert the input array x into a matrix
    X=np.matrix(x)

    # Get the size of the matrix
    m=X.size

    # Create an identity matrix with the same size as X
    E=eye(m,format='csc')

    # Apply the difference operator to the identity matrix 'differences' number of times
    for i in range(differences):
        E=E[1:]-E[:-1] 

    # Create a diagonal matrix with the weight array w as the diagonal
    W=diags(w,0,shape=(m,m))

    # Create a sparse matrix A by adding the product of the transpose of E and E multiplied by lambda_ to W
    A=csc_matrix(W+(lambda_*E.T*E))

    # Create a sparse matrix B by multiplying the transpose of X by W
    B=csc_matrix(W*X.T)

    # Solve the linear system Ax = B for x and store the result in the variable 'background'
    background=spsolve(A,B)

    # Convert the result to a numpy array and return it
    return np.array(background)

# Define a function called 'airPLS' that takes in an array x, a smoothing parameter lambda_, an integer porder, and an integer itermax

def airPLS(x, lambda_=1, porder=1, itermax=15):

    # Get the size of the input array x
    m=x.shape[0]

    # Create an array of ones with the same size as x
    w=np.ones(m)

    # Iterate 'itermax' number of times
    for i in range(1,itermax+1):
        
        # Smooth the input array x using the WhittakerSmooth function with the weight array w, the smoothing parameter lambda_, and the polynomial order porder
        z=WhittakerSmooth(x,w,lambda_, porder)
        
        # Subtract the smoothed array z from the input array x to get the difference array d
        d=x-z
        
        # Calculate the sum of the negative values in the difference array d
        dssn=np.abs(d[d<0].sum())
        
        # If the sum of the negative values in d is less than 0.001 times the sum of the absolute values of x or if the maximum number of iterations has been reached, break out of the loop
        if(dssn<0.001*(abs(x)).sum() or i==itermax):
            if(i==itermax): print('WARING max iteration reached!')
            break
        
        # Set the weight array w to 0 for all elements in d that are greater than or equal to 0
        w[d>=0]=0 
        
        # Set the weight array w to exp(i*|d|/dssn) for all elements in d that are less than 0
        w[d<0]=np.exp(i*np.abs(d[d<0])/dssn)
        
        # Set the first element of the weight array w to exp(i*max(d)/dssn) for all elements in d that are less than 0
        w[0]=np.exp(i*(d[d<0]).max()/dssn) 
        
        # Set the last element of the weight array w to the same value as the first element
        w[-1]=w[0]

    # Return the smoothed array z
    return z

# resizing things

def averagePooling(data, 
                    pool_size, pool_window_size, 
                    pooling_strides, pooling_padding):
    
    # Get the shape of the input data
    shapes = np.asarray(data).shape
    # Reshape the data to have a third dimension of size 1
    data = tf.reshape(data, 
                    [shapes[0], 
                    shapes[1],
                    1]
                    )
    
    poollayer = tf.keras.layers.AveragePooling1D(pool_size=pool_window_size,
                                                strides=pooling_strides,
                                                padding=pooling_padding)
    data = poollayer(data)
    
    data = tf.reshape(data,shape = (data.shape[0],data.shape[1]))
    
    return data

def rescale(data,axis,resize_size):
    
    # Get the shape of the input data
    shapes = np.asarray(data).shape
    # Reshape the data to have a third dimension of size 1
    data = tf.reshape(data, 
                    [shapes[0], 
                    shapes[1],
                    1]
                    )
    
    resizer = lambda x, newsize : tf.reshape(tf.image.resize(tf.reshape(x,(len(x),1,1)), [newsize,1]),(newsize,))
    
    data = np.apply_along_axis(resizer, 
                            1, 
                            data, 
                            resize_size)
    data = tf.reshape(data,shape = (data.shape[0],data.shape[1]))
    
    return data
    
def preprocessing(data,
               resize,
               pooling,
               resize_size,
               pool_window_size,
               pooling_strides,
               pooling_padding):
    '''
    Preprocesses the input data by reshaping, resizing, and/or pooling.
    :param data: input data
    :param resize: boolean indicating whether to resize the data
    :param pooling: boolean indicating whether to pool the data
    :param resize_size: size to resize the data to
    :param pool_window_size: size of the pooling window
    :param pooling_strides: size of the pooling strides
    :param pooling_padding: type of padding to use for pooling
    :return: preprocessed data
    '''
    
    # If resize is True, resize the data using the specified size
    if resize:
        data = rescale(data, 1, resize_size)


    # If pooling is True, apply average pooling to the data using the specified window size, strides, and padding
    if pooling:
        data = average_pooling(data, 
                               pool_size=pool_window_size,
                               pooling_strides=pooling_strides,
                               pool_window_size = pool_window_size,
                               pooling_padding=pooling_padding)

    return data

