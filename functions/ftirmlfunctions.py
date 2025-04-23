# Core libraries
import numpy as np # for numerical operations
import pandas as pd # for data manipulation and analysis
import tensorflow as tf # for building and training machine learning models
from tensorflow import keras
from keras import backend as K 
import re # for regular expressions
import glob # for finding all pathnames matching a specified pattern
import os # for doing file system stuff

# Graphing libraries
import matplotlib # for creating visualizations
import matplotlib.pyplot as plt # for creating plots
import seaborn as sns # for creating statistical graphics

# Extra data manipulation libraries
import sklearn # for machine learning algorithms and tools
from sklearn.metrics import confusion_matrix # for calculating confusion matrix
from sklearn.cluster import MiniBatchKMeans # for clustering data
kmeansestimator = MiniBatchKMeans(batch_size=2048, n_clusters = 3) # creating an instance of MiniBatchKMeans

from sklearn.model_selection import train_test_split # for splitting data into training and testing sets
from imblearn.over_sampling import RandomOverSampler, KMeansSMOTE, SVMSMOTE, ADASYN # for oversampling imbalanced data
 
# Sparse matrix libraries
from scipy.sparse import csc_matrix, eye, diags # for creating sparse matrices
from scipy.sparse.linalg import spsolve # for solving sparse linear systems

from tensorflow.keras.saving import register_keras_serializable

def funcy_dics(params,func):
    return func(**params)

@register_keras_serializable()
def categorical_focal_loss(name='categorical_focal_loss', gamma=2.0):
    """
    Focal loss for multi-classification problem.
    
    :param gamma: float, the focusing parameter gamma.
    :param alpha: float, the class balance parameter alpha.
    :return: A loss function object that can be used with TensorFlow model.
    """
    
    def focal_loss(y_true, y_pred, name="focal loss"):
        """
        Compute the focal loss given the ground truth labels (y_true) and predicted labels (y_pred).
        
        :param y_true: tensor of true labels.
        :param y_pred: tensor of predicted labels.
        :return: scalar tensor representing the focal loss value.
        """
        
        # Clip predictions to prevent log(0)
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        
        # Compute cross entropy that doesn't look at errors
        cross_entropy = -float(y_true) * K.log(y_pred)
        #cross_entropy = K.categorical_crossentropy(y_true, y_pred)
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
    
    return focal_loss

# Define a function called 'WhittakerSmooth' that takes in an array x, a weight array w, a smoothing parameter lambda_, and an optional parameter called differences which defaults to 1. 
def WhittakerSmooth(x,w,lambda_,differences=1):

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


def MeanAndStdDevOfArrays(arrays):
    '''
    Calculates the mean and standard deviation for arrays of inconsistent lengths.
    :param arrays: list of arrays
    :return: tuple of mean and standard deviation
    '''
    # Get the length of each array in the list
    lens = [len(i) for i in arrays]
    # Create an empty masked array with the maximum length of the arrays in the list
    arr = np.ma.empty((np.max(lens),len(arrays)))
    # Set all values in the array to be masked
    arr.mask = True
    # Loop through each array in the list and add it to the masked array
    for idx, l in enumerate(arrays):
        arr[:len(l),idx] = l
    # Calculate the mean and standard deviation of the masked array
    return arr.mean(axis = -1), arr.std(axis=-1)

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

def average_pooling(data, 
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

def make_label(dataframes,choice1,choice2):
    '''
    This function takes a list of dataframes and two column names. It extracts the values 
    from the specified columns in each dataframe and returns them as two separate lists.

    Parameters
    ----------
    dataframes : list
        This should be a list of pandas dataframes.
    choice1 : str
        This should be the name of a column in the dataframes.
    choice2 : str
        This should be the name of a column in the dataframes.

    Returns
    -------
    coutput : list
        This list contains the values from the choice1 column in each dataframe.
    routput : list
        This list contains the values from the choice2 column in each dataframe.

    '''
    # Initialize empty lists to store the extracted values
    coutput = []
    routput = []

    # Iterate through each dataframe in the input list
    for i in dataframes:
        # Extract the values from the specified columns and add them to the output lists
        j = i[[choice1]]
        i = i[[choice2]]
        coutput = coutput+j.values.tolist()
        routput = routput+i.values.tolist()

    # Flatten the output lists
    coutput = [item for sublist in coutput for item in sublist]
    routput = [item for sublist in routput for item in sublist]

    # Return the flattened output lists
    return coutput, routput

# This function takes a string as input and checks if it can be converted to a float
# If it can be converted, it returns True, otherwise it returns False
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

# This function takes a dataframe as input and splits the numeric data from the non-numeric data
# The numeric data is returned in x_out and the non-numeric data is returned in y_out
def split_numeric(df):
    '''
    Splits the numeric data from the non-numeric data in a dataframe.

    Parameters:
    df (pandas.DataFrame): The input dataframe.

    Returns:
    x_out (pandas.DataFrame): The numeric data.
    y_out (pandas.DataFrame): The non-numeric data.
    '''
    x_out = df.apply(lambda x: pd.to_numeric(x, errors='coerce')).dropna(how='all')
    y_out = df[df.columns.difference(x_out.columns)]
    return x_out, y_out

def xymaker(dataframe):
    '''
    Separates the numeric and non-numeric data in a dataframe.

    Parameters:
    dataframe (pandas.DataFrame): The input dataframe.

    Returns:
    x (pandas.DataFrame): The data under colums with names castable to float.
    y (pandas.DataFrame): The data under columns with names not able to be cast to float.
    nums (list): The column names that were able to be cast to float.
    names (list): The column names that couldn't be cast to float.
    '''
    nums = []
    names = []
    for i in dataframe.columns:
        if is_number(i):
            nums.append(i)
        else:
            names.append(i)
    x = dataframe[nums]
    y = dataframe[names]
    return x, y, nums, names

def guess_spectrum_type(series):
    '''
    Guesses whether a given FTIR spectrum is in transmission or absorbance mode based on the median of the values.

    Parameters:
    series (pandas.Series): The FTIR spectrum to be analyzed.

    Returns:
    str: 'Transmission' if the spectrum is likely in transmission mode, 'Absorbance' if it is likely in absorbance mode.
    '''
    
    return 'Transmission' if np.median(series) > 50 else 'Absorbance'

def convert_to_absorbance(transmission_series):
    '''
    Converts a series of transmission values to absorbance values using the Beer-Lambert law.

    Parameters:
    transmission_series (pandas.Series): The series of transmission values to be converted. 
                                         Values are assumed to be percentages (i.e., a value of 100 represents 100% transmission).

    Returns:
    absorbance_series (pandas.Series): The converted series of absorbance values.
    '''
    
    absorbance_series = 2 - np.log10(transmission_series)
    
    return absorbance_series

def normaliser(row, offset, correctionmethod, guessmode=True):
    '''
    Normalizes a row of data by subtracting the minimum value, dividing by the maximum value, and subtracting an offset.

    Parameters:
    row (pandas.Series): The row of data to be normalized.
    offset (float): The offset to be subtracted from the normalized values.

    Returns:
    row (pandas.Series): The normalized row of data.
    '''
    
    if guessmode:
        mode = guess_spectrum_type(row)
        
    match correctionmethod:
        case 'flip':
            row = row-row.min()
            row = row/row.max()
            row = row - offset
            
            if mode == 'Transmission':
                row = row*(-1)+1
                
        case 'beer-lambert':
            if mode == 'Transmission':
                row = convert_to_absorbance(row)
            row = row-row.min()
            row = row/row.max()
            row = row - offset
        case 'none':
            row = row-row.min()
            row = row/row.max()
            row = row - offset
        case _:
            raise ValueError('method for homogenising aquisition modes not recognised. Is there a typo?')

    return row


def fetchfilelabel(string, prefix, postfix):
    '''
    This function retrieves a string between a prefix and postfix using regex.
    Please note this uses regex, which means things like the '\' character can
    cause a few issues, especially important to consider when manipulating
    windows paths etc.

    Parameters
    ----------
    string : str
        The string that we wish to process.
    prefix : str
        This is a string that is expected to appear BEFORE the text of interest.
    postfix : str
        This is a string that is expected to appear AFTER the text of interest.
        
    Returns
    -------
    stringy : str
        The string between the prefix and postfix that was defined.
    '''

    # This regex pattern describes everything after the prefix and before the postfix.
    # It returns all 0 or more characters between that are not a newline.
    pattern = r'(?<=' + re.escape(prefix) + r')[a-zA-Z0-9_.]*(?=' + re.escape(postfix) + r')'
    try:
        stringy = re.search(pattern, string).group(0)
    except:
        stringy = 'ERROR_'+string

    return stringy

def make_dataset(data, #input data
                label_column, #column to use as labels
                shiftby, #value to shift data by
                dataset_name,
                baselinefix,
                correctionmethod): #name of dataset

    #call xymaker function to extract X and Y data, as well as nums and names
    # this should give us wavenumbers from column names
    X, Y, nums, names = xymaker(data)

    #convert X to float, should now be able to work with wavenumbers
    X = X.astype(float)
    nums = [float(i) for i in nums]
    
    match baselinefix:
        case 'airPLS':
            #apply airPLS (commented out)
            X = np.apply_along_axis(airPLS, 1, X) 
        case 'wavenumbermult':
            #multiply X by wavenumber (commented out)
            X = X*np.asarray(nums).astype(float)
        case 'combo':
            X = X*np.asarray(nums).astype(float)
            X = np.apply_along_axis(airPLS, 1, X)
    
    #create a copy of X to use as features
    features = X.copy()

    #extract labels from Y using label_column
    labels = Y[label_column]

    #apply normaliser function to features using shiftby
    features = np.apply_along_axis(lambda x: normaliser(x,shiftby, correctionmethod), 
                                1, 
                                features)

    #create a pandas dataframe using features and labels
    df = pd.DataFrame(features,labels)

    #set column names to nums
    df.columns = nums

    #return the dataframe and dataset_name
    return (df,dataset_name,labels)

    
def process_datasets(model_parameters):
    # Initialize empty lists and variables
    dfs_list = []
    dfs_listnames = []
    trainable_dfs_list = []

    # Loop through data paths in model parameters
    for dataset in model_parameters['datasets']:
        
        datasetdictname = dataset
        pathtodata = model_parameters['datasets'][datasetdictname]['path']
        datatype = model_parameters['datasets'][datasetdictname]['type']
        dataformat = model_parameters['datasets'][datasetdictname]['format']
        datasetname = model_parameters['datasets'][datasetdictname]['name']
        datalabel_column = model_parameters['datasets'][datasetdictname]['label_column']
        datatrainable = model_parameters['datasets'][datasetdictname]['trainable']
        #drop data with unwanted labels
        drop_list = model_parameters['ignore_classes']
        
        # If data type is a file, read in the data and create a dataset
        knowntypes = ['file', 'folder']
        knownformats = ['.csv']
        
        match datatype:
            case 'file' if dataformat == '.csv':
                # clean labels, remove unwanted and relabel as required
                rawdata = pd.read_csv(pathtodata, low_memory=False)
                data_remove_unwanted = rawdata[~rawdata[datalabel_column].isin(drop_list)]
                unwanted_data = rawdata[rawdata[datalabel_column].isin(drop_list)]
                data_unifiedlabels = data_remove_unwanted.replace({datalabel_column:model_parameters['collect_terms']})
                #pre-processes data
                incoming_data = make_dataset(data=data_unifiedlabels,
                                            label_column=datalabel_column,
                                            shiftby=model_parameters['baseline_shift'],
                                            dataset_name=datasetname,
                                            baselinefix = model_parameters['baseline_fix'],
                                            correctionmethod = model_parameters['mode_correction'])
                dfs_list.append(incoming_data[0])
                dfs_listnames.append(datasetname)
                if datatrainable:
                    trainable_dfs_list.append(datasetname)
                #save data to dictionary entry
                model_parameters['datasets'][datasetdictname]['data'] = incoming_data[0]
                if not unwanted_data.empty:
                    model_parameters['datasets'][datasetdictname]['ignored_data'] = unwanted_data
                else:
                    model_parameters['datasets'][datasetdictname]['ignored_data']  = 'nothing was ignored'
                    
                Y = list(incoming_data[0].index)
                model_parameters['datasets'][datasetdictname]['distribution'] = [(i, list(Y).count(i)) for i in list(set(Y))]
                #plot pareto chart for this dataset
                if not os.path.exists(model_parameters['unique']+f'paretocharts/{datasetname}class_distribution.svg'):
                    plot_pareto_chart(Y, datasetname, cumsumpercent=True,
                                      savename = model_parameters['unique']+f'paretocharts/{datasetname}class_distribution.svg')
                else:
                    print(f"pareto plot of {datasetname} appears to already exist, skipping")
            # If data type is a folder, read in all files and create a dataset for each
            
            case 'folder' if dataformat == '.csv':
                print('no folder based processing in new version yet')
            case _:
                print('\n couldn\'t find matching processing case for dataset with name '+datasetname+'\n')                
                if dataformat not in knownformats:
                    print('problem might be the format "'+dataformat+'" was specified, but only the following types are known \n|'+'|'.join(knownformats)+'|')
                if datatype not in knowntypes:
                    print('problem might be the format "'+datatype+'" was specified, but only the following types are known \n|'+'|'.join(knowntypes)+'|')
                
                raise ValueError('specified file format or filetype is not recognised \n\nif it should be ok \ncheck for typos or missing "." on file format')
                
    return model_parameters, dfs_list, dfs_listnames, trainable_dfs_list

def determine_wave_numbers(dfs_list):
    maxwavenumber = 10000.0
    minwavenumber = 0.0
    for j in dfs_list:
        maxwavenumber = min(maxwavenumber, max(j.columns.astype(float)))
        minwavenumber = max(minwavenumber, min(j.columns.astype(float)))
    return maxwavenumber, minwavenumber

def crop_datasets(dfs_list, maxwavenumber, minwavenumber):
    croppeddf_list = []
    for k in dfs_list:
        dropabove = k.columns.astype(float) > maxwavenumber
        dropbelow = k.columns.astype(float) < minwavenumber
        droplist = np.logical_not(np.logical_or(dropabove, dropbelow))
        cropped_data = k.iloc[:, droplist].copy()
        croppeddf_list.append(cropped_data)
    return croppeddf_list

def preprocess_datasets(croppeddf_list, dfs_listnames, trainable_dfs_list, model_parameters):
    finaldf_lists = []
    finaldf_list_trainable = []
    for m in zip(croppeddf_list,dfs_listnames):
        X = preprocessing(m[0].values,
                        model_parameters['use resizing'],
                        model_parameters['use pooling'],
                        model_parameters['resize size'],
                        model_parameters['pooling - window size'],
                        model_parameters['pooling - strides'],
                        model_parameters['pooling - padding'])
        finaldf_lists.append(pd.DataFrame(X, m[0].index))
        if m[1] in trainable_dfs_list:
            finaldf_list_trainable.append(pd.DataFrame(X, m[0].index))

    finaldf = pd.concat(finaldf_lists)
    finaldf_dict = {name: df for name, df in zip(dfs_listnames, finaldf_lists)}
    finaldf.index = pd.CategoricalIndex(finaldf.index)
    finaldf.sort_index(level=0, inplace=True)

    return finaldf_lists, finaldf_list_trainable, finaldf, finaldf_dict

def concatenate_datasets(finaldf_lists, finaldf_list_trainable, dfs_listnames):
    finaldf = pd.concat(finaldf_list_trainable)
    finaldf_dict = {name: df for name, df in zip(dfs_listnames, finaldf_lists)}
    finaldf.index = pd.CategoricalIndex(finaldf.index)
    finaldf.sort_index(level=0, inplace=True)
    return finaldf, finaldf_dict

def convert_labels(finaldf, model_parameters):
    X = finaldf.values
    Y = np.asarray([str(i) for i in list(finaldf.index.values)])
    model_parameters.update({'dimension_in': X.shape[1],
                            'dimension_out': len(set(Y)),
                            'unique_labels': list(set(Y))})
    model_parameters.update({'label_dictionary': {model_parameters['unique_labels'][i]: i for i in range(len(model_parameters['unique_labels']))}})
    Y_onehot = tf.one_hot(np.vectorize(model_parameters['label_dictionary'].get)(Y), 
                          model_parameters['dimension_out'],
                          dtype=tf.float32)
    Y_onehot = np.asarray([tuple(i.numpy()) for i in Y_onehot])
    Y_onehotaslist = list(set([tuple(i) for i in Y_onehot]))
    YOHlistlist = [list(i) for i in list(Y_onehot)]
    
    model_parameters.update({'reverse_dict':{value:key for key,value in zip(model_parameters['label_dictionary'].keys(), model_parameters['label_dictionary'].values())}})
    
    onehotdict = {model_parameters['reverse_dict'][np.array(onehot).argmax()]:onehot for onehot in zip(Y_onehotaslist)}
    
    model_parameters.update({'label_onehot_dictionary': {model_parameters['unique_labels'][i]: Y_onehot[i] for i in range(len(model_parameters['unique_labels']))}})
    
    return X, Y_onehot, model_parameters

class DataSplitError(Exception):
    pass

def split_data(X, Y_onehot, model_parameters):
    try:
        X_learn, X_test, Y_learn, Y_test = train_test_split(X, Y_onehot,
                                                            test_size=model_parameters['data(%) for testing'],
                                                            stratify=Y_onehot)
    except:
        classes = Y_onehot.argmax(1)
        classes = [model_parameters["reverse_dict"][i] for i in Y_onehot.argmax(1)]
        classes, counts = np.unique(np.array(classes), return_counts=True)
        class_counts = {cl:cnt for cl,cnt in zip(classes, counts)}
        raise DataSplitError(f"you have such few examples of some classes, such that a holdout/train set cannot be created.\nDatasets were therefore not created.\nYour class counts are:\n{class_counts }")
    try:
        X_train, X_val, Y_train, Y_val = train_test_split(X_learn, Y_learn,
                                                            test_size=model_parameters['data(%) for testing'],
                                                            stratify=Y_learn)
    except:
        classes = Y_onehot.argmax(1)
        classes = [model_parameters["reverse_dict"][i] for i in Y_onehot.argmax(1)]
        classes, counts = np.unique(np.array(classes), return_counts=True)
        class_counts = {cl:cnt for cl,cnt in zip(classes, counts)}
        raise DataSplitError(f"you have such few examples of some classes, such that the validaiton/train set cannot be created.\nDatasets were therefore not created.\nYour class counts are:\n{class_counts }")
        
    return X_train, X_val, Y_train, Y_val, X_test, Y_test, model_parameters

def balance_data(X_train, Y_train, model_parameters):
    if model_parameters['balance_via_oversample']:
        oversamplers = {'RandomOverSampler': RandomOverSampler(sampling_strategy='not majority'),
                        'KMeansSMOTE': KMeansSMOTE(sampling_strategy='not majority',
                                                cluster_balance_threshold='auto',
                                                n_jobs=None),
                        'SVMSMOTE': SVMSMOTE(sampling_strategy='minority',
                                            k_neighbors=3,
                                            m_neighbors=3),
                        'ADASYN': ADASYN()}
        X_resample, Y_resample = (X_train.copy(), Y_train.copy())
        for oversamplingmethod in model_parameters['oversampler']:
            oversampler = oversamplers[oversamplingmethod]
            X_resample, Y_resample = oversampler.fit_resample(X_resample, Y_resample)
            Y_resample = Y_resample.astype(np.float32)
        return X_resample, Y_resample, model_parameters
    else:
        return X_train, Y_train, model_parameters

def update_model_parameters(X_train, Y_train, X_val, Y_val, X_test, Y_test, finaldf, finaldf_dict, minwavenumber, maxwavenumber, model_parameters):
    model_parameters.update({'X_train': X_train,
                            'Y_train': Y_train,
                            'X_val': X_val, 'Y_val': Y_val,
                            'X_test': X_test,
                            'Y_test': Y_test})

    model_parameters.update({'data':finaldf,
                        'split_data': finaldf_dict})

    classcounts = []
    for i in set(list(finaldf.index)):
        classcounts.append((i,list(finaldf.index).count(i)))

    model_parameters.update({'minwavenumber':minwavenumber,
                        'maxwavenumber':maxwavenumber})

    return model_parameters

def create_datasets(model_parameters):
    # initialize things we will need
    dfs_list, croppeddf_list, finaldf_lists, dfs_listnames = [], [], [], []
    # get data ingested
    model_parameters, dfs_list, dfs_listnames, trainable_dfs_list = process_datasets(model_parameters)
    # get max and min wavenumbers accross datasets ingested
    maxwavenumber, minwavenumber = determine_wave_numbers(dfs_list)
    # crop all data to the broadest common wavenumber range
    croppeddf_list = crop_datasets(dfs_list, maxwavenumber, minwavenumber)
    # pre-process data using whatever was specified, baseline correction, minmax, etc.
    finaldf_lists, finaldf_list_trainable, finaldf, finaldf_dict = preprocess_datasets(croppeddf_list, 
                                                                                       dfs_listnames, 
                                                                                       trainable_dfs_list, 
                                                                                       model_parameters)
    # concatenate data into a single set    
    finaldf, finaldf_dict = concatenate_datasets(finaldf_lists, finaldf_list_trainable, dfs_listnames)
    # onehot encode labels
    X, Y_onehot, model_parameters = convert_labels(finaldf, model_parameters)
    # split data into trianing, validation, and testin (holdout)
    X_train, X_val, Y_train, Y_val, X_test, Y_test, model_parameters = split_data(X, Y_onehot, model_parameters)
    # supersample the training data to balance classes
    X_train, Y_train, model_parameters = balance_data(X_train, Y_train, model_parameters)
    # update model parameters with what we've done
    model_parameters = update_model_parameters(X_train, Y_train, 
                                               X_val, Y_val, 
                                               X_test, Y_test, 
                                               finaldf, finaldf_dict, 
                                               minwavenumber, maxwavenumber, 
                                               model_parameters)
    if not os.path.exists(model_parameters["unique"]+"all_data_pareto.svg"):
        # plot pareto for cumulative data
        plot_pareto_chart(labels = model_parameters["data"].index.to_numpy(), 
                          dataname = "All", 
                          cumsumpercent = True,
                          savename = model_parameters["unique"]+"all_data_pareto.svg", 
                          dpi=1600)
    else:
        print("pareto plot of all data appears to already exist, skipping")
        
    if not os.path.exists(model_parameters["unique"]+"paretocharts/training_data_pareto.svg"):
        # plot pareto of the oversampled data
        plotting_data = [model_parameters["reverse_dict"][i] for i in model_parameters["Y_train"].argmax(1)]
        plot_pareto_chart(labels = pd.DataFrame(plotting_data)[0], 
                          dataname = "Balance Training", 
                          cumsumpercent = True,
                          savename = model_parameters["unique"]+"paretocharts/training_data_pareto.svg", 
                          dpi=1600)
    else:
         print("pareto plot of oversampled data appears to already exist, skipping")
         
    return model_parameters

def prepare_data_for_prediction(path_to_csv, model_parameters, prediction_only=False):
    # Read the CSV file
    doflip = False
    print(f"reading file {path_to_csv}")
    rawdata = pd.read_csv(path_to_csv)
    newcols = rawdata['Wavenumber']
    df = rawdata.T
    df.columns = newcols
    df = df.drop("Wavenumber")
    features = df.values
    features = np.apply_along_axis(lambda x: normaliser(x,
                                                        model_parameters['baseline_shift'],
                                                        model_parameters['baseline_fix']), 
                                1, 
                                features)
    df = df.T
    df['Intensity'] = features[0]
    df = df.T
    if df.columns[0] < df.columns[1]:
        print('flipping')
        doflip = True
    df = crop_datasets([df],model_parameters['maxwavenumber'],model_parameters['minwavenumber'] )
    df = preprocessing(df[0],
                       model_parameters['use resizing'],
                       model_parameters['use pooling'],
                       model_parameters['resize size'],
                       model_parameters['pooling - window size'],
                       model_parameters['pooling - strides'],
                       model_parameters['pooling - padding'])
    if doflip:
        df = tf.reverse(df,axis=[1])
    return df

#%%
# This function counts the number of unique elements in a list of strings
def countunique(list_of_strings):
    # Loop through the set of unique elements in the list
    for i in set(list_of_strings):
        # Count the number of occurrences of the current element in the list
        count = list_of_strings.count(i)
        # Print the element and its count
        print(f'{i} occurances: {count}')

# This function performs dummy encoding on a dataframe
def dummyencoder (dataframe,dummies):
    # Reindex the columns of the dataframe with the columns of the dummies dataframe and fill missing values with 0
    dataframe.reindex(columns = dummies.columns, fill_value=0)
    # Return the modified dataframe
    return dataframe


# This function converts an array to a dataframe and relabels its columns
def labeller(array,names):
    # Convert the array to a dataframe
    output = pd.DataFrame(array)
    # Rename the columns of the dataframe with the names provided
    output.columns = names

    return output


#Define Plots

# This function plots the training loss
def plot_loss(history, label, n, colors):
    '''
    This function takes in the training history, label, index and colors and plots the training and validation loss.
    
    Parameters
    ----------
    history : object
        The training history object.
    label : str
        The label for the plot.
    n : int
        The index of the color to use.
    colors : list
        The list of colors to use for the plot.
      
    Returns
    -------
    None.
      
    '''
    # Use a log scale on y-axis to show the wide range of values.
    plt.semilogy(history.epoch, history.history['loss'],
                 color=colors[n], label='Train ' + label)
    plt.semilogy(history.epoch, history.history['val_loss'],
     color=colors[n], label='Val ' + label,
     linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
# This function plots the precision-recall curve
def plot_prc(name, labels, predictions, **kwargs):
    '''
    This function takes in the name, labels, predictions and additional keyword arguments and plots the precision-recall curve.
    
    Parameters
    ----------
    name : str
        The name of the plot.
    labels : array-like
        The true labels.
    predictions : array-like
        The predicted labels.
    **kwargs : dict
        Additional keyword arguments to pass to the plot function.
      
    Returns
    -------
    None.
      
    '''
    precision, recall, _ = sklearn.metrics.precision_recall_curve(labels, predictions)
    
    plt.plot(precision, recall, label=name, linewidth=2, **kwargs)
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.grid(True)
    ax = plt.gca()
    ax.set_aspect('equal')
  
#Training History
def plot_metrics(model_parameters, colors, dpi = 600, figure_size = (10,10),
                 save = False, savename = None, savepath = None):
    '''
    A function to plot the training history of a model.
    
    Parameters
    ----------
    model_parameters : dict
        A dictionary containing the model parameters and evaluation metrics.
    colors : list
        A list of colors to use for the plot.
    dpi : int, optional
        The resolution of the saved figure. The default is 600.
    figure_size : tuple, optional
        The size of the figure. The default is (10,10).
    save : bool, optional
        Whether to save the figure. The default is False.
    savename : str, optional
        The name to use when saving the figure. The default is None.
    savepath : str, optional
        The path to save the figure. The default is None.
    
    Returns
    -------
    plt : matplotlib.pyplot
        The plot object.
    
    '''
    
    # Extract the metrics from the model parameters
    metrics = []
    for metricname, metricmoniker in model_parameters['training_metrics']:
        metrics.append(metricmoniker)
    
    # Extract the history from the model evaluation
    history = model_parameters['model_eval']['model_history']
    
    # Save the old figure size and set the new figure size
    old_rcParams_figsize = matplotlib.rcParams['figure.figsize'].copy()
    matplotlib.rcParams['figure.figsize'] = figure_size
    
    # Plot each metric
    for n, metric in enumerate(metrics):
        # Format the metric name
        name = metric.replace('_',' ').capitalize()
        
        # Create a subplot for the metric
        plt.subplot(2,2,n+1)
        
        # Plot the training data
        plt.plot(history.epoch, 
                 history.history[metric], 
                 color=colors[0], 
                 label='Train')
        
        # Plot the validation data
        plt.plot(history.epoch, 
                 history.history['val_'+metric],
                 color=colors[0], 
                 linestyle='--', 
                 label='Val')
        
        # Set the axis labels and limits
        plt.xlabel('Epoch')
        plt.ylabel(name)
        plt.title(name)
        if metric == 'loss':
            plt.ylim([0, plt.ylim()[1]])
        elif metric == 'auc':
            plt.ylim([0,1.1])
        else:
            plt.ylim([0,1.1])
        
        # Add the legend
        plt.legend()
        
        # Save the figure if requested
        if save:
            plt.savefig(savepath+savename,dpi=dpi)
    
    # Reset the figure size to the old value
    matplotlib.rcParams['figure.figsize'] = old_rcParams_figsize
    del old_rcParams_figsize
    
    # Return the plot object
    return plt

#Confusion Matrix
#labels = Yw_d

#labels = test_labels
#predictions = test_predictions_resampled
#title = 'test test'
#dictlist = labelnames

# This function plots a circular connectivity plot
# It takes in predictions, labels, figure title, colormap, face color, node edge color, text color, and pred_is_cm as inputs
# If pred_is_cm is True, it uses predictions as the confusion matrix, otherwise it calculates the confusion matrix using labels and predictions
# It then plots the connectivity circle using the confusion matrix and other inputs

def plot_pareto_chart(labels, dataname,
                      y_scaling = 'log',
                      cumsumpercent = True,
                      showcounts = False,
                      savename = None,
                      dpi = 1600):
    # Calculate the class distribution
    try:
        labels = pd.DataFrame(labels)[0]
        
        class_distribution = labels.value_counts()
        
        
        if cumsumpercent:
            # Calculate the cumulative percentage
            cumulative_percentage = class_distribution.cumsum() / class_distribution.sum() * 100
        else:
            cumulative_percentage = class_distribution.cumsum()
        # Plot the Pareto chart
        fig, ax1 = plt.subplots(figsize=(10, 5))
        ax1.set_yscale(y_scaling)
        ax1.set_ylabel('Frequency (log scale)')
        ax1.tick_params(axis='x', labelrotation=90)
        
        bars = ax1.bar(class_distribution.index, class_distribution)
        
        
        if showcounts:
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width() / 2.0,
                         height,
                         f'{height}',
                         ha='center',
                         va='bottom')
    
        ax2 = ax1.twinx()
        ax2.plot(cumulative_percentage.index, cumulative_percentage, 
                 color='r', marker='o', alpha = 0.5)
        if cumsumpercent:
            ax2.set_ylabel('Cumulative Percentage')
            ax2.set_ylim(0, 105)
        else:
            ax2.set_ylabel('Cumulative Count')
            
        plt.grid(False)
        
        plt.title('Pareto Chart of Class Distributions for ' + dataname + ' Data')
        if savename:
            plt.savefig(savename,dpi=dpi)
        plt.show()
        
    except:
        print('nothing in list')
        print('nothing in labels for '+dataname)
        
def sub_pareto_chart(ax, labels, dataname, cumsumpercent=True, showcounts=False):
    if labels:
        # Calculate the class distribution
        labels = pd.DataFrame(labels)[0]
        class_distribution = labels.value_counts()
    
        # Calculate the cumulative percentage
        if cumsumpercent:
            cumulative_percentage = class_distribution.cumsum() / class_distribution.sum() * 100
        else:
            cumulative_percentage = class_distribution.cumsum()
    
        # Plot the Pareto chart
        bars = ax.bar(class_distribution.index, class_distribution)
        ax.set_ylabel('Frequency')
        ax.tick_params(axis='x', labelrotation=90)
    
        if showcounts:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2.0,
                        height,
                        f'{height}',
                        ha='center',
                        va='bottom')
    
        ax2 = ax.twinx()
        ax2.grid(False)
        ax2.plot(cumulative_percentage.index, cumulative_percentage, color='r', marker='o', alpha=0.5)
        if cumsumpercent:
            ax2.set_ylabel('Cumulative Percentage')
        else:
            ax2.set_ylabel('Cumulative Count')
    
        ax2.set_ylim(0, 105)  # Set the y-limits of the second axis to 0 and 100
    
        ax.set_title('Pareto Chart of Class Distributions for ' + dataname)
        
    else:
        ax.set_title('Pareto Chart of Class Distributions for ' + dataname)
        ax.text(0.5, 0.5, 'no instances', horizontalalignment='center',
                verticalalignment='center', transform=ax.transAxes)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)


def circlplot(predictions, labels, figure_title, cmap, 
         face_color = (1, 1, 1, 1), 
         node_edge_color = (1, 1, 1, 1), 
         text_color = (0, 0, 0, 1), 
         pred_is_cm = False):
 
    # Import the required library
    from mne_connectivity.viz import plot_connectivity_circle

    # If pred_is_cm is True, use predictions as the confusion matrix
    if pred_is_cm:
        cm = predictions
    # Otherwise, calculate the confusion matrix using labels and predictions
    else:
        cm = confusion_matrix(y_true = labels, y_pred = predictions, normalize='true')

    # Get the set of unique labels
    labels = set(labels)

    # Plot the connectivity circle using the confusion matrix and other inputs
    circplot = plot_connectivity_circle(cm,
                                        set(labels),
                                        n_lines=400,
                                        node_angles=None, 
                                        node_colors=None,
                                        linewidth = 10,
                                        colormap = cmap,
                                        title=figure_title,
                                        facecolor = face_color,
                                        node_edgecolor=node_edge_color,
                                        textcolor = text_color,
                                        vmin = 0,
                                        vmax = 1)
    return circplot


# This function plots a confusion matrix with category labels
# It takes in model parameters, Y_evaluation, Y_true, Graph_title_prefix, figure_size, title, rotate_xy, save, filepath, filename, saveformat, transparentbg, image_dpi, normalise, subtitles, fontscale, and colourmap as inputs
def calculate_average_model_data(model_data):
    average_model_data = {}
    for run in model_data:
        dataset_names = list(model_data[run]['Evaluation'].keys())[1:]
        for dataset_name in dataset_names:
            if isinstance(model_data[run]['Evaluation'][dataset_name], dict) and dataset_name != 'individual_datasets':
                average_model_data[dataset_name] = {}
    for run in model_data:
        for dataset in average_model_data:
            for metric in model_data[run]['Evaluation'][dataset]:
                average_model_data[dataset][metric] = []
    for run in model_data:
        for dataset in average_model_data:
            for metric in model_data[run]['Evaluation'][dataset]:
                value = model_data[run]['Evaluation'][dataset][metric]
                average_model_data[dataset][metric].append(value)
                        
    for key in average_model_data:
        for subkey in average_model_data[key]:
                if isinstance(average_model_data[key][subkey][0], float):
                    average_model_data[key][subkey] = np.mean(average_model_data[key][subkey], axis=0)
                elif isinstance(average_model_data[key][subkey][0], np.ndarray):
                    pass#print("numpy array, nothing is being done with this yet. Perhaps with time we could group by class then average")
                else:
                    print(key)
                    print(subkey)
                    print(type(average_model_data[key][subkey][0]))            
    return average_model_data

def confusion_matrix_plot(model_parameters,
                          unique,
                          which_run,
                          Y_evaluation ='holdout_evaluation',
                          Y_true = 'Y_test',
                          Graph_title_prefix = 'Testing ',
                          figure_size = (10,10),
                          title = None,
                          rotate_xy = (None,None),
                          save=False, filepath = None, filename = None, 
                          saveformat = None, transparentbg = True, 
                          image_dpi = 300,
                          normalise = None,
                          subtitles=True,
                          fontscale=1.5,
                          colourmap=None,
                          savecm=False,
                          cmpath=None
                          ):
 
    # Get the model from the model parameters
    if which_run == 'all':
        model = calculate_average_model_data(model_parameters)
    else:
        model = model_parameters[which_run]
        
        
    outputfolder = unique

    # Set the title addendum based on the normalise input
    match normalise:
        case 'true':
            titleaddendum = 'Matrix values as % of Actual'
        case 'pred':
            titleaddendum = 'Matrix values as % of predicted'
        case None:
            titleaddendum = 'Matrix values as count'

    # Get the unique labels from the model parameters
    run0 = list(model_parameters.keys())[0]
    dictionary = [i for i in model_parameters[run0]['unique_labels'] if i not in model_parameters[run0]['ignore_classes']]

    # Get the F1 score and categorical accuracy from the model
    f1_dset = 'F1: '+str(np.round(model[Y_evaluation]['F1']*100,2))+'%'
    CAC_dset = 'CA: '+str(np.round(model[Y_evaluation]['categorical_accuracy']*100,2))+'%'
    loss_dset = 'L: '+str(np.round(model[Y_evaluation]['loss'],4)) #TODO functionalise this

    # Set the subtitles if required
    if subtitles:
        subtitle1 = f1_dset+' | '+CAC_dset+' | '+loss_dset
        subtitle2 = titleaddendum

    # Set the title
    temptitle = str(Graph_title_prefix+title)

    # Calculate the confusion matrix
    sortedlabels = sorted(list(model_parameters[run0]['label_dictionary'].keys()))

    sortedlabels = [model_parameters[run0]['label_dictionary'][i] for i in sortedlabels]
    
    if which_run == 'all':
        for evalset in model:
            cm_list = []
            y_true = model[evalset]['true_values']
            y_pred = model[evalset]['prediction_values']
            for t,p in zip(y_true, y_pred):
                cm = confusion_matrix(t.argmax(1),
                                      p.argmax(1),
                                      normalize=normalise,
                                      labels=sortedlabels)
                cm_list.append(cm)
                number_of_cm = len(cm_list)
                finalcm = sum(cm_list)/number_of_cm
            model[evalset]["CM"] = finalcm
        cm = model[Y_evaluation]["CM"]
        
    else:
        cm = confusion_matrix(model[Y_evaluation]['true_values'].argmax(1),
                            model[Y_evaluation]['prediction_values'].argmax(1),
                            normalize=normalise,
                            labels=sortedlabels)

    # Get the size of the confusion matrix
    cmsize = cm.shape

    # Create a pandas dataframe from the confusion matrix
    df_cm = pd.DataFrame(cm, range(cmsize[0]),range(cmsize[1]))

    # If there are labels, set the column and index names of the dataframe to the labels
    if len(dictionary,)>0:
        df_cm = pd.DataFrame(df_cm)
        df_cm.columns = sorted(dictionary)
        df_cm.index = sorted(dictionary)

    # Create the heatmap using seaborn
    plt.figure(figsize=figure_size)
    sns.set(font_scale=fontscale)  

    # If normalise is not None, multiply the dataframe by 100
    if not normalise == None:
        df_cm = df_cm*100

    # Plot the heatmap
    sns.heatmap(df_cm, 
                annot=True, 
                fmt='.0f',
                cmap=colourmap,
                linewidths=1.5,
                square=True,
                cbar_kws={'shrink': 0.85, 'aspect': 20, 'pad': 0.01}
                )

    # Set the ylabel and xlabel
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')

    # Set the title and subtitles if required
    plt.text(0,-1.9, s=temptitle, fontsize=24*fontscale)
    if subtitles:
        plt.text(0,-0.9, s=subtitle1, fontsize=14*fontscale, ha='left')
        plt.text(0,-0.2, s=subtitle2, fontsize=14*fontscale, ha='left')

    # Rotate the x and y labels if required
    if not rotate_xy[1] == None:
        plt.xticks(rotation = rotate_xy[1], ha='right')
    if not rotate_xy[0] == None:
        plt.yticks(rotation = rotate_xy[0], ha='right')

    # Save the plot if required
    if save:
        plt.savefig(fname = filepath+filename+'.'+saveformat, 
                    transparent = transparentbg, 
                    bbox_inches='tight', 
                    format = saveformat, 
                    dpi = image_dpi)
        print(f"saved SVG graph of CM from {filename} to {filepath}")
    # Show the plot
    plt.tight_layout()

    plt.show()
    
    #save the confusion matrix
    if savecm:
        savename = title.replace(" ","_")
        df_cm.to_csv(filepath+filename+'.csv',sep=",")
        print(f"saved csv formatted data of CM from {filename} to {filepath}")
    # Reset the seaborn font scale
    sns.set(font_scale=1)
    
# =============================================================================
# # Define a function that plots an average confusion matrix
# def plot_cm_averages(model_parameters,
#                      Y_evaluation='holdout_evaluation',
#                      Y_true='Y_test',
#                      Graph_title_prefix='Testing ',
#                      figure_size=(10, 10),
#                      title=None,
#                      rotate_xy=(None, None),
#                      save=False,
#                      filepath=None,
#                      filename=None,
#                      saveformat=None,
#                      transparentbg=True,
#                      image_dpi=300,
#                      normalise=None,
#                      subtitles=True,
#                      fontscale=1.5,
#                      colourmap=None,
#                      savecm=False,
#                      cmpath=None):
#     
#     # Get the model from the model parameters
#     model_data = model_parameters['model_data']
#     outputfolder = model_parameters['unique']
#     
#     # Set the title addendum based on the normalise input
#     match normalise:
#         case 'true':
#             titleaddendum = 'Values as % of Actual'
#         case 'pred':
#             titleaddendum = 'Values as % of predicted'
#         case None:
#             titleaddendum = 'Values as Counts'
#     
#     # Get the unique labels from the model parameters
#     dictionary = list(model_parameters['label_dictionary'].keys())
#     dictionary = [model_parameters['label_dictionary'][i] for i in dictionary]
#     
#     # Initialize lists to store F1 scores, categorical accuracies, and losses
#     f1_scores = []
#     categorical_accuracies = []
#     losses = []
#     
#     # Calculate the confusion matrix for each run and collect F1 scores, categorical accuracies, and losses
#     confusion_matrices = []
#     for run in model_data:
#         model = model_data[run]
#         f1_scores.append(model[Y_evaluation]['F1'])
#         categorical_accuracies.append(model[Y_evaluation]['categorical_accuracy'])
#         losses.append(model[Y_evaluation]['loss'])
#         
#         cm = confusion_matrix(model_parameters[Y_true].argmax(1),
#                               model[Y_evaluation]['prediction_values'].argmax(1),
#                               normalize=normalise,
#                               labels=dictionary)
#         confusion_matrices.append(cm)
#     
#     # Calculate the average F1 score, categorical accuracy, and loss
#     avg_f1_score = np.mean(f1_scores) * 100
#     avg_categorical_accuracy = np.mean(categorical_accuracies) * 100
#     avg_loss = np.mean(losses)
#     
#     # Set the subtitles if required
#     if subtitles:
#         subtitle1 = f'Average F1: {avg_f1_score:.2f}% | Average Cat. Acc: {avg_categorical_accuracy:.2f}% | Average Loss: {avg_loss:.4f}'
#         subtitle2 = titleaddendum
#     
#     # Set the title
#     temptitle = str(Graph_title_prefix + title)
#     
#     # Calculate the average confusion matrix
#     avg_cm = np.mean(confusion_matrices, axis=0)
#     
#     # Re-organize the confusion matrix rows/columns in alphabetical order
#     sorted_indices = np.argsort(dictionary)
#     sorted_labels = [dictionary[i] for i in sorted_indices]
#     avg_cm = avg_cm[sorted_indices][:, sorted_indices]
#     
#     # Create a pandas dataframe from the confusion matrix
#     df_cm = pd.DataFrame(avg_cm, sorted_labels, sorted_labels)
#     
#     # Create the heatmap using seaborn
#     plt.figure(figsize=figure_size)
#     sns.set(font_scale=fontscale)
#     
#     # If normalise is not None, multiply the dataframe by 100
#     if not normalise == None:
#         df_cm = df_cm * 100
#     
#     # Plot the heatmap
#     sns.heatmap(df_cm,
#                 annot=True,
#                 fmt='.0f',
#                 cmap=colourmap,
#                 linewidths=1.5,
#                 square=True)
#     
#     # Set the ylabel and xlabel
#     plt.ylabel('Actual label')
#     plt.xlabel('Predicted label')
#     
#     # Set the title and subtitles if required
#     plt.text(0, -1.9, s=temptitle, fontsize=24 * fontscale)
#     if subtitles:
#         plt.text(0, -0.9, s=subtitle1, fontsize=14 * fontscale, ha='left')
#         plt.text(0, -0.2, s=subtitle2, fontsize=14 * fontscale, ha='left')
#     
#     # Rotate the x and y labels if required
#     if not rotate_xy[1] == None:
#         plt.xticks(rotation=rotate_xy[1], ha='right')
#     if not rotate_xy[0] == None:
#         plt.yticks(rotation=rotate_xy[0], ha='right')
#     
#     # Save the plot if required
#     if save:
#         plt.savefig(fname=filepath + filename + Graph_title_prefix + '.' + saveformat,
#                     transparent=transparentbg,
#                     bbox_inches='tight',
#                     format=saveformat,
#                     dpi=image_dpi)
#     
#     # Show the plot
#     plt.tight_layout()
#     plt.show()
#     
#     # Save the confusion matrix
#     if savecm:
#         df_cm.to_csv(outputfolder + cmpath, sep=",")
#     
#     # Reset the seaborn font scale
#     sns.set(font_scale=1)
# =============================================================================
  
# This function plots ROC curves
def plot_roc(name, labels, predictions, **kwargs):
    # Calculate false positive rate, true positive rate and thresholds
    fp, tp, _ = sklearn.metrics.roc_curve(labels, predictions)
    
    # Plot the ROC curve
    plt.plot(100*fp, 100*tp, label=name, linewidth=2, **kwargs)
    plt.xlabel('False positives [%]')
    plt.ylabel('True positives [%]')
    plt.xlim([-0.5,20])
    plt.ylim([80,100.5])
    plt.grid(True)
    ax = plt.gca()
    ax.set_aspect('equal')
 
# This function calculates the F1 score
def f1score(prediction_results,round_to = 4):
   # Get recall and precision from prediction results
   recall = prediction_results['recall']
   precision = prediction_results['precision']
   
   # Calculate F1 score
   try:
       f1 = (2*recall*precision)/(recall+precision)
       f1 = np.round_(f1,round_to)
   except ZeroDivisionError:
       print('divide by zero error when trying to calculate F1 Score')
       f1 = 0
   except:
       print('something went wrong calculating F1 score')
       f1 = 0 
       
   return float(f1)

# This function merges loose CSV files into a single table
def LooseCsvMerger(directory,KeyName,DataName):
    '''
    -merge csv
    -merge files
    -merge loose files
    
    This function combines loose CSV files into a single table. It is assumed that data
    being combined has the same key values, e.g. for FTIR data that all
    files being combined have the same number of observations and have the
    exact same wavenumbers recorded
    
    Parameters
    ----------
    directory : str
        path to the directory that contains the files that need joining
    KeyName : str
        the title of the column that contains they key info
    DataName : str
        the title of the column that contains the data

    Returns
    -------
    dataframe : pandas dataframe
        this will output a pandas dataframe, which can be saved readily to a
        csv as well. That which was specified as 'key' will form column names
        while that which was specified as 'data' will populate rows. Row names
        will be the file names.

    '''

    # Get list of file names in directory
    filenames = [fetchfilelabel(i) for i in glob.glob(directory+'/*.csv')]
    
    # Read data from each file into a dictionary
    data = {}
    for i in filenames:
        data[i] = pd.read_csv(directory+'/'+i+'.csv')
       
    # Create an empty dataframe with column names from first file
    dataframe = pd.DataFrame(columns=list(data[filenames[0]][KeyName]))
    
    # Add data from each file to the dataframe
    for key,value in zip(data.keys(),data.values()):
        row = pd.DataFrame(value[DataName])
        row.index, row.columns = value[KeyName], [key]
        dataframe = pd.concat([dataframe,row.transpose()])
    
    return dataframe