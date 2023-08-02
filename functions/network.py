# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 15:47:07 2023

@author: fritz
"""
import os
import tempfile
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from functions.extrafunctions import categoricalFocalLoss
import functions.datawrangler as dw

class DataNotLoadedError(Exception):
    pass

class NoConfigError(Exception):
    pass


class MinMaxScaling(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(MinMaxScaling, self).__init__(**kwargs)

    def call(self, inputs):
        min_val = tf.reduce_min(inputs, axis=[1, 2], keepdims=True)
        max_val = tf.reduce_max(inputs, axis=[1, 2], keepdims=True)
        return (inputs - min_val) / (max_val - min_val + tf.keras.backend.epsilon())
    
class AreaFlip(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(AreaFlip, self).__init__(**kwargs)

    def call(self, inputs):
        # Calculate areas
        area_0_values = tf.reduce_sum(inputs, axis=[1, 2])
        area_1_values = tf.reduce_sum(1 - inputs, axis=[1, 2])

        # Create mask where area between 0 and values is larger than between 1 and values
        mask = tf.greater(area_0_values, area_1_values)

        # Reshape mask to match inputs shape
        mask = tf.reshape(mask, (-1, 1))

        # Flip spectrum by multiplying with -1 and subtract spectrum minimum
        inputs_flipped = -inputs - tf.reduce_min(inputs, axis=[1, 2], keepdims=True)

        # Use mask to select original or flipped inputs
        outputs = tf.where(mask, inputs_flipped, inputs)

        return outputs

class model:
    def __init__(self, name = "Default"):
        self.name = name
        
        self.configloaded = False
        self.config = {}
        self.dataloaded = False
        
        self.data = {}
        self.tfeatures = np.empty(1)
        self.tlabels = np.empty(1)
        self.vfeatures = np.empty(1)
        self.vlabels = np.empty(1)
        self.hfeatures = np.empty(1)
        self.hlabels = np.empty(1)
        
        self.model = None
    
    def setConfig(self,configdict,mode='update'):
        '''
        This allows updating and replacing of the existing configuration.

        Parameters
        ----------
        configdict : Dict
            DESCRIPTION. This should be a dictionary which contains relevant
            entries for the configuration/trianing/use of this model
        mode : Str, optional
            DESCRIPTION. The default is 'update', the alternative being 'replace'
            as the terms might suggest, one updates the dictionary using the 
            provided values, while the other replaces the existing dictionary
            with the one provided

        Returns
        -------
        None.

        '''
        if mode == 'update':
            self.config.update(configdict)
            self.configloaded = True
        if mode == 'replace':
            self.config = configdict
            self.configloaded = True
        
    def getConfig(self):
        return self.config
    
    def ingestData(self):
        data = self.config["datasets"]
        data = dw.read(self.data)
        data = dw.process(config = self.config, data = data)
        
        self.data = data
    
    def loadData(self, data): #TODO
        if not self.configloaded:
            raise NoConfigError("No config loaded, please set configuration before trying to load data")
                
        try:
            self.tfeatures = data['training_features']
            self.tlabels = data['training_labels']
            self.vfeatures = data['validation_features']
            self.vlabels = data['validation_labels']
            self.hfeatures = data['holdout_features']
            self.hlabels = data['holdout_labels']
        except:
            raise DataNotLoadedError("could not load some data, please check for issues")
            
        self.dataloaded = True
        
    def loadModel(): #TODO
        # put in a check to see if data has been built alrteady and warn that if loading models
        # then data should be built afterwards so that it can be modified to match
        # the requriements of the loaded model
        print("loading not implemented yet")

    def buildModel(self): #TODO
        '''
        This builds the model according to the specificaitons set in the config
        and according to the shape of the data ingested for trianing on

        Raises
        ------
        DataNotLoadedError
            DESCRIPTION.

        Returns
        -------
        model : TYPE
            DESCRIPTION. A tensorflow model

        '''
        # get the model configurations
        model_parameters = self.getConfig()
        
        if not self.databuilt:
            raise DataNotLoadedError("Data must be loaded before building the model.\n\n This is required because the model builds its input and output dimensions\n around information from the dataset such as sample length or number of classes.")
        
        indim = model_parameters['dimension_in']
        outdim = model_parameters['dimension_out']
        
        #set up metrics using the configuration settings
        trainingmetrics = []
        for metric in model_parameters['training_metrics']:
            metrictype, metricmoniker = metric
            
            match metrictype:
                case 'auc':
                    trainingmetrics.append(keras.metrics.AUC(name=metricmoniker))
                case 'precision':
                    trainingmetrics.append(keras.metrics.Precision(name=metricmoniker))
                case 'recall':
                    trainingmetrics.append(keras.metrics.Recall(name=metricmoniker))
                case 'categorical_accuracy':
                    trainingmetrics.append(keras.metrics.CategoricalAccuracy(name=metricmoniker))
                case 'PR AUC':
                    trainingmetrics.append(keras.metrics.AUC(name=metricmoniker, curve= 'PR'))
              
        #intitialize a seqential keras model
        model = keras.Sequential(name=model_parameters['model_name'])
        
        # go through the layers defined in the config and add these to the sequential model
        for layer in model_parameters['network_layers']:
            
            lcfg = model_parameters['network_layers'][layer] #set layer config
            
            # setup more useful names
            layer_group = lcfg["name"]
            layertype = lcfg["type"]
            neurons = lcfg["neurons"]
            layer_activation = lcfg["activation"]
            layer_weight_init = lcfg["init"]
            layer_bias = lcfg["bias"]
            l1 = lcfg["L1"]
            l2 = lcfg["L2"]

            #get string to set right reg. type later
            if l1 and l2:
                reguralization = "l1_l2"
            elif l1 and not l2:
                reguralization = "l1"
            elif not l1 and l2:
                reguralization = "l2"
            elif not l1 and not l2:
                reguralization = None
                
            # get weight intitialiser
            match layer_weight_init:
                case "ones":
                    weightini = tf.keras.initializers.Constant(value=1)
                case "GlorotUniform":
                    weightini = keras.initializers.GlorotUniform()
                case "GlorotNormal":
                    weightini = keras.initializers.GlorotNormal()
                    
            #set the specified layer type
            match layer_group:
                case 'input':
                    model.add(
                        layers.Dense(
                            units = neurons, 
                            activation = layer_activation,
                            kernel_initializer=tf.keras.initializers.Constant(value=1),
                            input_dim = indim
                            )
                        )
                case 'output':
                    model.add(
                        layers.Dense(
                            outdim,
                            activation = layer_activation,
                            use_bias=layer_bias
                            )
                        )
                case 'hidden':
                    match layertype:
                        case 'dense':
                            model.add(
                                layers.Dense(
                                    units = neurons,
                                    activation = layer_activation,
                                    kernel_initializer=weightini,
                                    kernel_regularizer=reguralization,                            
                                    use_bias=layer_bias
                                    )
                                )
                        case '1dconv':
                            model.add(
                                layers.Conv1D(
                                    filters = 32, # TODO: if there is ever interest in CNN, need to update this
                                    kernel_size= 5, 
                                    padding = 'casual',
                                    use_bias = layer_bias
                                    )
                                )
                case "preprocessing":
                    match layertype:
                        case 'minmax':
                            model.add(MinMaxScaling())
                        case 'areaflip':
                            model.add(AreaFlip())

        #Optimiser stuff
        match model_parameters['optimizer']['name']:
            case 'Adam' | 'adam':
                #set up default values                
                Adam_params_dict = {'name': 'Adam',
                                    'learning_rate': 0.001,
                                    'beta_1': 0.9,
                                    'beta_2': 0.999,
                                    'epsilon': 1e-07,
                                    'amsgrad': False}
                #update with config
                Adam_params_dict.update(model_parameters['optimizer'])
                
                #set uptimiser
                optimisingAlgorithm = tf.keras.optimizers.Adam(**Adam_params_dict)
                
        match model_parameters['loss']['name']:
            case 'categorical_crossentropy'|'CCE'|'cce':
                CCE_params_dict = {'from_logits':False,
                                   'label_smoothing':0.0,
                                   'axis':-1,
                                   'name':'categorical_crossentropy'}
                CCE_params_dict.update(model_parameters['loss'])
                lossAlgorithm = tf.keras.losses.CategoricalCrossentropy(**CCE_params_dict)
            case 'Focal'|'focal'|'cfce':
                Focal_params_dict = {'gamma':2.0,
                                     'name':'categorical_focal_crossentropy'}
                Focal_params_dict.update(model_parameters['loss'])
                
                lossAlgorithm = CategoricalFocalLoss(**Focal_params_dict)
        
        model.compile(loss = lossAlgorithm, 
                      optimizer = optimisingAlgorithm, 
                      metrics = trainingmetrics)
        
        model.build()
        model.summary()
        initial_weights = os.path.join(tempfile.mkdtemp(), 'initial_weights')
        model.save_weights(initial_weights)
            
        return model
    
    def train(self): #TODO
        print("t")

    def evaluate(self): #TODO
        print("e")