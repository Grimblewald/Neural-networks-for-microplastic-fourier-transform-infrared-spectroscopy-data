# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 15:47:07 2023

@author: frith
"""
import os
import tempfile
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from functions.extrafunctions import CategoricalFocalLoss


class model:
    def __init__(self, name = "Default"):
        self.name = name
        self.config = {}
    
    def setConfig(self,configdict,update=True):
        if update:
            self.config.update(configdict)
        if not update:
            self.config = {}
        
    def getConfig(self):
        return self.config

    def build(self):
        # get the model configurations
        model_parameters = self.getConfig()
        
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
                    model.add(layers.Conv1D(
                        filters = 32, # TODO: if there is ever interest in CNN, need to update this
                        kernel_size= 5, 
                        padding = 'casual',
                        use_bias = layer_bias)
                        )

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
    
    def train(self):
        print("t")

    def evaluate(self):
        print("e")
        
        
"""
workspace below, delete everything below here
"""
