import os
import tempfile

import numpy as np

import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.utils import to_categorical
import time

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils import compute_class_weight
import numpy as np

from functions.ftirmlfunctions import f1score, funcy_dics, categorical_focal_loss

class MinMaxScaling(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(MinMaxScaling, self).__init__()

    def build(self, input_shape):
        pass

    def call(self, inputs):
        min_val = tf.reduce_min(inputs, axis=[1, 2], keepdims=True)
        max_val = tf.reduce_max(inputs, axis=[1, 2], keepdims=True)
        return (inputs - min_val) / (max_val - min_val + tf.keras.backend.epsilon())

    def compute_output_shape(self, input_shape):
        return input_shape

class AreaFlip(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(AreaFlip, self).__init__()

    def build(self, input_shape):
        pass

    def call(self, inputs):
        # Calculate sums
        sum_values = tf.reduce_sum(inputs - 0.5, axis=[1, 2])

        # Create mask where sum_value is less than 0 (indicating larger area under curve)
        mask = tf.less(sum_values ,0)

        # Reshape mask to match inputs shape
        mask = tf.reshape(mask , (-1 ,1))

        # Flip spectrum by multiplying with -1 and add 1 to get back original scale  
        inputs_flipped = -(inputs - 1)

        # Use mask to select original or flipped inputs
        outputs = tf.where(mask ,inputs_flipped ,inputs )  

        return outputs

class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)

#this allows us to use stratified minibatches
def stratified_batch_generator(X, y, batch_size, num_classes):
    # Convert one-hot encoded labels to label-encoded form
    y_label_encoded = np.argmax(y, axis=1)
    
    # Convert one-hot encoded labels to float32 format
    y = y.astype('float32')
    
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=None, train_size=batch_size)
    while True:
        for train_index, _ in splitter.split(X, y_label_encoded):
            X_batch, y_batch = X[train_index], y[train_index]
            # Ensure y_batch has the correct shape
            y_batch = np.reshape(y_batch, (batch_size, -1))
            y_batch = tf.convert_to_tensor(y_batch)
            X_batch = tf.convert_to_tensor(X_batch)
            
            yield X_batch, y_batch
            


#this function requires build dataset have been run, so that things like
#indim/outdime are generated
def make_model(model_parameters):
    '''
    

    Parameters
    ----------
    model_parameters : TYPE
        DESCRIPTION.

    Returns
    -------
    model_data : TYPE
        DESCRIPTION.

    '''

    indim = model_parameters['dimension_in']
    outdim = model_parameters['dimension_out']
    modelname = model_parameters['model_name']

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
            case 'f1_score_micro':
                trainingmetrics.append(keras.metrics.F1Score(name=metricmoniker, average='micro'))
            case 'f1_score_macro':
                trainingmetrics.append(keras.metrics.F1Score(name=metricmoniker, average='macro'))
            case 'f1_score_weighted':
                trainingmetrics.append(keras.metrics.F1Score(name=metricmoniker, average='weighted'))
            case 'f1_score':
                trainingmetrics.append(keras.metrics.F1Score(name=metricmoniker, average=None))
            case 'categorical_crossentropy'|'CCE'|'cce':
                trainingmetrics.append(tf.keras.metrics.CategoricalCrossentropy(name=metricmoniker))
            
                
    layerdict = {
        'input'     : layers.Dense,
        'dense'     : layers.Dense,
        'output'    : layers.Dense,
        '1dcnn'     : layers.Conv1D,
        'dropout'   : layers.Dropout,
        'nobias'    : False,
        'bias'      : True
        }
    
    model = keras.Sequential(name=model_parameters['model_name'])
    layercount = 0
    
    for layer in model_parameters['network_layers']:
        layercount += 1
        match layer['layer_group']:
            case 'minmax':
                model.add(MinMaxScaling(input_dim = indim))
            case 'modefix':
                model.add(AreaFlip(input_dim = indim))
            case 'input': # This can be it's own layer, but can be a dense layer by adding inpput shape in first layer like this https://www.tensorflow.org/api_docs/python/tf/keras/layers/InputLayer
                numneurons = layer['neurons']
                if layer['neurons'] == "auto":
                    numneurons = indim
                model.add(layerdict[layer['layertype']](
                    units = numneurons,
                    name = f"{modelname}-input-layer_{layercount}",
                    activation = layer['layer_activation'],
                    input_dim = indim))
            case 'output':
                model.add(layerdict[layer['layertype']](outdim,
                                       name = f"{modelname}-output-layer_{layercount}",
                                       activation = layer['layer_activation'],
                                       use_bias=layerdict[layer['layer_bias']])
                          )
            case 'hidden':
                match layer['layertype']:
                    case 'dense':
                        model.add(layerdict[layer['layertype']](
                            units = layer['neurons'],
                            activation = layer['layer_activation'],
                            kernel_regularizer=layer['regularization'],                            
                            use_bias=layerdict[layer['layer_bias']],
                            input_dim = indim,
                            name = f"{modelname}-hidden-layer_{layercount}"
                            )
                            )
                    case '1dcnn':
                        model.add(layerdict[layer['layertype']](
                            filters = 32, # TODO: if there is ever interest in CNN, need to update this
                            kernel_size= 5, 
                            padding = 'casual',
                            use_bias = layerdict[layer['layer_bias']]),
                            name = f"{modelname}-1D)CNN-Hidden-layer_{layercount}"
                            )
                    case 'dropout':
                        model.add(layerdict[layer['layertype']](
                            rate = layer['rate'],
                            name = f"{modelname}-Dropout-layer_{layercount}"
                            ))
                
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
            optimisingAlgorithm = funcy_dics(Adam_params_dict, tf.keras.optimizers.Adam)
            
    match model_parameters['loss']['name']:
        case 'categorical_crossentropy'|'CCE'|'cce':
            CCE_params_dict = {'from_logits':False,
                               'label_smoothing':0.0,
                               'name':'categorical_crossentropy'}
            CCE_params_dict.update(model_parameters['loss'])
            lossAlgorithm = funcy_dics(CCE_params_dict, tf.keras.losses.CategoricalCrossentropy)
            
        case 'Focal'|'focal'|'cfce':
            Focal_params_dict = {'gamma':2.0,
                                 'name':'categorical_focal_crossentropy'}
            Focal_params_dict.update(model_parameters['loss'])
            lossAlgorithm = funcy_dics(Focal_params_dict, categorical_focal_loss)
            
        case 'tfFocal':            
            # Calculate class weights using compute_class_weight
            train_labels = [model_parameters['reverse_dict'][i] for i in model_parameters['Y_train'].argmax(1)]
            class_weights = compute_class_weight(class_weight='balanced', 
                                                 classes=model_parameters['unique_labels'], 
                                                 y=train_labels)
            class_weights = list(class_weights)  # Convert to list if not already
            
            # Create the CategoricalFocalCrossentropy loss with dynamically calculated alpha
            loss = tf.keras.losses.CategoricalFocalCrossentropy(alpha=class_weights)

            Focal_params_dict = {'alpha':class_weights,
                                 'gamma':2.0,
                                 'name':'categorical_focal_crossentropy'}
            
            Focal_params_dict.update(model_parameters['loss'])
            lossAlgorithm = funcy_dics(Focal_params_dict, tf.keras.losses.CategoricalFocalCrossentropy)
    
    model.compile(loss = lossAlgorithm, 
                  optimizer = optimisingAlgorithm, 
                  metrics = trainingmetrics)
    
    model.build()
    model.summary()
    initial_weights = os.path.join(tempfile.mkdtemp(), 'initial_weights')
    model.save_weights(initial_weights)
        
    return model

def get_positional_encoding(max_length, d_model):
    angle_rads = get_angles(np.arange(max_length)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)
    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)

def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates

def train_model(model_parameters):
    d_model = max_length = model_parameters['X_train'].shape[1]
    #model_parameters['positional_encoding'] = get_positional_encoding(max_length, d_model)
    
    model = model_parameters['model']
    try:
        model_name = model_parameters['model_name']
    except:
        model_name = ''
        
    log_dir = model_parameters['unique']+'/model_training_checkpoints/' + model_name + '/' # set the log directory for tensorboard'
    
   
    callback_list = []
    
    if model_parameters["checkpoints"]["EarlyStopping"]:
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor=model_parameters['monitor_metric'], 
            verbose=model_parameters['verbose_monitor_ES'],
            patience=model_parameters['monitor patience'],
            mode=model_parameters['monitor_goal'],
            restore_best_weights=model_parameters['restore_top_weight'],
            start_from_epoch=model_parameters['start_from_epoch'],
            min_delta=model_parameters['minimum_change'])
        
        callback_list += [early_stopping]
        
    if model_parameters["checkpoints"]["Tensorboard"]:
        tensorboard = tf.keras.callbacks.TensorBoard(log_dir, 
                                       update_freq='epoch',
                                       write_images=False,
                                       write_graph=False,
                                       histogram_freq=100,
                                       write_steps_per_second=False,
                                       profile_batch=0,
                                       embeddings_freq=0,
                                       embeddings_metadata=None,
                                       )
        callback_list += [tensorboard]
        
    if model_parameters["checkpoints"]["SaveCheckpoints"]:
        save_checkpoints = tf.keras.callbacks.ModelCheckpoint(
            filepath=log_dir+'/checkpoint/',
            save_weights_only=True,
            monitor=model_parameters['monitor_metric'],
            mode=model_parameters['monitor_goal'],
            verbose=model_parameters['verbose_monitor_CP'],
            save_best_only=True),
        TimeHistory(),
        funcy_dics(model_parameters['LR_reducer'], tf.keras.callbacks.ReduceLROnPlateau)
        callback_list += [save_checkpoints]
        
    
    batchmode = model_parameters['batch_params']["batch_mode"]
    batchsize = model_parameters['batch_params']["batch_size"]
    
    match batchmode:
        case "normal":
            training_history = model.fit(
                x=model_parameters['X_train'],
                y=model_parameters['Y_train'],
                batch_size = batchsize,
                shuffle = True,
                steps_per_epoch=model_parameters['steps_per_epoch'], #og 20
                epochs=model_parameters['max_epochs'], #og 10
                verbose=model_parameters['verbose_monitor'],
                callbacks=[callback_list],
                validation_data= (model_parameters['X_val'],
                                  model_parameters['Y_val'])
                )
        case "stratified_minibatch":
            num_classes = model_parameters['Y_train'].shape[1]
            batchgen = stratified_batch_generator(
                X=model_parameters['X_train'],
                y=model_parameters['Y_train'],
                batch_size = batchsize*num_classes, 
                num_classes=num_classes)
            
            val_gen = stratified_batch_generator(
                X=model_parameters['X_val'],
                y=model_parameters['Y_val'],
                batch_size = batchsize*num_classes, 
                num_classes=num_classes)
            
            # Calculate the validation steps
            val_steps = len(model_parameters['X_val']) // (batchsize*num_classes)

            training_history = model.fit(
                batchgen,
                steps_per_epoch=model_parameters['steps_per_epoch'],
                epochs=model_parameters['max_epochs'],
                verbose=model_parameters['verbose_monitor'],
                callbacks=[early_stopping],
                validation_data= val_gen,
                validation_steps=val_steps
                )
        case _:
            print("error, training mode isn't set right\nAvliable modes: \n'normal'\n'stratified_minibatch'\ncheck for typos")
            return "error", "error"
        
    return model, training_history

def evaluate_model(model_parameters):
    model = {}
    model["model"] = model_parameters["model"]
    def evaldict(X,Y,model):
        evaluation_values = model['model'].evaluate(X,
                                                    Y,
                                                    verbose=model_parameters['verbose_monitor']
                                                    )
        predicted_values = model['model'].predict(X)
        
        metricsnames = []
        
        for metricname in model_parameters['training_metrics']:
            metricsnames.append(metricname[0])
        metricsnames = [model['model'].metrics_names[0]]+metricsnames
        evaluation_results = dict(zip(metricsnames,
                                      evaluation_values)
                                  )

        evaluation_results.update({'F1':f1score(evaluation_results)})   
        evaluation_results.update({'prediction_values':predicted_values})
        evaluation_results.update({'true_values':Y})
        return evaluation_results
    
    print('evaluating on trianing data')
    train_evaluation = evaldict(model_parameters['X_train'],
                                model_parameters['Y_train'],
                                model
                                )
    print('evaluating on validation data')
    validation_evaluation = evaldict(model_parameters['X_val'],
                                model_parameters['Y_val'],
                                model
                                )
    print('evaluating on holdout data')
    holdout_evaluation = evaldict(model_parameters['X_test'],
                                model_parameters['Y_test'],
                                model
                                )
    x_all = np.append(model_parameters['X_test'],
                      model_parameters['X_train'],
                      axis=0)
    x_all = np.append(x_all,
                      model_parameters['X_val'],
                      axis=0)
    y_all = np.append(model_parameters['Y_test'],
                      model_parameters['Y_train'],
                      axis=0)
    y_all = np.append(y_all
                      ,model_parameters['Y_val'],
                      axis=0)
    
    print('evaluating on all data')
    wholedata_evaluation = evaldict(x_all,
                                y_all,
                                model
                                )
    
    individual_datasets = {}
    
    for i in model_parameters['split_data']:
        if model_parameters['datasets'][i]['trainable']:
            Y = np.array(model_parameters['split_data'][i].index)
            Y_oh = tf.one_hot(np.vectorize(model_parameters['label_dictionary'].get)(Y),
                              model_parameters['dimension_out'],
                              dtype=tf.float32)
            print(f'evaluating on {i} only')
            evaluation = evaldict(model_parameters['split_data'][i],
                                  Y_oh,
                                  model)
            model_parameters['datasets'][i]['evaluation_results'] = evaluation
            individual_datasets[i] = evaluation
        
        if not model_parameters['datasets'][i]['trainable']:
            Y = np.array(model_parameters['split_data'][i].index)
            Y_oh = tf.one_hot(np.vectorize(model_parameters['label_dictionary'].get)(Y),
                              model_parameters['dimension_out'],
                              dtype=tf.float32)
            print(f'evaluating on {i} only')
            predictions = model['model'].predict(np.array(model_parameters['split_data'][i]))
            model_parameters['datasets'][i]['prediction_results'] = predictions
            evaluation = evaldict(model_parameters['split_data'][i],
                                  Y_oh,
                                  model)
            model_parameters['datasets'][i]['evaluation_results'] = evaluation
            individual_datasets[i] = evaluation

    model.update({
        'train_evaluation':train_evaluation,
        'validation_evaluation':validation_evaluation,
        'holdout_evaluation':holdout_evaluation,
        'all_evaluation':wholedata_evaluation,
        'individual_datasets':individual_datasets
        })
    
    model_parameters["Evaluation"] = model
    
    return model_parameters

