
# project specific functions

from functions import network, plotting, datawrangler

model_parameters = {
    'unique'                :'./trainingoutput/',
    'datasets'            :{
        'Kedzierski' : {'path':'./data/Mode_data_A.csv',
                        'type':'file',
                        'name':'Kedzierski',
                        'label_column':'identified'},
        
        'jung'     : {'path':'./data/plain_jung_data.csv',
                      'type':'file',
                      'name':'jung',
                      'label_column':'identified'},
        'carbery'  : {'path':'./data/known_degradation_MC.csv',
                      'type':'file',
                      'name':'carbery',
                      'label_column':'identified'},
        'carbery_loose' : {'path':'./data/loose/',
                           'type':'folder',
                           'name':'carbery_loose',
                           'label_column':'identified'},
        'Brignac'       :{'path':'./data/plain_Brignac.csv',
                          'type':'file',
                          'name':'Brignac',
                          'label_column':'ART-FT-IR polymer ID'}
                       },
    'ignore_classes'        :['Unknown',
                              'Non-plastic',
                              'N-MP',
                              'Petroleum wax',
                              'Unidentifiable',
                              'Unidentifiable ',
                              'ABS',
                              'latex',
                              'Latex',
                              'noInfo',
                              'Cellulose',
                              'phthalate',
                              'Phthalate',
                              ],
    'collect_terms'         :{'HDPE':'PE',
                              'LDPE':'PE',
                              'LLDPE':'PE',
                              'Other PE':'PE',
                              'dPE':'PE',
                              'd-PE':'PE',
                              'PE_c':'PE',
                              'PE_d':'PE',
                              'PE_f':'PE',
                              'Unknown PE':'PE',
                              'Unknown PE ':'PE',
                              'PP/PET mix':'PEPP',
                              'Mixture':'PEPP',
                              'PP_w_HDPE':'PEPP',
                              'MT1':'PEPP',
                              'MT2':'PEPP',
                              'PE/PP mix':'PEPP',
                              'PP_d':'PP',
                              'EPS':'PS',
                              'PS ':'PS',
                              'PEst':'PEstr',
                              'PETE':'PET',
                              'Poly(vinylidene fluoride)':'PVC',
                              'Cellulose Triacetate':'CA',
                              'PP/PMMA mix':'PMMA',
                              'Nylon':'PA',
                              'Nylon ':'PA',
                              'EVA wax':'EVA',
                              'PEVA':'EVA',
                              },    
    'data(%) for testing'   :0.35,
    'data(%) for validation':0.35,
    'use resizing'          :True,
    'resize size'           :4000,
    'use pooling'           :False,
    'pooling - window size' :3,
    'pooling - strides'     :1,
    'pooling - padding'     :'same',
    'label_column'          :'identified',
    'model_name'            :'Single_Run',
    'network_layers'        :{0:{"name":"preprocessing",
                                 "type":"minmax"},
                              1:{"name":"preprocessing",
                                 "type":"areaflip"},
                              2:{"name":'input',
                                 "type":'dense',
                                 "neurons":64,
                                 "activation":'relu',
                                 "init":"ones",
                                 "bias":True,
                                 "L1":True,
                                 "L2":True},
                              3:{"name":'Hidden',
                                 "type":'dense',
                                 "neurons":64,
                                 "activation":'relu',
                                 "init":"GlorotUniform",
                                 "bias":True,
                                 "L1":True,
                                 "L2":True},
                              4:{"name":'Hidden',
                                 "type":'dense',
                                 "neurons":64,
                                 "activation":'relu',
                                 "init":"GlorotUniform",
                                 "bias":True,
                                 "L1":True,
                                 "L2":True},
                              5:{"name":'Output',
                                 "type":'dense',
                                 "neurons":1,
                                 "activation":'softmax',
                                 "init":"GlorotUniform",
                                 "bias":True,
                                 "L1":False,
                                 "L2":False}},
    'training_metrics'      :[('categorical_accuracy','CAC'),
                              ('precision','PRE'),
                              ('recall','REC'),
                              ('auc','AUC')
                              ],
    'baseline_shift'        :0,
    'baseline_fix'          :'none', # airPLS, none, wavenumbermult, combo
    'balance_via_oversample':False,
    'oversampler'           :['RandomOverSampler'], #RandomOverSampler, KMeansSMOTE, SVMSMOTE, ADASYN
    'max_epochs'            :10_000,
    'steps_per_epoch'       :20, #default 1
    'batch_size'            :None, #default none
    'loss'                  :{'name':'focal',
                              'gamma':8}, # default cat. C.E. #alt: 'categorical_focal_crossentropy', aka focal loss
    'monitor_metric'        :'loss',
    'minimum_change'        :0.0001,
    'restore_top_weight'    :True,
    'verbose_monitor'       :1,
    'verbose_monitor_ES'    :1,
    'verbose_monitor_CP'    :0,
    'monitor patience'      :500,
    'start_from_epoch'      :300, #TODO not currently working because using old version of TF
    'monitor_goal'          :'min', #min
    'optimizer'             :{'learning_rate':0.01,
                              'beta_1':0.9,
                              'beta_2':0.999,
                              'epsilon':1e-07,
                              'amsgrad':False,
                              'weight_decay':None,
                              'clipnorm':None,
                              'clipvalue':None,
                              'global_clipnorm':None,
                              'use_ema':False,
                              'ema_momentum':0.99,
                              'ema_overwrite_frequency':None, 
                              'jit_compile':True, 
                              'name':'Adam'},
# =============================================================================
#     'optimizer'             :{'name': 'Adam',
#                               'learning_rate': 0.01,
#                               'decay': 0.0, #no decay, using learning rate schedueler instead
#                               'beta_1': 0.9, #default 0.9, increase to increase momentum
#                               'beta_2': 0.95, #default 0.999, decrease to increase response to gradient changes
#                               'epsilon': 1e-07,
#                               'amsgrad': False}, #amsgrad true because likely non-convex problem given highly variable final peformances
# =============================================================================
    'LR_reducer'            :{'monitor':'val_loss',
                              'mode':'min',
                              'factor':0.999, #multiply learning rate by this
                              'patience':5, 
                              'min_lr':8e-6} #starts at 2 orders of magnitude below initial LR
    }


model = network.model("ModelName")

model.setConfig(model_parameters, mode="replace")

model.ingestData()

#model.loadData()

#model.preProcessData()

#model.build()

#model.train()

#model.evaluate()


# =============================================================================
# PROBLEM: got rid of input and output defenitions, turns out i had those for valid reasons, need ot bring back.
# =============================================================================
