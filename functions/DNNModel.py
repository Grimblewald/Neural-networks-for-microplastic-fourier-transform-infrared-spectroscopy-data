import yaml
import functions.ftirmlfunctions as ff
import functions.ftirmltraining as ft
import functions.ftirmlplotting as fp
import tensorflow as tf
import os
from copy import deepcopy

class PrettySafeLoader(yaml.SafeLoader):
    def construct_python_tuple(self, node):
        return tuple(self.construct_sequence(node))

PrettySafeLoader.add_constructor(
    u'tag:yaml.org,2002:python/tuple',
    PrettySafeLoader.construct_python_tuple)

class LoadError(Exception):
    pass

class SaveError(Exception):
    pass

class RunError(Exception):
    pass

class base_model:
    def __init__(self):
        self.config = None
        self.config_original = None
        self.processed_data = {}
        self.dnn = None
        self.run = 0
        self.root_path = "./"
        self.run_path = ""
        self.run_config = {}
        self.config_history = {}
        
    def run_update(self):
        if f"run_{self.run}_history" in self.config_history:
            self.config_history[f"run_{self.run}_history"].update(deepcopy(self.run_config))
        else:
            self.config_history[f"run_{self.run}_history"] = deepcopy(self.run_config)

    def load_config(self, config_file):
        try:
            with open(config_file, "r") as f:
                self.config = yaml.load(f,Loader=PrettySafeLoader)
                self.config_original = deepcopy(self.config)
        except FileNotFoundError:
            print(f"Error: Configuration file '{config_file}' not found.")
        except yaml.YAMLError as e:
            print(f"Error: Failed to parse configuration file '{config_file}': {e}")

    def save_config(self, config_file):
        try:
            with open(config_file, "w") as f:
                yaml.dump(self.config, f, default_flow_style=False)
        except OSError as e:
            print(f"Error: Failed to write configuration file '{config_file}': {e}")
        
    def init_newrun(self):
        self.run += 1
        if not os.path.exists(self.config['unique']):
            os.mkdir(self.config['unique'])
        self.run_path = f"{self.config['unique']}/run_{self.run}/"
        if not os.path.exists(self.run_path):
            os.mkdir(self.run_path)
        self.run_config = deepcopy(self.config)
        self.run_config["model"] = None
        
    def build_datasets(self, init_newrun=True):
        if not os.path.exists(self.config['unique']+"paretocharts/"):
            os.mkdir(self.config['unique']+"paretocharts/")
        self.run_path = f"{self.config['unique']}/run_{self.run}/"
        
        if init_newrun:
            self.init_newrun()
            print(f"\nInitiated a new run, run index is set to {self.run}")
        self.processed_data[f"run_{self.config['current_run']}"] = ff.create_datasets(deepcopy(self.config))
        self.run_config.update(self.processed_data[f"run_{self.config['current_run']}"])
        self.run_update()
        print("\nDatasets loaded, preprocessed, split into specified sets, and are ready for use\n")
        
    def build_model(self):
        print("building model...")
        if self.run_config["model"] == None:
            self.run_config["model"] = ft.make_model(self.run_config)
            self.run_config["model_name"] += f"_run_{str(self.run)}"
            self.run_update()
            print("\nModel built, model summary provided above\n")
        else:
            print("\nModel already built, skipping instructions. If you wish to train a new model, initiate a new run\n")
    def set_model_weights(self, weights_path):
        self.build_model(self)
        self.run_config["model"].load_model.load_weights(weights_path)
        
    def save_model(self,model_path):
        try:
            self.run_config["model"].save(model_path)
        except:
            raise SaveError("Uh oh! Looks like maybe you haven't made your model yet?")
            
    def load_model(self,model_path):
        try:
            self.run_config["model"] = tf.keras.models.load_model(model_path)
        except:
            raise LoadError("Uh oh! Looks like maybe you haven't built your data?")
            
    def train(self, epochs=None, batch_size=None, save_best=None):
        if epochs:
            self.run_config["max_epochs"] = epochs
        if batch_size:
            self.run_config["batch_params"]["batch_size"] = batch_size
        if save_best != None:
            self.run_config["checkpoints"]["SaveCheckpoints"] = save_best
        model, training_history = ft.train_model(self.run_config)
        self.run_config["model"] = model
        self.run_config["training history"] = training_history
        self.run_update()

    def evaluate(self):
        self.run_config = ft.evaluate_model(self.run_config)
        self.run_update()

    def graph_evaluations(self):
        plot_folder = f"{self.config['unique']}/run_{self.run}/EvalPlots/"
        if not os.path.exists(plot_folder):
            try:
                os.mkdir(plot_folder)
            except:
                print("threshold plot folder already exists for this run")
                
        print("\nGraphing confusion matrix")
        fp.cm_plotter(self.config_history, plot_folder, self.run)
        
        print("\nGraphing performance metrics box plots")
        fp.metric_boxplots(self.config_history, 
                           plot_folder)
        
        #print("\nStarting t-SNE plotting...") #TODO - adapt for this
        #fp.tsne_plots(self.config_history)
        
        #print("\nGraphing ROC curves") #TODO - adapt for this
        #fp.tsne_plots(self.config_history)
        #fp.plot_multi_ROC(y_true_bin, y_pred_bin, n_classes, savename=f"ROCgrid_{run}.svg")
        print(f"saved all plots and related data to {plot_folder}")
    
    def evaluations_to_latex(self):
        fp.eval_to_latex(self.config_history)
    
    def do_runs(self, number_of_runs=None):
        if type(number_of_runs)==int:
            number_of_runs = number_of_runs
        elif number_of_runs == None:       
            number_of_runs = self.config["models_to_train"]
        elif True:
            raise RunError("Invalid number of runs set, try using 'None' to use config, or some integer amount instead.")
        for i in range(number_of_runs):
            self.build_datasets()
            self.build_model()
            self.train()
            self.evaluate()
            self.graph_evaluations()
