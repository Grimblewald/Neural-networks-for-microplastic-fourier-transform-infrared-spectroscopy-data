import yaml
import functions.ftirmlfunctions as ff
import functions.ftirmltraining as ft
import functions.ftirmlplotting as fp
import tensorflow as tf
import os

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


class base_model:
    def __init__(self):
        self.config = None
        self.processed_data = {}
        self.dnn = None
        self.run = 0
        self.root_path = "./"
        self.run_path = ""
        self.run_config = {}
        self.config_history = {}
        
    def run_update(self):
        if f"run_{self.run}_history" in self.config_history:
            self.config_history[f"run_{self.run}_history"].update(self.run_config.copy())
        else:
            self.config_history[f"run_{self.run}_history"] = self.run_config.copy()

    def load_config(self, config_file):
        try:
            with open(config_file, "r") as f:
                self.config = yaml.load(f,Loader=PrettySafeLoader)
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
        self.run_config = self.config.copy()
        
    def build_datasets(self):
        if not os.path.exists(self.config['unique']+"paretocharts/"):
            os.mkdir(self.config['unique']+"paretocharts/")
        self.run_path = f"{self.config['unique']}/run_{self.run}/"
        
        self.init_newrun()
        self.processed_data[f"run_{self.config['current_run']}"] = ff.create_datasets(self.config.copy())
        self.run_config.update(self.processed_data[f"run_{self.config['current_run']}"])
        self.run_update()
        
    def build_model(self):
        self.run_config["model"] = ft.make_model(self.run_config)
        self.run_config["model_name"] += f"_run_{str(self.run)}"
        self.run_update()
    
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
        plot_folder = f"{self.config['unique']}/EvalPlots/"
        if not os.path.exists(plot_folder):
            try:
                cmsavepath = plot_folder
                os.mkdir(plot_folder)
            except:
                print("threshold plot folder already exists for this run")
                
        fp.cm_plotter(self.config_history, plot_folder, self.run)
        fp.metric_boxplots(self.config_history, 
                           plot_folder)
        #fp.tsne_plots(self.config_history)
        #fp.plot_multi_ROC(y_true_bin, y_pred_bin, n_classes, savename=f"ROCgrid_{run}.svg")
    
    def evaluations_to_latex(self):
        fp.eval_to_latex(self.config_history)
    
    def do_runs(self, number_of_runs):
        for i in range(number_of_runs):
            self.build_datasets()
            self.build_model()
            self.train()
            self.evaluate()
            self.graph_evaluations()
