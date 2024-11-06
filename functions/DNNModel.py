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


class base_model:
    def __init__(self):
        self.config = None
        self.processed_data = {}
        self.dnn = None
        self.run = 0
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

    def build_datasets(self):
        self.run += 1
        if not os.path.exists(self.config['unique']):
            os.mkdir(self.config['unique'])
        self.run_path = f"{self.config['unique']}/run_{self.run}/"
        if not os.path.exists(self.run_path):
            os.mkdir(self.run_path)
            
        self.processed_data[f"run_{self.config['current_run']}"] = ff.create_datasets(self.config.copy())
        self.run_config.update(self.processed_data[f"run_{self.config['current_run']}"])
        self.run_update()
        
    def build_model(self):
        self.run_config["model"] = ft.make_model(self.run_config)
        self.run_config["model_name"] += f"_run_{str(self.run)}"
        self.run_update()
        
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
        print("saved evaluation metrics for individual datasets as latex table to {self.config['unique']}/run_{self.run}")
        self.run_update()

    def graph_evaluations(self):
        plot_folder = f"{self.run_path}confusionmatrixplots/"
        if not os.path.exists(plot_folder):
            try:
                cmsavepath = plot_folder
                os.mkdir(plot_folder)
            except:
                print("threshold plot folder already exists for this run")
        fp.cm_plotter(self.config_history, plot_folder, self.run)
    
    def evaluations_to_latex(self):
        save_path =  f"{self.run_path}/"
        fp.eval_to_latex(self.config_history, save_path)
    
    def do_run(self):
        self.build_dataset()
        self.build_model()
        self.train()
        self.evaluate()
        self.graph_evaluations()
