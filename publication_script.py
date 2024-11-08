from functions.DNNModel import base_model

# basic usage

## Build the model
my_model = base_model()

## Load the a configuration
my_model.load_config("./config.yaml")

## Change the model name
my_model.name = "example_model"

## Save the updated configuration
my_model.save_config("./my_config.yaml")

## Consturcts datasets, accessible via to model.processed_data
### "processed_data" is a dictionary
### this also creates model.run_config
### run_config stores everything about a run.
my_model.build_datasets()

## Stores the compiled model and prints a summary.
### the model is accessible via model.run_config["model"]
my_model.build_model()

### the model summary for example can be accessed using
#my_model.run_config["model"].summary()

## This trains the model and saves output from trianing such as checkpoints.

### save path for the checkpoints is  f'{config['unique']}/fit/{config.model_name}/checkpoint'
### For sake of example, epoch is set to 1, which overides whatever is set in the config.
### remove the epoch argument for epochs or set to None to use model config epcohs.

#### If you are training on a powerful machine, try setting batch_size to None
#### which should speed up training. setting to None on resource constrained
#### machine will create a bottleneck that significantly slows trianing.

#### Also consider setting save_best to true if on powerful machine or time is 
#### in abundance. This saves best saves the best found model, but 
#### on slower machines this puts siginficant time between early epochs.
#### so for slower machines, first run a few epochs, then when model approaches
#### convergence,set to True to begin saving model weights when new best models 
#### are found. 

#### The full model can also be saved using the command
#### my_model.run_config["model"].save("model_save_name")
#### this will create a folder in your working directory, with relevant data
#### you can load this using tf.keras.models.load_model("model_save_name")

my_model.train(epochs=100, batch_size=16, save_best=True)

## Evaluate the model and store under model.run_config["Evaluation"]
my_model.evaluate()

# graphs and saves plots, graphs and figures related to model performance.
my_model.graph_evaluations()

# saves a table in latex format to the specified path, useful for reporting
# this will show performance on holdout data as well as over individual
# datasets.
my_model.evaluations_to_latex()
#%%
my_model.do_runs(2)
