from functions.DNNModel import base_model

# basic usage

## Build the model
my_model = base_model()

## Load the a configuration
my_model.load_config("./config.yaml")

## Change a configuration value
my_model.config["model_name"] = "Example_Model"

## Save the updated configuration
my_model.save_config("./my_config.yaml")

# The following will train a number of models as specified in the config
# file, or for the interger amount you pass as an argument, overiding the
# amount speficied in the configuration file.
# this will build datasets, initialize a new model, train it, evaluate it,
# plot evaluations, and save latex tables for evaluations.

# my_model.do_runs() #uncomment this line to simply run everything.

#------------------------------------------------------------------------------
# Altrantively, you could take the steps the `do_runs()` funciton takes
# Manually.

## Consturcts datasets, accessible via to model.processed_data
### "processed_data" is a dictionary
### this also creates model.run_config
### run_config stores everything about a run.
my_model.build_datasets()

## Stores the compiled model and prints a summary.
### the model is accessible via model.run_config["model"]
my_model.build_model()

### the model summary for example can be accessed using
my_model.run_config["model"].summary()

# The following trains the current run's model further and saves output from 
# training such as checkpoints to that run's folder.

### save path for the checkpoints is  f'{config['unique']}/fit/{config.model_name}/checkpoint'
### For sake of example, epoch is set to 10, which overides whatever is set in the config.
### remove the epoch argument for epochs or set to None to use model config epcohs.

#### If you are training on a powerful machine, try setting batch_size to None
#### which should speed up training. setting to None on resource constrained
#### machine will create a bottleneck that significantly slows trianing.

#### Also consider setting save_best to true if on powerful machine or time is 
#### in abundance. This saves the best model after each epoch, but 
#### on slower machines this puts siginficant time between early epochs.
#### so for slower machines, first run a few epochs, then when model approaches
#### convergence,set to True to begin saving model weights when new best models 
#### are found. 

#### The current model can also be saved using the command
#### my_model.run_config["model"].save("model_save_name")
#### this will create a folder in your working directory, with relevant data
#### you can load this using tf.keras.models.load_model("model_save_name")
my_model.train(epochs=2, batch_size=320, save_best=True)

# You can evaluate the current model using
my_model.evaluate()

# You can then graph the current evaluations using
my_model.graph_evaluations()

# You can also get latex formatted tables for your results using
my_model.evaluations_to_latex()
