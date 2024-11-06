from functions.DNNModel import base_model, tf
import os

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
my_model.run_config["model"].summary()

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

my_model.train(epochs=10, batch_size=64, save_best=True)

## Evaluate the model and store under model.run_config["Evaluation"]
my_model.evaluate()

# graphs and saves plots, graphs and figures related to model performance.
my_model.graph_evaluations()

# saves relevant tables in latex format to the specified path, useful for reporting
my_model.evaluations_to_latex()
#%%
# --------------------------- EXTRAS ------------------------------------------
# We can also look at things directly, as everything is retained
#for example, lets look at a the F1 score for the carbery dataset
print(my_model.run_config["Evaluation"]["individual_datasets"]["Carbery"]["F1"])
# print the predicitons for all data
print(my_model.run_config["Evaluation"]["all_evaluation"]["prediction_values"])
# get the predicitons
predictions = my_model.run_config["Evaluation"]["all_evaluation"]["prediction_values"]
# get the column with the maximum value
predictions_maxval = predictions.argmax(1)
# replace these numbers with the text labels we are familiar with
predictions_human = [str(my_model.run_config["reverse_dict"][i]) for i in predictions_maxval]
# get the original labels
true_values = my_model.run_config["Evaluation"]["all_evaluation"]["true_values"].argmax(1)
# make these human readable as well
true_values = [str(my_model.run_config["reverse_dict"][i]) for i in true_values]

# collect pairs that are incorrect
incorrect = [tuple(sorted([i,j])) for i,j in zip(predictions_human, true_values) if i!=j]
# get unique pairs
unique_incorrect = set(incorrect)
# count these
count_dict = {}
for item in incorrect:
    if item in count_dict:
        count_dict[item] += 1
    else:
        count_dict[item] = 1