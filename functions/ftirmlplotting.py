# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 15:03:48 2024

@author: frith
"""
from functions.ftirmlfunctions import create_datasets, confusion_matrix_plot, calculate_average_model_data
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy import stats as stat

def cm_plotter(config_history, cmsavepath, run):
    model_parameters = config_history
    j = ("Holdout ", "holdout_evaluation", "Y_test")
    for i in [None, "true", "pred"]:
        confusion_matrix_plot(
            model_parameters,
            cmsavepath,
            "all",
            title=f"Data Average over Models",
            Y_evaluation=j[1],
            Graph_title_prefix=j[0],
            Y_true=j[2],
            figure_size=(10, 10),
            subtitles=True,
            normalise=i,
            image_dpi=300,
            save=True,
            filepath=cmsavepath,
            filename=f"CM_for_{j[0]}_Normby_{i}",
            saveformat="svg",
            fontscale=1.5,
            colourmap=sns.diverging_palette(
                250, 30, l=65, center="dark", as_cmap=True
            ),
            rotate_xy=(45, 45),
            savecm=True,
            cmpath=f"{cmsavepath}CM_for_{j[0]}_{i}.csv",
        )
        
def get_split_metrics_dict(config_history, runs):
    model_data = config_history
    datasetkeys = list(model_data[runs[0]]["Evaluation"]["individual_datasets"].keys())
    metrics = list(model_data[runs[0]]["Evaluation"]["individual_datasets"][datasetkeys[0]].keys())
    metrics = [i for i in metrics if i not in ['true_values', 'prediction_values']]
    splitmetricsdict = {}
    
    for run in runs:
        model_version = run
        splitmetricsdict[model_version] = {}
        for datakey in datasetkeys:
            splitmetricsdict[model_version][datakey] = {}      
            for metrickey in metrics:
                splitmetricsdict[model_version][datakey][metrickey] = model_data[run]["Evaluation"]['individual_datasets'][datakey][metrickey]
            splitmetricsdict[model_version]['Holdout'] = {metrickey:model_data[run]["Evaluation"]['holdout_evaluation'][metrickey] for metrickey in metrics}
    
    # Create a DataFrame to store the aggregated metrics
    aggregated_metrics_data = []

    for datakey in datasetkeys + ['Holdout']:
        row_data = {'Dataset': datakey}
        for metrickey in metrics:
            metric_values = [splitmetricsdict[model_version][datakey][metrickey] for model_version in splitmetricsdict.keys()]
            row_data[metrickey + '__Median'] = np.median(metric_values)
            row_data[metrickey + '__Min'] = min(metric_values)
            row_data[metrickey + '__Max'] = max(metric_values)
            row_data[metrickey + '__MAD'] = stat.median_abs_deviation(metric_values)
        aggregated_metrics_data.append(row_data)

    aggregated_metrics = pd.DataFrame(aggregated_metrics_data)

    # Set 'Dataset' as the index
    aggregated_metrics.set_index('Dataset', inplace=True)

    # Create a multi-level column index
    aggregated_metrics.columns = pd.MultiIndex.from_tuples([(col.split('__')[0], col.split('__')[1]) for col in aggregated_metrics.columns])
    average_metrics = aggregated_metrics
    return splitmetricsdict, average_metrics
    
def eval_to_latex(config_history):
    model_data = config_history
    root_path = config_history[list(config_history.keys())[0]]["unique"]
    runs = list(model_data.keys())
    
    splitmetricsdict, aggregated_metrics = get_split_metrics_dict(config_history, runs)
    
    # Save individual dataset latex tables
    for run in runs:
        df = pd.DataFrame(splitmetricsdict[run])
        run_path = run.replace("_history","")
        # Transpose the DataFrame to get datasets as columns and metrics as rows
        df = df.transpose()
        # Replace underscore with escaped underscore in column names
        df.columns = df.columns.str.replace("_", " ")
        
        #df = round(df,2)
        
        # Determine the number of columns in your DataFrame
        num_columns = len(df.columns)
    
        # Generate a column format string: one 'c' for each column, plus an extra 'l' for the index column if you want it included
        column_format = 'l' + 'c' * num_columns  # 'l' for the index column, 'c' for each data column
        
        # Convert the DataFrame to a LaTeX table
        latex_table = df.to_latex(float_format="{:0.2E}".format, column_format=column_format)
        
        # Print the LaTeX table
        #print(latex_table)
        
        # Open the file in write mode
        with open(
            root_path + f"{run_path}/{run_path}_performance_over_individual_datasets.tex", "w"
        ) as file:
            # Write a message to the file
            file.write(latex_table)


    # Save latex table for average
    
    # Round the values to 2 decimal places
    #aggregated_metrics = aggregated_metrics.round(2)

    # Replace underscore with escaped underscore in column names
    aggregated_metrics.columns = aggregated_metrics.columns.map(lambda x: (x[0].replace("_", " "), x[1]))
    aggregated_metrics.index = aggregated_metrics.index.str.replace("_", " ")
    
    # Convert the DataFrame to a LaTeX table
    latex_table = aggregated_metrics.to_latex(float_format="{:0.2E}".format, 
                                              multicolumn_format='c', 
                                              column_format='l' + 'c' * len(aggregated_metrics.columns), 
                                              escape=False)

    # Print the LaTeX table
    #print(latex_table)

    # Open the file in write mode
    with open(f"{root_path}average_metrics_table.tex", "w") as file:
        # Write the LaTeX table to the file
        file.write(latex_table)
            
def metric_boxplots(config_history, savelocation):
    runs = list(config_history.keys())
    run_metrics_list = []
    splitmetricsdict,_ = get_split_metrics_dict(config_history, runs)
    
    data_for_plotting = []

    # Introduced to make setting of things people want changed a bit easier
    SMALL_SIZE = 20
    MEDIUM_SIZE = 24
    BIGGER_SIZE = 36
    
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    # Iterate through each model version in the dictionary
    for model_version, datasets in splitmetricsdict.items():
        for dataset_name, metrics in datasets.items():
            for metric_name, metric_value in metrics.items():
                data_for_plotting.append({
                    "Model Version": model_version,
                    "Dataset": dataset_name,
                    "Metric": metric_name,
                    "Value": metric_value
                })

    # Convert the list of dictionaries into a DataFrame
    df = pd.DataFrame(data_for_plotting)

    # Filter out 'Holdout' if you want it as a separate category or leave it for comparison
    # df = df[df['Dataset'] != 'Holdout']

    # Define the order of datasets as you prefer
    datasets_order = ['OpenSpeccy', 'Primpke', 'Carbery', 'Kedzierski', 'Brignac', 'Jung', 'Holdout'] 

    # Plotting
    fig, axs = plt.subplots(2, 3, figsize=(18, 12))  # Adjust the size as needed
    axs = axs.flatten()  # Flatten the array for easy iteration

    # Define the order of metrics based on your preference
    metrics_order = ['categorical_accuracy', 'f1_score_weighted', 'precision', 'recall', 'auc', 'categorical_crossentropy']

    # Iterate through each metric and create a boxplot
    for idx, metric in enumerate(metrics_order):
        ax = axs[idx]
        sns.boxplot(data=df[df['Metric'] == metric], 
                    x='Dataset', y='Value', 
                    ax=ax,
                    boxprops={"facecolor": (0.4, 0.6, 0.8, 0.0)},
                    medianprops={"color": "coral"},
                    order=datasets_order)
        
        match metric:
            case 'categorical_accuracy':
                ax.set_title("Categorical Accuracy")
            case 'f1_score_weighted':
                ax.set_title("F1")
            case 'precision':
                ax.set_title("Precision")
            case 'recall':
                ax.set_title("Recall")
            case 'auc':
                ax.set_title("AUC")
            case 'categorical_crossentropy':
                ax.set_title("Categorical Cross Entropy")
            case _:
                ax.set_title("error, no title")
                
        if metric !='categorical_crossentropy':
            ax.set_ylim((-0.1,1.1))

        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")  # Rotate x labels for better readability

    # Adjust layout
    plt.tight_layout()

    # Save the figure
    print(f"saving boxplots for metrics over the runs to {savelocation}")
    plt.savefig(f"{savelocation}metrics_boxplots.png", dpi=300)
    plt.savefig(f"{savelocation}metrics_boxplots.svg")
    plt.show()
    
# Needs adapting, probably need to bring
# the preds/true_labs back into split dict
# and address the issues this will raise in
# other functions
# =============================================================================
# def tsne_plots(config_history, savelocation):
#     #config_history = my_model.config_history
#     keys = list[config_history.keys()]
#     perf = []
#     for i in keys:
#         perf.append(config_history[i]["Evaluation"]["all_evaluation"]["F1"])
# =============================================================================

# Needs adapting
# =============================================================================
# def plot_multi_ROC(y_true_bin, y_pred_bin, n_classes, savename):
#     def letter_generator():
#         letters = list(string.ascii_uppercase)
#         for letter in letters:
#             yield letter
# 
#     # Create a generator
#     letter_gen = letter_generator()
# 
#     # Compute ROC curve and ROC area for each class
#     fpr = dict()
#     tpr = dict()
#     class_label = dict()
#     roc_auc = dict()
#     for i in range(n_classes):
#         class_label[i] = model_parameters["label_dictionary"]
#         fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_bin[:, i])
#         roc_auc[i] = auc(fpr[i], tpr[i])
# 
#     # Create a figure and a grid of subplots
#     ncols = 4
#     nrows = int(np.ceil(n_classes / ncols))
#     fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 2 * nrows))
# 
#     # Flatten the array of axes, for easier iteration
#     ax = ax.flatten()
# 
#     colors = cycle(["aqua", "darkorange", "cornflowerblue"])
#     for i, color, letter in zip(range(n_classes), colors, letter_gen):
#         classlabel = reverse_dict[i]
#         ax[i].set_xlim([-0.01, 1.01])
#         ax[i].set_ylim([-0.01, 1.05])
#         ax[i].plot(
#             fpr[i],
#             tpr[i],
#             color=color,
#             lw=4,
#             label="ROC for {0}\n(area = {1:0.2f})" "".format(classlabel, roc_auc[i]),
#         )
# 
#         ax[i].plot([0, 1], [0, 1], "k--", lw=2)
#         ax[i].set_xlabel("False Positive Rate")
#         ax[i].set_ylabel("True Positive Rate")
#         ax[i].set_title("OvM ROC for {}".format(classlabel))
#         ax[i].legend(loc="lower right")
# 
#         # panel labels
#         textycoord = (
#             ax[i].get_ylim()[1] + (ax[i].get_ylim()[1] - ax[i].get_ylim()[0]) / 10
#         )
#         ax[i].text(
#             0,
#             textycoord,
#             letter,
#             ha="center",
#             backgroundcolor="w",
#             bbox=dict(facecolor="white", alpha=0.25),
#         )
# 
#     # Remove empty subplots
#     if n_classes % ncols != 0:
#         for ax in ax[n_classes:]:
#             fig.delaxes(ax)
# 
#     plt.tight_layout()
#     plt.savefig(
#         fname=outputfolder + savename,
#         transparent=True,
#         bbox_inches="tight",
#         pad_inches=0.1,
#     )
#     plt.show()
# =============================================================================
