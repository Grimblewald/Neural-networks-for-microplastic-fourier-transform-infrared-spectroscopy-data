# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 15:03:48 2024

@author: frith
"""
from functions.ftirmlfunctions import create_datasets, confusion_matrix_plot, calculate_average_model_data
import seaborn as sns
import pandas as pd

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

def eval_to_latex(config_history, savelocation):
    model_data = config_history
    runs = list(model_data.keys())
    datasetkeys = list(model_data[runs[0]]["Evaluation"]["individual_datasets"].keys())
    metrics = list(model_data[runs[0]]["Evaluation"]["individual_datasets"][datasetkeys[0]].keys())
    splitmetricsdict = {}
    
    
    for run in runs:
        model_version = run
        splitmetricsdict[model_version] = {}
        for datakey in datasetkeys:
            splitmetricsdict[model_version][datakey] = {}      
            for metrickey in metrics:
                splitmetricsdict[model_version][datakey][metrickey] = model_data[run]["Evaluation"]['individual_datasets'][datakey][metrickey]
            splitmetricsdict[model_version]['Holdout'] = {metrickey:model_data[run]["Evaluation"]['holdout_evaluation'][metrickey] for metrickey in metrics}
    
    
        df = pd.DataFrame(splitmetricsdict[model_version])
        
        # Transpose the DataFrame to get datasets as columns and metrics as rows
        df = df.transpose()
        df.pop('true_values')
        df.pop('prediction_values')
        # Replace underscore with escaped underscore in column names
        df.columns = df.columns.str.replace("_", " ")
        
        df = round(df,2)
        
        # Determine the number of columns in your DataFrame
        num_columns = len(df.columns)
    
        # Generate a column format string: one 'c' for each column, plus an extra 'l' for the index column if you want it included
        column_format = 'l' + 'c' * num_columns  # 'l' for the index column, 'c' for each data column
        
        # Convert the DataFrame to a LaTeX table
        latex_table = df.to_latex(float_format="{:0.2f}".format, column_format=column_format)
        
        # Print the LaTeX table
        print(latex_table)
        
        # Open the file in write mode
        with open(
            savelocation + f"performance_over_individual_datasets.tex", "w"
        ) as file:
            # Write a message to the file
            file.write(latex_table)