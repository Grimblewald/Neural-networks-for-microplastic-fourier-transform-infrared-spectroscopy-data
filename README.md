# Work in progress
There are elements of this repository which are clearly unfinished, these are things that can only be finished after the peer-review process concludes and the related work is published. Currently, this is made public to make it accessible to peer-reviewers.

# Neural Networks for Microplastic Fourier Transform Infrared Spectroscopy Data

This repository provides code for training artificial neural networks, with a specific focus on models that support automated classification of Fourier Transform Infrared (FTIR) spectroscopic data related to microplastics. 

The easiest way to run this code and explore the repository is through the google colab notebook found [here](https://colab.research.google.com/drive/17tDtN3pFHYkQCpytrv1TYdfoJkn8snik?usp=sharing), however this was written in a way where it should also run in jupyter with minimal editing (widgets were coded so they don't rely on colab specific features). The repository also includes example output, as produced by `publication_script.py` in the folder `trianingoutput`.

The code in this repository compliments the research published in **[Link to publication]**. It represents a snapshot of the work at the time of publication and may not reflect the latest developments. For ongoing updates, we recommend visiting the actively maintained version of this repository at **[Link to active repository]**. Additionally, we welcome discussions, support, feedback, and community engagement through our **[Community Link]**.

## Code Overview

The code was initially developed with a functional programming approach but has been adapted here to an object-oriented paradigm. To use it, follow the methods outlined below.

The main class is `DNNModel`, which provides a set of methods to configure, train, and evaluate models:

- **`load_config`**  
  Loads a configuration file in YAML format. An example is included in this repository, and you can also generate a custom YAML config file at **[Link to YAML generator]**.

- **`save_config`**  
  Saves your current configuration, including models, results, and processed datasets. Be cautious, as this will overwrite existing files if not specified otherwise.

- **`build_datasets`**  
  Prepares datasets according to the specified configurations for ingestion by the model.

- **`build_model`**  
  Compiles the model architecture, preparing it for training or loading weights.

- **`set_model_weights`**  
  Allows loading model weights.

- **`save_model`**  
  Saves the model to the specified path.

- **`load_model`**  
  Loads a model from a specified path.

- **`train`**  
  Trains the model on the data designated for training.

- **`evaluate`**  
  Evaluates the model's performance across all data.

- **`graph_evaluations`**  
  Generates and saves evaluation-related plots to the results folder.

- **`evaluations_to_latex`**  
  Exports evaluation results as a LaTeX-formatted table in the current run folder.

- **`do_runs`**  
  Conducts multiple runs in sequence, with each run executing `build_dataset`, `build_model`, `train`, `evaluate`, and `graph_evaluations` steps for reliability assessment.

## Data

The project utilizes publicly accessible data, though specific datasets may be limited in availability for direct inclusion. We provide an example dataset created with data from environmental and controlled sources, curated to approximate the variance in public datasets. This example data is sufficient for training a model with reasonable performance but is not a replacement for the original data sources. Users are encouraged to download the original datasets and format them as CSV files following project conventions for optimal results.

### Data Formatting Assumptions

When using your own data:
- Wavenumber columns (representing intensities) must use numeric headers, while non-numeric columns should have text headers.
- Data labels must be included in a designated column.
  
If using publicly available data referenced (with the exception of OpenSpeccy), the provided formats should work. For custom datasets, specify the target column and apply any necessary relabeling rules, either directly or by specifying these in the config.

## Requirements

Python version 3.10–3.12 is required (untested in later versions). The code utilizes `match-case` syntax, which was introduced in Python 3.10. Required packages include:
- `spyder`, `tensorflow`, `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `seaborn`, and `scipy`.

## Quick Start
The fastest way to get started and inspect the code output is to simply run the google collab notebook we make avaliable via this [link](https://colab.research.google.com/drive/17tDtN3pFHYkQCpytrv1TYdfoJkn8snik?usp=sharing)
## Installation

We recommend creating a virtual environment to install the necessary packages. An installation script is provided, which you can use to set up a virtual environment and install relevent files. This script will work after you set up Python, `pip`, and `venv`. Python must be on path.

### Installation Instructions
1. Download and install python. We assume Non windows users know how to find and install python. Windows users can use the following links to download a [32 bit installer](https://www.python.org/ftp/python/3.11.9/python-3.11.9.exe) or a [64 bit installer](https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe).
2. Download and install git (or download this repository manually)
4. Once installed, open a python terminal 
5. Navigate to the folder where the repositroy is stored, or will be stored
6. If you haven't downloaded the repository, do so using `git clone <link>`
7. use `cd <path>` to navigate into the repository folder
8. install requirements using `pip install -r requirements.txt`
9. launch spyder by entering `spyder` into terminal
10. open the script `publication_script.py`
11. Run this, either line by line or as a whole.
12. Explore results either within spyder, or in the output folder.

## Usage

To use this code interactively, we recommend running it in Spyder:

1. Open the file `publication_script.py` in Spyder.
2. Click "Run" in the toolbar, or press F5. This will execute the entire script, displaying plots in the Spyder plots panel and saving them to the output folder.

Or simply run methods as required.

# Referencing

To reference this work, please use {ref block}
