# Company Bankruptcy Prediction: Early Warning System for Corporate Distress

ECE 143 - Group 14

Team Members: Kendra Chen, Zhuosheng Song, Matthew Alegrado, Jinglin Cao

This notebook presents a comprehensive analysis of Company Bankruptcy Prediction in a Taiwan dataset, investigating financial indicators that signal corporate distress. The objective is to construct a robust predictive pipeline capable of identifying firms at elevated risk of bankruptcy.

Access the [**video presentation**](https://youtu.be/kfGxHz3kUjc) through the hyperlink. 

## Primary Task

**Main task**: Build a model pipeline to assess company distress using financial indicators.

We propose a pipeline consisting of PCA-based feature engineering and selection, followed by VIF pruning to remove multicollinearity, then training a classification model to determine whether a company is likely to experience distress given its current business statistics. The pipeline also derives interpretable decision thresholds with SHAP analysis and monitoring rules based on influential financial indicators and visualizes model explanations.

## Dataset

**Source**: Company Bankruptcy Prediction

**URL**: [https://www.kaggle.com/datasets/fedesoriano/company-bankruptcy-prediction/data](https://www.kaggle.com/datasets/fedesoriano/company-bankruptcy-prediction/data)

**References**:
- [Bankruptcy Detection](https://www.kaggle.com/code/marto24/bankruptcy-detection)

## Requirements
This project requires Python 3.10. The required modules are listed in `requirements.txt`.

### Setting Up the Conda Environment

Follow these steps to create the project environment and install all required dependencies.

#### 1. Create and Activate the Conda Environment (Python 3.10)

```bash
conda create -n Bankruptcy_env python=3.10
conda activate Bankruptcy_env
```
#### 2. Install Depencencies
```bash
pip install -r requirements.txt
```

#### 3. Set up Jupyter Kernel
```bash
python -m ipykernel install --user --name=Bankruptcy_env --display-name "Python Bankruptcy_env"
```
Then select the kernel "Python Bankruptcy_env" to run the notebook in the conda environment.

## How to Run
Simply run our notebook, `visualization.ipynb`, to generate the results and plots locally.

For Early Warning System

File Mode (Batch Processing)

Use the --file argument to process a CSV file containing company data. The script will automatically clean the column names, run the prediction, and save the full results and flagged records to the output/ directory.

Command:

```bash
python early_warning_system.py --file <path/to/your/input_data.csv>
```

Interactive Mode (Single Record)

Use the -i or --interactive flag to enter a mode where you are prompted to manually enter the feature values for a single record. The script will output the prediction and the detailed reason directly to the console.

Command:

```bash
python early_warning_system.py -i
# OR
python early_warning_system.py --interactive
```

Sample Usage
   
If you run the script without any arguments, it will run on the sample instances.

## Limitations
Our current model performance shows only a marginal improvement over random guessing (AUC $\approx 0.5$). This sub-optimal result is due to several intrinsic challenges in financial distress prediction: Extreme class imbalance (bankruptcy is rare), high non-linearity and correlation among financial features, and the data's inability to account for critical external macroeconomic shocks or internal operational events that are often the true catalysts for failure. Future work will focus on time-series analysis and cost-sensitive learning to overcome these hurdles.

