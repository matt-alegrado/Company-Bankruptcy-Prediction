# Company Bankruptcy Prediction: Early Warning System for Corporate Distress

ECE 143 - Group 14

Team Members: Kendra Chen, Zhuosheng Song, Matthew Alegrado, Jinglin Cao

This notebook presents a comprehensive analysis of Company Bankruptcy Prediction in a Taiwan dataset, investigating financial indicators that signal corporate distress. The objective is to construct a robust predictive pipeline capable of identifying firms at elevated risk of bankruptcy.

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
To test on real data, run with `python early_warning_system.py [input_path.csv]` to see which companies are flagged, or simply run with `python early_warning_system.py` to test with sample instances. 
