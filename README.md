# ritter-ipo
This project is an in-depth legal machine learning experiment using IPO, Initial Public Offering, data. The experiment is written in Python, and is described in detail on my blog, [Legal BI Guy](https://legalbiguy.com/2019/07/08/legal-machine-learning-experiment-part-1/)

## NOTE
This experiment is in process.  I intend to finish it by mid-August, 2019

## Overview
The business objective is to predict whether or not an IPO will be under-priced, using only the information that we have available before the offering. An offering is considered under-priced if the offering price is higher than the price at the end of the first day of trading. Given the small size of the data set and the difficulty of the business problem, our primary objective is to understand the process and techniques of a data science experiment rather than to generate a significant model.

This experiment is based on the work of Jay R. Ritter, the Joseph B. Cordell Eminent Scholar in the Department of Finance at the University of Florida, who has kindly made his data files available on-line. The data is from 1975-1984.

For a more detailed analysis, see Professor Ritter’s 1991 article in The Journal of Finance entitled The Long-Run Performance of Initial Public Offerings. This paper documents the apparent overpricing of public offerings in the long term by comparing the 1526 offerings in this dataset to a control sample.

The goal of this example is to work through a real-world legal data science experiment. Emphasis will be placed on process and techniques, rather than results. 

## Technical Notes
This experiment is written in Python. The source code includes a Jupyter notebook as well as separate Python scripts. The Jupyter notebook uses standard Python (scikit-learn) packages. The Python scripts use features included with the Microsoft Machine Learning Server version 9.3 and the revoscalepy package.

For more information on the Microsoft Machine Learning Server, see my Pluralsight course, [Scalable Machine Learning using the Microsoft Machine Learning Server](https://app.pluralsight.com/library/courses/scalable-machine-learning-microsoft-server/table-of-contents)

## How to use these files

The Jupyter notebook contains the entire experiment and does not reference any classes from thie Microsoft Machine Learning server.  This is the simplest way to run the experiment.

The python files correspond to different parts of the experiment, as detailed in the blog posting.

### Part 1: Data Preparation
Import, clean, transform and visualize data.
File: ritter_ipo_data_mls.py
The mls suffix indicates the use of Microsoft Machine Learning Server classes.  However, these references can easily be removed.
See the following blog post for a detailed description of this script:  [Ritter IPO - Part 1](https://legalbiguy.com/2019/07/08/legal-machine-learning-experiment-part-1/)

### Part 2: Feature Engineering
Use statistical tests, recursive feature elimination, indicator values, and other techniques to select features.
File: ritter_ipo_feature_mls.py
The mls suffix indicates the use of Microsoft Machine Learning Server classes.  However, these references can easily be removed.
See the following blog post for a detailed description of this script:  [Ritter IPO - Part 2](https://legalbiguy.com/2019/07/19/legal-machine-learning-experiment-part-2/)

### Part 3: Training and Evaluating a Binary Classification Model
Use logistic regression, decision trees, and random forests to train and evaluate a number of binary classification models.  Learn how to tune model hyperparameters using cross-validation and how to train an ensemble model.
File: ritter_ipo_twoclass_ensemble_mls.py
See the following blog post for a detailed description of this script:  [Ritter IPO - Part 3](https://legalbiguy.com/2019/07/28/legal-machine-learning-experiment-part-3-training-and-evaluating-a-model/)

### Part 4: Implementing the End-to-End Experiment in SQL Server
In process.
