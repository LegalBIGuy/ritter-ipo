# Ritter IPO Analysis
#   Step 2: Feature Engineering
#
#   This script uses the version of Python bundled with the
#     Microsoft Machine Learning Server version 9.3
#
#   This project is an example end-to-end data science experiment
#     and is not intended to generate meaningful results
#   
#   This file depends on XDF files generated by ritter_ipo_data_mls.py
#
#   Please see the Jupyter notebook for a version of this experiment
#     which does not use the Machine Learning Server

import numpy as np
import pandas as pd
from revoscalepy import rx_import, rx_dtree, rx_data_step
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, mutual_info_classif
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler

# Read Cleaned Ipo2609 XDF file
ipo2609Cleaned = rx_import("IPO2609Cleaned.xdf")

# Convert odate to YYYYMM
ipo2609Cleaned['odate'] = ipo2609Cleaned['odate'].astype(str).str[:4].astype("int")

# Univariate Feature Selection
uni_vars=ipo2609Cleaned.columns.values.tolist()

uni_vars.remove('underpriced') # The value we are predicting, so remove from factors
uni_vars.remove('Name') # Company identifier, not relevant
uni_vars.remove('perm') # permanent identifier, not relevant
uni_vars.remove('dt1') # Price after first day trading, unknown at time of prediction
uni_vars.remove('pr1') # Closing Bid after first day trading, unknown at time of prediction
uni_vars.remove('d1pctchg') # Price percent change after first day trading, unknown at time of prediction
uni_vars.remove('ex') # use numeric ex_code
uni_vars.remove('t') # Use numeric t_code
uni_X = ipo2609Cleaned[uni_vars]
uni_y = ipo2609Cleaned['underpriced']

#apply SelectKBest class to extract top 10 best features
# Use mutual_info_classif
bestfeatures = SelectKBest(score_func=mutual_info_classif,k=15)
fit = bestfeatures.fit(uni_X, uni_y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(uni_X.columns)
featureScores = pd.concat([dfcolumns, dfscores], axis=1)
featureScores.columns = ['Specs', 'Score']
print(featureScores.nlargest(15, 'Score'))

# Scale and use chi2
scaler = MinMaxScaler() 
scaled_values = scaler.fit_transform(uni_X) 
scaled_X = uni_X.copy()
scaled_X.loc[:,:] = scaled_values

bestfeatures = SelectKBest(score_func=chi2,k=15)
fit = bestfeatures.fit(scaled_X, uni_y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(scaled_X.columns)
featureScores = pd.concat([dfcolumns, dfscores], axis=1)
featureScores.columns = ['Specs', 'Score']
print(featureScores.nlargest(15, 'Score'))


# Look for features with low variance
sel = VarianceThreshold(1)
sel.fit_transform(uni_X, uni_y)
# Review retained features (all included)
sel.get_support(indices=True)


# Feature Importance
model = ExtraTreesClassifier(n_estimators=100)
model.fit(uni_X, uni_y)
feat_importances = pd.Series(model.feature_importances_, index=scaled_X.columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.show()

# Use revoscalepy to build a decision tree to predict underpriced with the full dataset
#  in order to plot the most important featuresn -- compare to ExtraTreesClassifier
model = rx_dtree("underpriced ~" + "+".join(uni_vars), data=ipo2609Cleaned, method="anova",
                 importance=True)
importance = model.importance
importance.columns = ["feature importance"]
importance.sort_values("feature importance").plot(kind="bar")
plt.show()


# Categorical Values and Recursive Feature Elimination (RFE)
rfe_vars=ipo2609Cleaned.columns.values.tolist()

rfe_vars.remove('underpriced') # The value we are predicting, so remove from factors
rfe_vars.remove('Name') # Company identifier, not relevant
rfe_vars.remove('perm') # permanent identifier, not relevant
rfe_vars.remove('dt1') # Price after first day trading, unknown at time of prediction
rfe_vars.remove('pr1') # Closing Bid after first day trading, unknown at time of prediction
rfe_vars.remove('d1pctchg') # Price percent change after first day trading, unknown at time of prediction
rfe_vars.remove('ex') # use numeric ex_code
rfe_vars.remove('t') # Use numeric t_code
rfe_X = ipo2609Cleaned[rfe_vars]
rfe_y = ipo2609Cleaned['underpriced']

# Temporarily override chained assignment warning
class ChainedAssignent:
    def __init__(self, chained=None):
        acceptable = [None, 'warn', 'raise']
        assert chained in acceptable, "chained must be in " + str(acceptable)
        self.swcw = chained

    def __enter__(self):
        self.saved_swcw = pd.options.mode.chained_assignment
        pd.options.mode.chained_assignment = self.swcw
        return self

    def __exit__(self, *args):
        pd.options.mode.chained_assignment = self.saved_swcw


# Scale floating point features for being logistic regression performance
with ChainedAssignent():
    rfe_X[['d','audit','op','max','min','sel','uses', 'of', 'expenses', 'risks', 'sp', 'reg']] = scaler.fit_transform(rfe_X[['d','audit','op','max','min','sel','uses', 'of', 'expenses', 'risks', 'sp', 'reg']])


cat_vars = ['odate', 'zip3', 'sic', 'sic_group', 'uw1', 'uw1_group', 'uw2', 'uw2_group', 'yr', 'sectype', 't_code', 'ex_code']
for var in cat_vars:
    var_dummies = pd.get_dummies(rfe_X[var], prefix=var)
    rfe_X = rfe_X.join(var_dummies)

rfe_X_vars = rfe_X.columns.values.tolist()
to_keep=[i for i in rfe_X_vars if i not in cat_vars]
rfe_X = rfe_X[to_keep]

# Perform RFE on categorical features
logreg = LogisticRegression(solver='liblinear')
rfe = RFE(logreg, 20)
# NOTE: This takes several minutes with more than 1000 features
rfe = rfe.fit(rfe_X, rfe_y.values.ravel())

rfe_ranked_features = sorted(zip(map(lambda x: round(x, 4), rfe.ranking_), rfe_X))
# Take top 50 features
rfe_features = [t[1] for t in rfe_ranked_features[:50]]
# Take features with rank of 10 or lower
#rfe_features = [t[1] for t in rfe_ranked_features if t[0] <= 10]
print(rfe_features)


# Correlation Matrix
cor_X = uni_X.copy()
# Drop columns with insignificant correlations so plot is easier to read
cor_X = cor_X.drop(['uw1', 'uw2', 'sectype', 'of', 'ex_code', 't_code', 'audit', 'risks', 'yr', 'sic', 'lockup', 'expenses'], axis=1)
corrmat = cor_X.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
# Plot heat map
g=sns.heatmap(cor_X[top_corr_features].corr(),annot=True,cmap="RdYlGn")
plt.show()

# Specify final vars to keep for training a model
# Start with non-indicator values with most importance, avoiding highly correlated features
# Note: model_vars is a set to avoid dups
model_vars = {'book', 'd', 'audit', 'in', 'lockup', 'op', 'max', 'sel', 'uses', 'of', 'sa', 'sp'}
# Add top 50 RFE features
model_vars.update(rfe_features)
model_df = rfe_X[list(model_vars)]
model_df = model_df.join(uni_y)

rx_data_step(input_data=model_df,
             output_file="IPO2609FeatureEngineering.xdf",
             overwrite=True,
             xdf_compression_level=5)
