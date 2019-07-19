# Ritter IPO Analysis
#   Step 1: Import and Clean Ritter IPO datasets
#
#   This script uses the version of Python bundled with the
#     Microsoft Machine Learning Server version 9.3
#     and uses the revoscalepy package to write XDF output files
#     for use in ritter_ipo_feature_mls.py
#
#   Please see the Jupyter notebook for a version of this experiment
#     which does not use the Machine Learning Server
#
# pip install wget

from revoscalepy import rx_data_step
import numpy as np
import pandas as pd
import math
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import wget

# Download Ritter IPO 2609 file
ipo2609_filename = "IPO2609.xls"
url = "https://site.warrington.ufl.edu/ritter/files/2016/01/IPO2609.xls"
wget.download(url, ipo2609_filename)

# Download Ritter IPO 1526 file
ipo1526_filename = "IPO1526.xls"
url = "https://site.warrington.ufl.edu/ritter/files/2016/01/IPO1526.xls"
wget.download(url, ipo1526_filename)

# Read Excel file into dataframe
# NOTE:  YOu can igore WARNING *** OLE2 inconsistency: SSCS size is 0 but SSAT size is non-zero
df = pd.read_excel(ipo2609_filename, sheetname='ipo2609')

# Clean Zip Codes and generate zip3
#   Replace 0, 999 with 00000
df['zip'] = df['zip'].astype(str).str.ljust(5, '0')
df['zip'] = df['zip'].replace(
    to_replace=['99900'],
    value='00000')
#      Add trailing zeros so 5 digits
#df['zip'] = pd.Categorical(df['zip'])
#      Use zip3 for region
df['zip3'] = df['zip'].str[:3].astype("int")

#   Security Type: Last two digits of CUSIP represent security type
df['sectype'] = df['cusip'].astype(str).str[-2:].astype("int")

# Convert Categorical
#   Type
#      Create a type code
df['t_code'] = df['t'].copy().astype("int")
#      There are two NA values 0,9.  Use 9 only
df['t_code'] = df['t_code'].replace(
    to_replace=[0],
    value=9)
df['t'] = df['t'].replace(
    to_replace=[0, 9],
    value='NA')
df['t'] = df['t'].replace(
    to_replace=[1],
    value='Best Efforts')
df['t'] = df['t'].replace(
    to_replace=[2],
    value='Firm Committment')
df['t'] = df['t'].replace(
    to_replace=[3],
    value='Combination')
df['t'] = pd.Categorical(df['t'])
#   Exchange
#      Create an exchange code
df['ex_code'] = df['ex'].copy().astype("int")
df['ex'] = df['ex'].replace(
    to_replace=[1],
    value='NASDAQ')
df['ex'] = df['ex'].replace(
    to_replace=[2],
    value='AMEX')
df['ex'] = df['ex'].replace(
    to_replace=[3],
    value='NYSE')
df['ex'] = df['ex'].replace(
    to_replace=[4],
    value='Non_NASDAQ_OTC')
df['ex'] = pd.Categorical(df['ex'])

# Clean Up Integer Fields
df['odate'] = df['odate'].astype("int")
#   SIC Code
#     999 represents missing
df['sic'] = df['sic'].astype("int")
#     Add 2-digit group code
df['sic_group'] = df['sic'].apply(lambda x: math.floor(x / 10)).astype("int")

#   Lead Underwriter
#      0 represents missing
#      Add group -- see Ritter documentation
df['uw1'] = df['uw1'].astype("int")
df['uw1_group'] = df['uw1'].apply(lambda x: math.floor(x / 100) * 100).astype("int")
#   Other underwriters
df['uw2'] = df['uw2'].astype("int")
df['uw2_group'] = df['uw2'].apply(lambda x: math.floor(x / 10) * 10).astype("int")
df['uw3'] = df['uw3'].astype("int")
df['uw3_group'] = df['uw3'].apply(lambda x: math.floor(x / 10)).astype("int")
#   NOTE: uw4 and uwS (should be 5) dropped below.  Just use first three underwriters
#   Year Organized
#     99 represents missing
df['yr'] = df['yr'].astype("int")
#   Perm (Key to join 1526 file)
df['perm'] = df['perm'].astype("int")

# Convert and scale decimals
#   Offering Price 7.3
df['op'] = df['op'].astype(float) / 1000
#   Closing bid on first aftermarket day 9.5
df['pr1'] = df['pr1'].astype(float) / 100000
#   Value of S&P on Closing Date 5.2
df['sp'] = df['sp'].replace(
    to_replace=[99999],
    value=np.NaN)
df['sp'] = df['sp'].astype(float) / 100
#   Aftermarket standard deviation
df['uncer'] = df['uncer'].replace(
    to_replace=[999],
    value=np.NaN)
df['uncer'] = df['uncer'].astype(float) / 1000
#   Continuousely compounded growth rate
df['gs'] = df['gs'].replace(
    to_replace=[999],
    value=np.NaN)
df['gs'] = df['gs'].astype(float) / 1000
#   Offering as a fraction of shares outstanding
df['of'] = df['of'].replace(
    to_replace=[999],
    value=np.NaN)
df['of'] = df['of'].astype(float) / 100

# Review columns with missing values
# uncer 757
# gs 1506
# of 37
df.isnull().sum()

# Drop columns uncer and gs as well as cusip, zip and uw3,4,5
df = df.drop(['uncer', 'gs', 'cusip', 'zip', 'uw3', 'uw4', 'uwS'], axis=1)

# Drop rows with missing data
df = df[df['op'] != 0]
df = df[df['sic'] != 999]
df = df[df['t'] != 'NA']
df = df[~df['of'].isnull()]

# Calculate D1 percent change in price
df['d1pctchg'] = (df['pr1'] - df['op']) / df['op'] * 100
# Calculate underpriced flag
df['underpriced'] = df['d1pctchg'] > 0

# Plot a Histogram of D1 Price Change Percent
# Remove NaN and Clip Outliers
clipped = np.clip(df['d1pctchg'][~np.isnan(df['d1pctchg'])], -100, 100)
num_bins = 50
n, bins, patches = plt.hist(clipped, num_bins, facecolor='blue', alpha=0.5)
plt.xlabel('D1 Price Change Percent')
plt.ylabel('Count')
plt.title(r'Histogram of D1 Price Change Percent (Outliers Clipped)')
plt.show()


# Simple Outlier implementation based on z-score
def is_outlier(points, thresh=3.0):

    if len(points.shape) == 1:
        points = points[:, None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh



# Plot a Histogram of Offering Price excluding outliers
filtered = df['op'][~is_outlier(df['op'])]
num_bins = 50
n, bins, patches = plt.hist(filtered, num_bins, facecolor='blue', alpha=0.5)
plt.xlabel('Offering Price')
plt.ylabel('Count')
plt.title(r'Histogram of Offering Price (Outliers Excluded)')
plt.show()

# Join IPO1526 (36 month returns for CRSP-listed)
# Parse columns returns perm and 38 months of returns
df_1526 = pd.read_excel(
    ipo1526_filename, sheetname='ipo1526', parse_cols='B,T:BE')

# Replace missing values -100 with NaN for all 38 month return columns
for x in range(1, 39):
    r_col_name = 'r' + str(x)
    df_1526[r_col_name] = df_1526[r_col_name].mask(
        np.isclose(df_1526[r_col_name].values, -99.99998))


# Calculate 12, 24 and 36 month returns
df_1526['r12_sum'] = df_1526.iloc[:, 2:14].sum(axis=1) * 100
df_1526['r24_sum'] = df_1526.iloc[:, 2:26].sum(axis=1) * 100
df_1526['r36_sum'] = df_1526.iloc[:, 2:38].sum(axis=1) * 100

df_1526['r36_sum'].mean()

# Drop r1-r38 columns
for x in range(1, 39):
    r_col_name = 'r' + str(x)
    df_1526.drop(r_col_name, inplace=True, axis=1)

# Merge on Perm
df_1526['perm'] = df_1526['perm'].astype("int")
df_merged = pd.merge(df, df_1526, on='perm')

# Plot Bar Charts of Average Returns by Exchange for Overpriced and underpriced
#    Overpriced Chart
df_bar = df_merged[['underpriced', 'ex',
                    'd1pctchg', 'r12_sum', 'r24_sum', 'r36_sum']]
df_bar = df_bar.groupby(['underpriced', 'ex']).mean()
df_over_amex = df_bar.loc[(df_bar.index.get_level_values(
    'underpriced') == False) & (df_bar.index.get_level_values('ex') == 'AMEX')]
df_over_amex = df_over_amex.values.tolist()[0]
df_over_nyse = df_bar.loc[(df_bar.index.get_level_values(
    'underpriced') == False) & (df_bar.index.get_level_values('ex') == 'NYSE')]
df_over_nyse = df_over_nyse.values.tolist()[0]
df_over_nasdaq = df_bar.loc[(df_bar.index.get_level_values(
    'underpriced') == False) & (df_bar.index.get_level_values('ex') == 'NASDAQ')]
df_over_nasdaq = df_over_nasdaq.values.tolist()[0]

n_groups = 4
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.35
opacity = 0.8

rects1 = plt.bar(index, df_over_amex, bar_width,
                 alpha=opacity,
                 color='b',
                 label='AMEX')
rects2 = plt.bar(index + bar_width, df_over_nyse, bar_width,
                 alpha=opacity,
                 color='g',
                 label='NYSE')
rects3 = plt.bar(index + bar_width * 2, df_over_nasdaq, bar_width,
                 alpha=opacity,
                 color='r',
                 label='NASDAQ')
plt.xlabel('Exchange')
plt.ylabel('Return')
plt.title('Overpriced Average Returns by Exchange')
plt.xticks(index + bar_width, ('D1', 'R12', 'R24', 'R36'))
plt.legend()
plt.tight_layout()
plt.show()

#    UnderPriced Chart
df_under_amex = df_bar.loc[(df_bar.index.get_level_values(
    'underpriced') == True) & (df_bar.index.get_level_values('ex') == 'AMEX')]
df_under_amex = df_under_amex.values.tolist()[0]
df_under_nyse = df_bar.loc[(df_bar.index.get_level_values(
    'underpriced') == True) & (df_bar.index.get_level_values('ex') == 'NYSE')]
df_under_nyse = df_under_nyse.values.tolist()[0]
df_under_nasdaq = df_bar.loc[(df_bar.index.get_level_values(
    'underpriced') == True) & (df_bar.index.get_level_values('ex') == 'NASDAQ')]
df_under_nasdaq = df_under_nasdaq.values.tolist()[0]

n_groups = 4
fig, ax = plt.subplots()
index = np.arange(n_groups)
rects1 = plt.bar(index, df_under_amex, bar_width,
                 alpha=opacity,
                 color='b',
                 label='AMEX')
rects2 = plt.bar(index + bar_width, df_under_nyse, bar_width,
                 alpha=opacity,
                 color='g',
                 label='NYSE')
rects3 = plt.bar(index + bar_width * 2, df_under_nasdaq, bar_width,
                 alpha=opacity,
                 color='r',
                 label='NASDAQ')
plt.xlabel('Exchange')
plt.ylabel('Return')
plt.title('Underpriced Average Returns by Exchange')
plt.xticks(index + bar_width, ('D1', 'R12', 'R24', 'R36'))
plt.legend()
plt.tight_layout()
plt.show()

# Write data to Output Files using rx_data_step

# ipo2609
rx_data_step(input_data=df,
             output_file="IPO2609Cleaned.xdf",
             overwrite=True,
             xdf_compression_level=5)

# Merged ipo1526
rx_data_step(input_data=df_merged,
             output_file="IPO1526Merged.xdf",
             overwrite=True,
             xdf_compression_level=5)
