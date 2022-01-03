# visualization libraries
import matplotlib.pyplot as pyplt
import seaborn as sb
import pandas as pd


def paramwise_bar_plot(dframe, param):
    pyplt.figure(figsize=(10, 5))
    dframe.groupby([param])['expenses'].mean().plot.bar()
    pyplt.ylabel('Average Medical Expense')
    pyplt.title("Average Expenses as per " + param, fontsize=18)
    pyplt.xticks(rotation=0)
    pyplt.show()


def plotting(dframe, param):
    pyplt.subplots(figsize=(15, 7))

    # box plot
    pyplt.subplot(1, 2, 1)
    dframe[param].plot.box()

    # histogram
    pyplt.subplot(1, 2, 2)
    pyplt.hist(dframe[param], bins=20)

    # show box plot and histogram
    pyplt.show()


def missing_vals(dframe):
    # find cols with missing values
    total = 0
    for col in dframe.columns:
        missing_vals = dframe[col].isnull().sum()
        total += missing_vals
        if missing_vals != 0:
            print(f"{col} => {dframe[col].isnull().sum()}")
    if total == 0:
        print('No columns left')


# loading dataset
data = pd.read_csv('insurance.csv')

# printing first 5 records
# print(data.head(5))
# print(data.info())

# count of null values
dframe = data.copy()

# print(dframe.isnull().sum())

# missing_vals(dframe)

# dropping rows where region is not mentioned
# dframe['region'].dropna(inplace=True)
dframe['region'].fillna(method='bfill', inplace=True)

# filling bmi using median
bmi_median_val = round(dframe['bmi'].median(), 2)
dframe['bmi'].fillna(bmi_median_val, inplace=True)

# filling number of children with 0
dframe['children'].fillna(0, inplace=True)

# print(dframe.isnull().sum())

# find the summary of expenses
# print(dframe.expenses.describe())
# OP
# count     1342.000000
# mean     13259.992787
# std      12094.794413
# min       1121.870000
# 25%       4746.517500
# 50%       9382.030000
# 75%      16584.320000
# max      63770.430000
# as max is very high than min refer to box plot and histogram to find outlier and limit

# plotting box plot and histogram of data frame
# plotting(dframe, 'expenses')

# before data cleaning (1342, 7)
dframe = dframe[dframe['expenses'] < 51240]
# print(dframe.shape)
# after data cleaning (1336, 7)

# plotting box plot and histogram of data frame
# plotting(dframe, 'expenses')

# find non numeric data with .info() and turn it in numeric data
dframe_dummy = pd.get_dummies(dframe)
# print(dframe_dummy)

# plot distribution graph
# sb.displot(dframe_dummy.expenses, kde=True)
# pyplt.title('Expenses distribution', fontsize=12)
# pyplt.show()

# for i in ['sex', 'smoker', 'children', 'region']:
#     paramwise_bar_plot(dframe, i)

# pair plotting with different features
pyplt.figure(figsize=(15, 15))
sb.pairplot(dframe)
pyplt.show()

pyplt.figure(figsize=(15, 15))
sb.heatmap(dframe_dummy.corr(), annot=True)
pyplt.show()

# maximum co relation with smoker feature only
