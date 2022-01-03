import matplotlib.pyplot as pyplt
import pandas as pd
from sklearn import model_selection as skl
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

data = pd.read_csv('insurance.csv')
dframe = data.copy()

# dropping rows where region is not mentioned
# dframe['region'].dropna(inplace=True)
dframe['region'].fillna(method='bfill', inplace=True)

# filling bmi using median
bmi_median_val = round(dframe['bmi'].median(), 2)
dframe['bmi'].fillna(bmi_median_val, inplace=True)

# filling number of children with 0
dframe['children'].fillna(0, inplace=True)

dframe = dframe[dframe['expenses'] < 51240]
dframe_dummy = pd.get_dummies(dframe)

# getting dependent and independent data
y = dframe_dummy.pop('expenses')
x = dframe_dummy

# splitting up train abd test data
x_train, x_test, y_train, y_test = skl.train_test_split(x, y, test_size=0.3, random_state=30)

# start of linear regression

reg = LinearRegression()
reg.fit(x_train, y_train)
# plotting scatter graph for accuracy
# Age vs Expenses
pyplt.subplots(figsize=(6, 6))
pyplt.subplot(1, 1, 1)
pyplt.scatter(dframe_dummy['age'], y, color='red')  # actual expense
pyplt.scatter(dframe_dummy['age'], reg.predict(x), color='blue')  # predicted expense
pyplt.title('Linear Regression : Actual vs Predicted Expenses', fontsize=16)
pyplt.xlabel('Age', fontsize=14)
pyplt.ylabel('Expenses', fontsize=14)
# pyplt.show()
# bmi vs expenses
pyplt.subplots(figsize=(6, 6))
pyplt.subplot(1, 1, 1)
pyplt.scatter(dframe_dummy['bmi'], y, color='red')  # actual expense
pyplt.scatter(dframe_dummy['bmi'], reg.predict(x), color='blue')  # predicted expense
pyplt.title('Linear Regression : Actual vs Predicted Expenses', fontsize=16)
pyplt.xlabel('BMI', fontsize=14)
pyplt.ylabel('Expenses', fontsize=14)
# pyplt.show()
# children vs expenses
pyplt.subplots(figsize=(6, 6))
pyplt.subplot(1, 1, 1)
pyplt.scatter(dframe_dummy['children'], y, color='red')  # actual expense
pyplt.scatter(dframe_dummy['children'], reg.predict(x), color='blue')  # predicted expense
pyplt.title('Linear Regression : Actual vs Predicted Expenses', fontsize=16)
pyplt.xlabel('Children', fontsize=14)
pyplt.ylabel('Expenses', fontsize=14)
# pyplt.show()
# accuracy of linear regression model
print('Linear Regression accuracy:' + str(reg.score(x_test, y_test)))  # 0.7541699723672123

# end of linear regression


# start of polynomial regression
poly_reg = PolynomialFeatures(degree=3)
x_poly = poly_reg.fit_transform(x_train)
reg_2 = LinearRegression()
reg_2.fit(x_poly, y_train)
# plotting scatter graph for accuracy
# Age vs Expenses
pyplt.subplots(figsize=(6, 6))
pyplt.subplot(1, 1, 1)
pyplt.scatter(dframe_dummy['age'], y, color='red')  # actual expense
pyplt.scatter(dframe_dummy['age'], reg_2.predict(poly_reg.fit_transform(x)), color='blue')  # predicted expense
pyplt.title('Polynomial Regression : Actual vs Predicted Expenses', fontsize=16)
pyplt.xlabel('Age', fontsize=14)
pyplt.ylabel('Expenses', fontsize=14)
pyplt.show()
# bmi vs expenses
pyplt.subplots(figsize=(6, 6))
pyplt.subplot(1, 1, 1)
pyplt.scatter(dframe_dummy['bmi'], y, color='red')  # actual expense
pyplt.scatter(dframe_dummy['bmi'], reg_2.predict(poly_reg.fit_transform(x)), color='blue')  # predicted expense
pyplt.title('Polynomial Regression : Actual vs Predicted Expenses', fontsize=16)
pyplt.xlabel('BMI', fontsize=14)
pyplt.ylabel('Expenses', fontsize=14)
pyplt.show()
# children vs expenses
pyplt.subplots(figsize=(6, 6))
pyplt.subplot(1, 1, 1)
pyplt.scatter(dframe_dummy['children'], y, color='red')  # actual expense
pyplt.scatter(dframe_dummy['children'], reg_2.predict(poly_reg.fit_transform(x)), color='blue')  # predicted expense
pyplt.title('Polynomial Regression : Actual vs Predicted Expenses', fontsize=16)
pyplt.xlabel('Children', fontsize=14)
pyplt.ylabel('Expenses', fontsize=14)
pyplt.show()
# accuracy of linear regression model
print('Polynomial Regression accuracy:' + str(reg_2.score(poly_reg.fit_transform(x_test), y_test)))
# end of polynomial regression
