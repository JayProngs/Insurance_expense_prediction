import pandas as pd
from sklearn import model_selection as skl
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

data = pd.read_csv('insurance.csv')
dframe = data.copy()
dframe['region'].fillna(method='bfill', inplace=True)
bmi_median_val = round(dframe['bmi'].median(), 2)
dframe['bmi'].fillna(bmi_median_val, inplace=True)
dframe['children'].fillna(0, inplace=True)
dframe = dframe[dframe['expenses'] < 51240]
dframe = dframe[dframe['expenses'] != 0]  # adding to remove record where expense has to be predicted
dframe_dummy = pd.get_dummies(dframe)
y = dframe_dummy.pop('expenses')
x = dframe_dummy
x_train, x_test, y_train, y_test = skl.train_test_split(x, y, test_size=0.0000001,
                                                        random_state=30)  # traininf with complete data without any test
# start of polynomial regression
poly_reg = PolynomialFeatures(degree=3)
x_poly = poly_reg.fit_transform(x_train)
reg_2 = LinearRegression()
reg_2.fit(x_poly, y_train)
# end of polynomial regression

# getting data for which expenses needs to be predicted
sample = pd.read_csv('insurance.csv')
dsample_dummy = pd.get_dummies(sample)
dsample_dummy = dsample_dummy[dsample_dummy['expenses'] == 0]
dsample_dummy = dsample_dummy.drop(['expenses'], axis=1)
print(sample.info())
y_pred = reg_2.predict(poly_reg.fit_transform(dsample_dummy))
print(y_pred)