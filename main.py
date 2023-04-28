import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score
# from matplotlib import style
# style.use('seaborn')
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor


# reading data
dataset = pd.read_csv('ford.csv')
# print(dataset.head(5))
# print('Information on Dataset:', dataset.info())


# checking for null values
dataset.isnull().sum()



plt.figure(figsize=(12,10))
sns.countplot(y='model', data=dataset)
plt.title('Model Types')
# plt.show()


sns.countplot(y='transmission', data=dataset)
plt.title('Transmission Types')
plt.show()


sns.countplot(x='fuelType', data=dataset)
plt.title('Fuel Types')
plt.show()




print(dataset['model'].value_counts())
print("\n\n")
print(dataset['transmission'].value_counts())
print("\n\n")
print(dataset['fuelType'].value_counts())



fuelType = dataset['fuelType']
transmission = dataset['transmission']
price = dataset['price']
fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
fig.suptitle('Visualizing categorical data columns')
sns.barplot(x=fuelType, y=price, ax=axes[0])
sns.barplot(x=transmission, y=price, ax=axes[1])




# converting categorical values to numerical values
dataset.replace({'transmission': {'Manual':0, 'Automatic':1, 'Semi-Auto':2}}, inplace=True)
dataset.replace({'fuelType': {'Petrol':0, 'Diesel':1, 'Hybrid':2, 'Electric':3, 'Other':4}}, inplace=True)

dataset = dataset.drop('model', axis=1)
# print(dataset.head())


# checking the correlation between the attributes and 
plt.figure(figsize=(10,7))
sns.heatmap(dataset.corr(), annot=True)
plt.title('Correlation between the columns')
plt.show()

dataset.corr()['price'].sort_values()

fig = plt.figure(figsize=(7, 5))
plt.title('Correlation between year and price')
sns.regplot(x='price', y='year', data=dataset)
plt.show()



X = dataset.drop('price', axis=1)
y = dataset['price']
# print("Shape of X is:", X.shape, "\t\t\tShape of y is:", y.shape)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# using standardscaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)

lr_mae = mean_absolute_error(y_test, lr_pred)
lr_r2 = r2_score(y_test, lr_pred)
print("MAE of Linear Regression Model is:", lr_mae)
print("R^2 score of Linear Regression Model is", lr_r2)

lr_score = cross_val_score(lr, X_test, y_test, cv=4)
print("================================================================================")
print("Linear Regression Model Accuracy Values are:", lr_score) # this gives a list of different accuracy values
print("Linear Regression Model Accuracy is: {}".format(lr_score.mean()*100))

print("================================================================================")

dtree = DecisionTreeRegressor()
dtree.fit(X_train, y_train)
dtree_pred = dtree.predict(X_test)

dtree_mae = mean_absolute_error(y_test, dtree_pred)
dtree_r2 = r2_score(y_test, dtree_pred)
print("MAE of Decision Tree Model is:", dtree_mae)
print("R^2 score of Decision Tree Model is", dtree_r2)

dtree_score = cross_val_score(dtree, X_test, y_test, cv=4)
print("Decision Tree Model Accuracy Values are:", dtree_score) # this gives a list of different accuracy values
print("Decision Tree Model Accuracy is: {}".format(dtree_score.mean()*100))

print("================================================================================")

xgb = XGBRegressor()
xgb.fit(X_train, y_train)
xgb_pred = xgb.predict(X_test)

xgb_mae = mean_absolute_error(y_test, xgb_pred)
xgb_r2 = r2_score(y_test, xgb_pred)
print("MAE of XGBoost Model is:", xgb_mae)
print("R^2 score of XGBoost Model is", xgb_r2)
xgb_score = cross_val_score(xgb, X_test, y_test, cv=4)
print("XGBoost Model Accuracy Values are:", xgb_score) # this gives a list of different accuracy values
print("XGBoost Model Accuracy is: {}".format(xgb_score.mean()*100))

print("================================================================================")



# using the best model performance in this model, the xgboost, we predict for a sample model of our own(i.e predict the car price for a new data)
# print(dataset.columns)
print(dataset.head(10))

print("================================================================================")

car_year = int(input('Input Car Year:\t'))
car_trans = int(input('The Car Transmission has three types, choose 0 for Manual, 1 for Automatic, 2 for Semi-Auto\nInput Car Transmission:\t'))
car_mileage = int(input('Input Car Mileage:\t'))
car_fueltype = int(input('Input the following for fuel type\n0 for Petrol, 1 for Diesel, 2 for Hybrid, 3 for Electric, 4 for Others\nInput Car Fuel Type:\t'))
car_tax = int(input('Input Car Tax:\t'))
car_mpg = float(input('Input Car MPG:\t'))
eng_size = float(input('Input Car Engine Size:\t'))


data = {'year':car_year, 'transmission':car_trans, 'mileage':car_mileage, 'fuelType':car_fueltype, 'tax':car_tax, 'mpg':car_mpg, 'engineSize':eng_size}
index = [0]
new_dataset = pd.DataFrame(data, index)

new_pred = xgb.predict(new_dataset)
print('The car price for the your new data is', new_pred)

