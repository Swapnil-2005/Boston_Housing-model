import pandas as pd
housing=pd.read_csv(r"C:\Users\sb505\OneDrive\Desktop\BOSTON_HOUSING.csv")
housing.head()
housing.info() # to check if any value is null or not
housing.describe()
import matplotlib.pyplot as plt
housing.hist(bins=50 , figsize=(20,15))
from sklearn.model_selection import train_test_split
train_set,test_set=train_test_split(housing,test_size=0.2,random_state=42)
print(f" Rows in train set : {len(train_set)} \n Rows in test set :{len(test_set)}")
from sklearn.model_selection import StratifiedShuffleSplit
split=StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
for train_index,test_index in split.split(housing,housing['CHAS']):
    strat_train_set=housing.loc[train_index]
    strat_test_set=housing.loc[test_index]
print(strat_test_set.info())
housing=strat_test_set.copy()
corr_matrix=housing.corr()
corr_matrix['MEDV'].sort_values(ascending=False) # shows correlation with other cols
from pandas.plotting import scatter_matrix

attributes = ['MEDV', 'RM', 'ZN', 'LSTAT']
scatter_matrix(housing[attributes], figsize=(12, 12))
# Corrected: (width, height)
housing.plot(kind='scatter',x='RM',y='MEDV',alpha=0.8)
median=housing['RM'].median()
housing['RM'].fillna(median)
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="median")  # You can change "mean" to "median" or "most_frequent"
imputer.fit(housing)
imputer.statistics_ 
x=imputer.transform(housing)
housing_tr=pd.DataFrame(x,columns=housing.columns)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
my_pipeline=Pipeline([   ('imputer',SimpleImputer(strategy="median")),  ('std_scaler',StandardScaler())
                                        ])
housing_num_tr=my_pipeline.fit_transform(housing_tr)
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
#model=DecisionTreeRegressor()
#model=LinearRegressor
model=RandomForestRegressor()
model.fit(housing_num_tr,housing_labels)
some_data=housing.iloc[:5]
some_labels=housing_labels.iloc[:5]
prepared_data=my_pipeline.transform(some_data)
model.predict(prepared_data)
list(some_labels)
