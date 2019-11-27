import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

train = pd.read_csv("./tcd-ml-1920-group-income-train.csv")

test = pd.read_csv("./tcd-ml-1920-group-income-test.csv")

rename_cols = {"Total Yearly Income [EUR]":'Income',
               "Yearly Income in addition to Salary (e.g. Rental Income)": 'Yearly Income in addition to Salary',
               "Work Experience in Current Job [years]": 'Work Experience in Current Job',
               "Body Height [cm]": 'Body Height'
               }

train = train.rename(columns=rename_cols)
test = test.rename(columns=rename_cols)

data = pd.concat([train,test],ignore_index=True)

def pre_process_data(train_and_test):

    ##Year of Record
    year = ['Year of Record']
    train_and_test[year] = train_and_test[year].fillna(method='ffill') ##
    train_and_test[year] = train_and_test[year].apply(pd.to_numeric, downcast='integer')


    ##Housing Situation
    ##print(train['Housing Situation'].unique())
    housing_situation = ['Castle', 'Large House', 'Medium House', 'Small House', 'Large Apartment', 'Medium Apartment', 'Small Apartment']
    train_and_test.loc[~train_and_test['Housing Situation'].isin(housing_situation), 'Housing Situation'] = 'unknown'


    #Crime Level in the City of Employement
    # print(train['Crime Level in the City of Employement'].unique())
    # print(test['Crime Level in the City of Employement'].unique())
    ## normalize or scale??

    #Work Experience in Current Job [years]
    # print(train_and_test['Work Experience in Current Job [years]'].unique())
    train_and_test.loc[train_and_test['Work Experience in Current Job'].isin(['#NUM!']), 'Work Experience in Current Job'] = 0.0
    train_and_test['Work Experience in Current Job'] = train_and_test['Work Experience in Current Job'].astype(float)
    # print(train_and_test['Work Experience in Current Job [years]'].unique())


    #Satisfation with employer
    # print(train_and_test['Satisfation with employer'].unique())
    Satisfation_with_employer = ['Unhappy', 'Average', 'Happy', 'Somewhat Happy']
    train_and_test.loc[~train_and_test['Satisfation with employer'].isin(Satisfation_with_employer), 'Satisfation with employer'] = 'Average'
    # print(train_and_test['Satisfation with employer'].unique())

    #Gender
    # print(train_and_test['Gender'].unique())
    train_and_test['Gender'] = train_and_test['Gender'].map({'f': 'female'})
    accepted_genders = ['other', 'female', 'male']
    train_and_test.loc[~train_and_test['Gender'].isin(accepted_genders), 'Gender'] = 'unknown'


    #Age
    train_and_test['Age'] = train_and_test['Age'].fillna(method='ffill') ##

    #Country
    # print(train_and_test['Country'].unique())
    train_and_test['Country'] = train_and_test['Country'].fillna(method='ffill') ##
    # print(train_and_test['Country'].unique())

    #Size of City
    #print(train_and_test['Size of City'].unique())
    train_and_test['Size of City'] = train_and_test['Size of City'].fillna(method='ffill') ##
    #print(train_and_test['Size of City'].unique())



    #profession
    # for profession in train_and_test['Profession'].unique():
    #     print (profession)
    train_and_test['Profession'] = train_and_test['Profession'].fillna(method='ffill')
    accepted_professions = train["Profession"].values
    train_and_test.loc[~train_and_test['Profession'].isin(accepted_professions), 'Profession'] = 'unknown'

    mean = train_and_test.groupby('Profession')['Income'].mean()
    train_and_test['Profession'] = train_and_test['Profession'].map(mean)


    #University Degree
    #print(train_and_test['University Degree'].unique())
    accepted_degrees =['No', 'Bachelor', 'Master', 'PhD']
    train_and_test.loc[~train_and_test['University Degree'].isin(accepted_degrees), 'University Degree'] = 'unknown'

    #Wears Glasses
    #print(train_and_test['Wears Glasses'].unique()) [0 1] nothing to clean


    #Hair Color
    # print(train_and_test['Hair Color'].unique())
    accepted_hair_colors = ['Black', 'Blond', 'Brown', 'Red', 'Unknown']
    train_and_test.loc[~train_and_test['Hair Color'].isin(accepted_hair_colors), 'Hair Color'] = 'Unknown'
    # print(train_and_test['Hair Color'].unique())

    #Body Height [cm]
    #print(train_and_test['Body Height [cm]'].unique())



    #Yearly Income in addition to Salary (e.g. Rental Income)
    train_and_test['Yearly Income in addition to Salary'] = train_and_test['Yearly Income in addition to Salary'].str.replace(' EUR', '')
    train_and_test['Yearly Income in addition to Salary'] = train_and_test['Yearly Income in addition to Salary'].fillna(0) ##
    train_and_test['Yearly Income in addition to Salary'] = train_and_test['Yearly Income in addition to Salary'].apply(pd.to_numeric, downcast='float')


    train_and_test['Income'] = train_and_test['Income'] - train_and_test['Yearly Income in addition to Salary']
    return train_and_test


data = pre_process_data(data)

print("pre processing finished")

del_col = ['Wears Glasses']

data = data.drop(del_col, axis = 1)

for col in data.dtypes[data.dtypes == 'object'].index.tolist():
    feat_le = LabelEncoder()
    feat_le.fit(data[col].unique().astype(str))
    data[col] = feat_le.transform(data[col].astype(str))

print("Label Encoding done!")

del_col = set(['Income','Instance'])
features_col =  list(set(data) - del_col)

print(features_col)

X_train,X_test = data[features_col].iloc[:1048574],data[features_col].iloc[1048574:]

Y_train = data['Income'].iloc[:1048574]
X_test_id = data['Instance'].iloc[1048574:]
x_train,x_val,y_train,y_val = train_test_split(X_train,Y_train,test_size=0.2,random_state=1234)


params = {
 'boosting_type': 'gbdt',
 'objective': 'regression',
 'device_type':'gpu',
 'metric': {'mae'},
 'num_leaves': 550,
 'learning_rate': 0.0495,
 'feature_fraction': 0.9,
 'bagging_fraction': 0.8,
 'bagging_freq': 5,
 'verbose': 1
}

#num_iteration, n_iter, num_tree, num_trees, num_round, num_rounds, num_boost_round, n_estimators

trn_data = lgb.Dataset(x_train, label=y_train)
val_data = lgb.Dataset(x_val, label=y_val)
# test_data = lgb.Dataset(X_test)
print("training starts")

clf = lgb.train(params, trn_data, 5000, valid_sets = [trn_data, val_data], verbose_eval=100, early_stopping_rounds=80)
print("training ends")

print("predict")

pre_test_lgb = clf.predict(X_test)
print("prediction ends")

adjusted_predict = pre_test_lgb + X_test['Yearly Income in addition to Salary']

with_income =  pd.DataFrame({'Instance':X_test_id, 'Total Yearly Income [EUR]':adjusted_predict})

with_income.to_csv("LightGBM with new parameters - glasses - group professions.csv",index=False)
