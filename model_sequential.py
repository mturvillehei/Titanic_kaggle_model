
import tensorflow as tf
import pandas as pd
import numpy as np
from keras import layers
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from keras.layers import Multiply
from tensorflow.python.ops.numpy_ops import np_config
from keras.regularizers import l2
np_config.enable_numpy_behavior()
from sklearn.model_selection import train_test_split

train_data = pd.read_csv('train_encoded.csv')
test_data = pd.read_csv('test_encoded.csv')

def generate_feature_crosses(X, cross_columns, column_names):
    crossed_features = []
    crossed_feature_names = []

    for i, column1 in enumerate(cross_columns):
        for j, column2 in enumerate(cross_columns):
            if j > i:
 
                crossed_feature = Multiply()([X[column1].values.reshape(-1, 1), X[column2].values.reshape(-1, 1)])
                crossed_features.append(crossed_feature)

                crossed_feature_name = f"{column_names[i]}_{column_names[j]}"
                crossed_feature_names.append(crossed_feature_name)

    feature_crosses = np.concatenate(crossed_features, axis=1)
    feature_crosses_df = pd.DataFrame(feature_crosses, columns=crossed_feature_names)
    feature_crosses_df.index = X.index
    return feature_crosses_df


feature_columns = []
male_data = train_data[train_data['Sex_male'] == 1]
female_data = train_data[train_data['Sex_female'] == 1]
male_test_data = test_data[test_data['Sex_male'] == 1]
female_test_data = test_data[test_data['Sex_female'] == 1]
y_male = male_data['Survived']
y_female = female_data['Survived']
male_data = male_data.drop(columns=['Sex_female','Sex_male'], axis=1)
female_data = female_data.drop(columns=['Sex_female','Sex_male'], axis=1)
male_test_data = male_test_data.drop(columns=['Sex_female','Sex_male'], axis=1)
female_test_data  = female_test_data .drop(columns=['Sex_female','Sex_male'], axis=1)


feature_numeric = [
    'Age',
    'SibSp',
    'Parch',
    'Rooms_Count',
    'Fare',
    
]
feature_OH = [

    'Pclass_1',
    'Pclass_2',
    'Pclass_3',
    'Section_A',
    'Section_B',
    'Section_C',
    'Section_D',
    'Section_E',
    'Section_F',
    'Section_G',
]
cross_columns = ['Age', 'Fare', 
                 'Rooms_Count', 
                 'SibSp', 'Parch', 
                 'Pclass_1', 'Pclass_2', 'Pclass_3',
                 ]

X_male = male_data[feature_OH + feature_numeric]
X_female = female_data[feature_OH + feature_numeric]
testX_male = male_test_data[feature_OH + feature_numeric]
testX_female = female_test_data[feature_OH + feature_numeric]
male_feature_crosses_df = generate_feature_crosses(X_male, cross_columns, cross_columns)
female_feature_crosses_df = generate_feature_crosses(X_female, cross_columns, cross_columns)
testmale_feature_crosses_df = generate_feature_crosses(testX_male, cross_columns, cross_columns)
testfemale_feature_crosses_df = generate_feature_crosses(testX_female, cross_columns, cross_columns)
X_male = pd.concat([X_male, male_feature_crosses_df], axis=1)
X_female = pd.concat([X_female, female_feature_crosses_df], axis=1)
testX_male = pd.concat([testX_male, testmale_feature_crosses_df], axis=1)
testX_female = pd.concat([testX_female, testfemale_feature_crosses_df], axis=1)

scaler = MinMaxScaler()
X_male[feature_numeric] = scaler.fit_transform(X_male[feature_numeric])
X_female[feature_numeric] = scaler.fit_transform(X_female[feature_numeric])
testX_male[feature_numeric] = scaler.fit_transform(testX_male[feature_numeric])
testX_female[feature_numeric] = scaler.fit_transform(testX_female[feature_numeric])


                #~~~~TRAINING OF BOTH MODELS~~~~#


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~MALE~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


split = 0.175
epochs_ = 125
batches = 40
reg2 = 0.015

male_model = Sequential([
    Dense(115, activation='tanh', input_shape=(X_male.shape[1],), kernel_regularizer=l2(reg2)),
    Dropout(0.03),
    Dense(1, activation='sigmoid')
])

male_model.compile(optimizer='RMSprop', loss='binary_crossentropy', metrics=['accuracy'])
history_male = male_model.fit(X_male, y_male, epochs=epochs_, batch_size=batches, validation_split=split)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~FEMALE~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

split = 0.2
epochs_ = 125
batches = 25
nodes = 115
dropout = 0.19
reg2 = 0.03
n_rows = 3
n_cols = 4
fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 15))
fig.tight_layout(pad=5.0)

female_model = Sequential([
    Dense(nodes, activation='relu', input_shape=(X_female.shape[1],), kernel_regularizer=l2(reg2)),
    Dropout(dropout),
    Dense(1, activation='sigmoid')
])
female_model.compile(optimizer='RMSprop', loss='binary_crossentropy', metrics=['accuracy'])
history_female = female_model.fit(X_female, y_female, epochs=epochs_, batch_size=batches, validation_split=split)


###~~~~PREDICTING TEST DATA~~~~###

y_pred_male = male_model.predict(testX_male)
y_pred_female = female_model.predict(testX_female)

y_pred_male_df = pd.DataFrame(y_pred_male, columns=['Survived'], index=testX_male.index)
y_pred_female_df = pd.DataFrame(y_pred_female, columns=['Survived'], index=testX_female.index)
y_pred_combined = pd.concat([y_pred_male_df, y_pred_female_df]).sort_index()

y_pred_combined['Survived'] = np.where(y_pred_combined['Survived'] >= 0.45, 1, 0)


test_raw = pd.read_csv('test.csv')

test_raw.set_index(y_pred_combined.index, inplace=True)
test_raw['Survived'] = y_pred_combined['Survived']

Output = test_raw[['PassengerId', 'Survived']]

Output.to_csv('Submission.csv', index=False)