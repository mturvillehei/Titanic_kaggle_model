#~~~~Encoding data before model training the model

import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

#Predicting ages, since Age is very important in the decisionmaking
def age_predict(data):
    
    # Split the data into known and unknown age subsetsf
    known_age_data = data[data['Age'].notna()]
    unknown_age_data = data[data['Age'].isna()]

    # Separate input features and target variable for the known age subset
    X_known = known_age_data[['SibSp', 'Parch', 'Fare']]
    y_known = known_age_data['Age']

    # Scale input features
    scaler = StandardScaler()
    X_known_scaled = scaler.fit_transform(X_known)

    # Split the known age data into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_known_scaled, y_known, test_size=0.2, random_state=42)

    # Create a simple linear regression model using TensorFlow
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(1, input_shape=(3,))
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='mse')

    # Train the model
    history = model.fit(X_train, y_train, epochs=250, validation_data=(X_val, y_val), verbose=0)

    # Predict missing age values
    X_unknown = unknown_age_data[['SibSp', 'Parch', 'Fare']]
    X_unknown_scaled = scaler.transform(X_unknown)
    age_predictions = model.predict(X_unknown_scaled).flatten()

    # Fill in the missing age values with the predictions
    data.loc[data['Age'].isna(), 'Age'] = age_predictions

    '''
    # Plot training and validation loss
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training loss')
    plt.plot(history.history['val_loss'], label='Validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.show()
    '''

    return data

#~~~Thanks GPT-4 for this~~~#
def encode_cabin_data(data, cabin_column):
    # Split the cabin string into sections and room numbers
    data['Sections'] = data[cabin_column].apply(lambda x: ''.join([c for c in str(x) if c.isalpha() or c.isspace()]) if pd.notnull(x) else '')
    data['Rooms_Count'] = data[cabin_column].apply(lambda x: len(str(x).split()) if pd.notnull(x) else 0)
    
    # One-hot encode the sections
    one_hot_sections = pd.get_dummies(data['Sections'].apply(lambda x: list(x.split()) if x else []).apply(pd.Series).stack(), prefix='Section').sum(level=0)
    
    # Merge the one-hot encoded sections, rooms count, and original data (excluding the original cabin column and temporary 'Sections' column)
    encoded_data = pd.concat([data.drop(columns=[cabin_column, 'Sections']), one_hot_sections, data['Rooms_Count']], axis=1)
    
    #Filling empty values in the OH columns
    one_hot_columns = ['Section_A', 'Section_B', 'Section_C', 'Section_D', 'Section_E', 'Section_F', 'Section_G']
    for column in one_hot_columns:
        encoded_data[column] = encoded_data[column].fillna(0)

    return encoded_data

def last_name(data):
    data['last_name'] = data['Name'].apply(lambda x: x.split(',')[0])
    encoder = LabelEncoder()
    data['last_name_encoded'] = encoder.fit_transform(data['last_name'])
    data['family_size'] = data['last_name'].map(data['last_name'].value_counts())

    return data

def convert_Z(column):
    col_mean = column.mean()
    col_std = column.std()
    column_ = (column - col_mean)/col_std
    return column_

def OH_encode_column(data, column_name):
    # Create a one-hot encoded DataFrame from the specified column
    one_hot_encoded_df = pd.get_dummies(data[column_name], prefix=column_name)
    
    # Drop the original column from the input DataFrame
    data_without_original_column = data.drop(column_name, axis=1)
    
    # Concatenate the DataFrame without the original column and the one-hot encoded DataFrame
    encoded_data = pd.concat([data_without_original_column, one_hot_encoded_df], axis=1)
    
    return encoded_data

#~~~Encoding my Dataframe~~~#
# --PassengerId, Survived, Pclass, --Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked

#Importing the data
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")
#shuffling my training data
train_data = train_data.reindex(np.random.permutation(train_data.index))

#Last name
train_data = last_name(train_data)
test_data = last_name(test_data)

#One-hot encoding
OH_encodes = ['Sex', 'Pclass', 'Embarked']
for key in OH_encodes:
    train_data = OH_encode_column(train_data, key)
    test_data = OH_encode_column(test_data, key)
    
#Normalized Z-number encoding
Z_encodes = ['Age', 'Fare', 'SibSp', 'Parch', 'family_size']
for key in Z_encodes:

    train_data[key] = convert_Z(train_data[key])
    test_data[key] = convert_Z(test_data[key])

#Cabin Room
train_data = encode_cabin_data(train_data, 'Cabin')
test_data = encode_cabin_data(test_data, 'Cabin')

#Age
train_data = age_predict(train_data)
test_data = age_predict(test_data)

train_data.to_csv('train_encoded.csv')
test_data.to_csv('test_encoded.csv')