# Libraries for math and graphing
import numpy as np
import tensorflow as tensorflow
import itertools
import matplotlib.pyplot as plt
import csv
import random
import graphviz

from graphviz import *
from random import randint
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.metrics import categorical_crossentropy
from sklearn.metrics import confusion_matrix
from IPython.display import SVG
from keras.utils import *
from keras.applications import *
# from google.colab import files
from keras.utils import plot_model


def plot_confusion_matrix(
    cm,
    classes,
    normalize = False,
    title = 'Confusion matrix',
    cmap = plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting 'normalize=True'.
    """

    plt.imshow(cm, interpolation='nearest', cmap = cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation = 45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float')/cm.sum(axis = 1)[:, np.newaxis]
    else:
        print("Confusion matrix, without normalization")

    print(cm)

    thresh = cm.max()/2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
            horizontalalignment = "center",
            color = "white" if cm[i, j] > thresh else "black")


# Creates list to store data
data = []

# Opens CSV file and reads data
with open('heart.csv', 'r') as csv_file:
    csv_reader = csv.reader(csv_file)
    next(csv_reader)

    # Appends data to list
    for line in csv_reader:
        {
            data.append((line[0], line[1], line[2], line[3], line[4], line[5], line[6], line[7], line[8], line[9], line[10], line[11], line[12], line[13]))
        }

# Parameters from CSV
age = []
sex = []
chest_pain_type = []
resting_blood_pressure = []
serium_cholestoral = []
fasting_blood_sugar = []
resting_ECG = []
maximum_heart_rate = []
exercise_induced_angina = []
oldpeak = []
peak_exercise_slope = []
major_vessels = []
thal = []

# Labels
labels = []

# Shuffles the data so we can try new learning combinations each time we execute
np.random.shuffle(data)

# Takes data from list created earlier in program and stores them in the sublists
for i in data:
    age.append(i[0])
    sex.append(i[1]),
    chest_pain_type.append(i[2]),
    resting_blood_pressure.append(i[3]),
    serium_cholestoral.append(i[4]),
    fasting_blood_sugar.append(i[5]),
    resting_ECG.append(i[6]),
    maximum_heart_rate.append(i[7]),
    exercise_induced_angina.append(i[8]),
    oldpeak.append(i[9]),
    peak_exercise_slope.append(i[10]),
    major_vessels.append(i[11]),
    thal.append(i[12]),
    labels.append(i[13])

# Converts lists into Numpy arrays which are much faster and use less memory
age = np.array(age)
sex = np.array(sex)
chest_pain_type = np.array(chest_pain_type)
resting_blood_pressure = np.array(resting_blood_pressure)
serium_cholestoral = np.array(serium_cholestoral)
fasting_blood_sugar = np.array(fasting_blood_sugar)
resting_ECG = np.array(resting_ECG)
maximum_heart_rate = np.array(maximum_heart_rate)
exercise_induced_angina = np.array(exercise_induced_angina)
oldpeak = np.array(oldpeak)
peak_exercise_slope = np.array(peak_exercise_slope)
major_vessels = np.array(major_vessels)
thal = np.array(thal)
labels = np.array(labels)


# Creates a scaler so we can take our data values and convert them into smaller values which is faster for processing
scaler = MinMaxScaler(feature_range=(0, 1))

# Applies the scaler to each of the sublists
age = scaler.fit_transform(age.reshape(-1, 1))
sex = scaler.fit_transform(sex.reshape(-1, 1))
chest_pain_type = scaler.fit_transform(chest_pain_type.reshape(-1, 1))
resting_blood_pressure = scaler.fit_transform(resting_blood_pressure.reshape(-1, 1))
serium_cholestoral = scaler.fit_transform(serium_cholestoral.reshape(-1, 1))
fasting_blood_sugar = scaler.fit_transform(fasting_blood_sugar.reshape(-1, 1))
resting_ECG = scaler.fit_transform(resting_ECG.reshape(-1, 1))
maximum_heart_rate = scaler.fit_transform(maximum_heart_rate.reshape(-1, 1))
exercise_induced_angina = scaler.fit_transform(exercise_induced_angina.reshape(-1, 1))
oldpeak = scaler.fit_transform(oldpeak.reshape(-1, 1))
peak_exercise_slope = scaler.fit_transform(peak_exercise_slope.reshape(-1, 1))
major_vessels = scaler.fit_transform(major_vessels.reshape(-1, 1))
thal = scaler.fit_transform(thal.reshape(-1, 1))

labels = scaler.fit_transform(labels.reshape(-1, 1))


# Creates a training set and a testing set for each of the 13 data parameters
age_train, age_test, sex_train, sex_test, chest_pain_type_train, chest_pain_type_test, resting_blood_pressure_train, resting_blood_pressure_test, serium_cholestoral_train, serium_cholestoral_test, fasting_blood_sugar_train, fasting_blood_sugar_test, resting_ECG_train, resting_ECG_test, maximum_heart_rate_train, maximum_heart_rate_test, exercise_induced_angina_train, exercise_induced_angina_test, oldpeak_train, oldpeak_test, peak_exercise_slope_train, peak_exercise_slope_test, major_vessels_train, major_vessels_test, thal_train, thal_test, labels_train, labels_test = train_test_split(
    age,
    sex,
    chest_pain_type,
    resting_blood_pressure,
    serium_cholestoral,
    fasting_blood_sugar,
    resting_ECG,
    maximum_heart_rate,
    exercise_induced_angina,
    oldpeak,
    peak_exercise_slope,
    major_vessels,
    thal,
    labels,
    # 0.2 or 20% of the data we have will go to the testing set, the remaining 80% will be used for training
    # We shuffle once again so we're using different data each time and aren't geting the same predictions
    test_size = 0.2, random_state = 1, shuffle = True)

# Creates Keras tensors
input1 = Input(shape = (1,))
input2 = Input(shape = (1,))
input3 = Input(shape = (1,))
input4 = Input(shape = (1,))
input5 = Input(shape = (1,))
input6 = Input(shape = (1,))
input7 = Input(shape = (1,))
input8 = Input(shape = (1,))
input9 = Input(shape = (1,))
input10 = Input(shape = (1,))
input11 = Input(shape = (1,))
input12 = Input(shape = (1,))
input13 = Input(shape = (1,))

# Concatenates the 13 data streams to then be compared to the labels
inputs = Concatenate()([input1, input2, input3, input4, input5, input6, input7, input8, input9, input10, input11, input12, input13])

# Creates 3 layers, an input layer, a hidden layer, and an output layer
dense1 = keras.layers.Dense(128, activation = 'relu')
dense2 = keras.layers.Dense(64, activation = 'relu')
dense3 = keras.layers.Dense(2, activation = 'softmax')

# Adjusts weights and biases
x = dense1(inputs)
x = dense2(x)

# Generates output
outputs = dense3(x)

# GENERATES MODEL
# Creates a model from the 13 parameters
model = keras.Model(inputs = [input1, input2, input3, input4, input5, input6, input7, input8, input9, input10, input11, input12, input13], outputs = outputs)

# COMPILES MODEL
model.compile(
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    # Learning rate of 0.001 using the Adam algorithm
    # This is a gradient descent algorithm so it moves in the direction that minimizes the loss funciton
    optimizer = keras.optimizers.Adam(lr = 0.001),
    metrics = ["accuracy"],
)

# TRAINING THE MODEL
# Uses the training data set which was 80% of the original
history = model.fit([age_train, sex_train, chest_pain_type_train, resting_blood_pressure_train, serium_cholestoral_train, fasting_blood_sugar_train, resting_ECG_train, maximum_heart_rate_train, exercise_induced_angina_train, oldpeak_train, peak_exercise_slope_train, major_vessels_train, thal_train],
                    labels_train,
                    batch_size= 41,
                    # 50 epochs
                    epochs = 20,
                    verbose = 2,
                    shuffle=True)

# TESTING THE MODEL
# Evalutes the model with the testing data set which was 20% of the original
# Sees how the model's predictions compare to the actual values
scores = model.evaluate([age_test, sex_test, chest_pain_type_test, resting_blood_pressure_test, serium_cholestoral_test, fasting_blood_sugar_test, resting_ECG_test, maximum_heart_rate_test, exercise_induced_angina_test, oldpeak_test, peak_exercise_slope_test, major_vessels_test, thal_test],
                    labels_test,
                    verbose = 2)


# Prints summary of the data
print("\nTest Loss:", scores[0])
print("Test Accuracy:", scores[1])
print(model.summary())

print(labels_test.size)

# Generates predicitons using the test
predictions = model.predict(x = [age_test, sex_test, chest_pain_type_test, resting_blood_pressure_test, serium_cholestoral_test, fasting_blood_sugar_test, resting_ECG_test, maximum_heart_rate_test, exercise_induced_angina_test, oldpeak_test, peak_exercise_slope_test, major_vessels_test, thal_test],
              batch_size = 41,
              verbose = 0)
rounded_predictions = np.argmax(predictions, axis = -1)


# Creates graphical representations of the model
cm = confusion_matrix(y_true = labels_test, y_pred = rounded_predictions)

plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')

cm_plot_labels = ['No Heart Disease', 'Has Heart Disease']
plot_confusion_matrix(cm = cm, classes = cm_plot_labels, title = "Confusion Matrix")

SVG(model_to_dot(model, show_shapes = True, show_layer_names = True, dpi = 65).create(prog = 'dot', format = 'svg'))

model = ResNet50(weights = None)

# Prints representation of neural network layers
SVG(model_to_dot(model, show_shapes= True, show_layer_names=True, dpi=65).create(prog='dot', format='svg'))
plot_model(model, to_file='model.png')





# Generate 20 random indices for test samples
num_tests = 20
random_test_indices = np.random.choice(len(age_test), num_tests, replace=False)

# Display the predicted values versus the true values for the selected test samples
print("Predicted Values vs True Values for 20 Test Samples:")
for index in random_test_indices:
    predicted_value = rounded_predictions[index]
    true_value = labels_test[index][0]  # Extracting the value without brackets

    print(f"Test Sample {index + 1}: Predicted Value = {predicted_value}, True Value = {true_value}")
