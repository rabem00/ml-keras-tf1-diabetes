#!/usr/bin/env python3

from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model
import numpy

# Set a fixed random seed for reproducibility
numpy.random.seed(7)

# Loading the pima indians dataset
dataset = numpy.loadtxt("pima-indians-diabetes.data.csv", delimiter=",")

# Split the dataset into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]

# Create a model
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# More examples using other loss functions
#model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
#model.compile(loss='mean_squared_logarithmic_error', optimizer='adam', metrics=['accuracy'])

# Fit the model
model.fit(X, Y, epochs=150, batch_size=10,  verbose=2)

# Evaluate the mode and print the accuracy
scores = model.evaluate(X, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# Calculate predictions and print number of matching result
predictions = model.predict(X)
# Round predictions
rounded = [round(x[0]) for x in predictions]
match = 0
nomatch = 0
for i in range(len(rounded)):
    if rounded[i] == Y[i]:
        match += 1
    else:
        nomatch += 1
print("match: %i" % (match))
print("no match: %i" % (nomatch))

# Save the model for later use and delete model from memory
model.save('pima_indians_diabetes_model.h5')
del model  # deletes the existing model
