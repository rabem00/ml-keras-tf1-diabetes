#!/usr/bin/env python3

from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model
import numpy

model = load_model('pima_indians_diabetes_model.h5')
# In this case it should be 0,1  -> none diabetes and diabetes
T = numpy.array([[1,93,70,31,0,30.4,0.315,23],[1,126,60,0,0,30.1,0.349,47]])

# Get the predictions, round them en print result
predictions = model.predict(T)
rounded = [round(x[0]) for x in predictions]
print(rounded)