from proj1_helpers import load_csv_data
import matplotlib.pyplot as pyplot
import numpy

import random

data_path = 'data/train.csv'
y, tX, ids = load_csv_data(data_path, sub_sample=False)

#print(y.shape) # (250000,)
#print(tX.shape) # (250000, 30) => 30 features
#print(ids.shape) # (250000,)

data1 = tX[y == -1]
data2 = tX[y == 1]

print(data1.shape) # (164333, 30)
print(data2.shape) # (85667, 30)

print(data1[:,1].shape)



# for each feature, display the histogram with color = label (y)

for i in range(tX.shape[1]):

    maximum = max(tX[:,i])
    minumum = min(tX[:,i])

    x = data1[:,i]
    y = data2[:,i]

    bins = numpy.linspace(minumum, maximum, 100)

    pyplot.hist(x, bins, alpha=0.5, label='x')
    pyplot.hist(y, bins, alpha=0.5, label='y')
    pyplot.legend(loc='upper right')
    pyplot.title('Feature n° ' + str(i))
    pyplot.show()

