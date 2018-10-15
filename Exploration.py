from proj1_helpers import load_csv_data
import matplotlib.pyplot as pyplot
import numpy

import random

data_path = 'data/train.csv'
y, tX, ids = load_csv_data(data_path, sub_sample=False)

#print(y.shape) # (250000,)
#print(tX.shape) # (250000, 30) => 30 features
#print(ids.shape) # (250000,)

nSample, nFeature = tX.shape

data1 = tX[y == -1]
data2 = tX[y == 1]

print(data1.shape) # (164333, 30)
print(data2.shape) # (85667, 30)

print(data1[:,1].shape)

## feature ranking according to the correlation between feature and label

cor = numpy.zeros(nFeature)
for iFeature in range(nFeature):
    #print(tX[:,iFeature])
    #print(y)
    tmp = numpy.corrcoef(tX[:,iFeature],y)**2
    cor[iFeature] = tmp[1][0]
orderedPower = -numpy.sort(-cor)
orderedInd = sorted(range(nFeature), key=lambda k: -cor[k])
#print(orderedPower)
print(orderedInd)

## for each feature, display the histogram with color = label (y)

if True:
    for i in range(nFeature):

        maximum = max(tX[:,i])
        minimum = min(tX[:,i])

        x = data1[:,i]
        y = data2[:,i]

        bins = numpy.linspace(minimum, maximum, maximum-minimum)

        pyplot.hist(x, bins, alpha=0.5, label='x')
        pyplot.hist(y, bins, alpha=0.5, label='y')
        pyplot.legend(loc='upper right')
        pyplot.title('Feature n° ' + str(i))
        pyplot.show()

