import json
import numpy as np
import matplotlib.pyplot as plt

epoch = 9600

json_file_name = 'barcodes/epoch_' + str(epoch) + '_cleaned.json'
ground_truth_file_name = 'barcodes/ground_truth_mini_dataset.json'
file = open(json_file_name)
ground_truth_file = open(ground_truth_file_name)
obj = json.load(file)
gt = json.load(ground_truth_file)

dim0_bar_lengths = []
dim1_bar_lengths = []

for key in obj:
    barcode = obj[key]
    barcode0 = barcode['dim0']
    for component in barcode0:
        length = component[0] - component[1]
        if length != np.inf:
            dim0_bar_lengths.append(length)

    barcode1 = barcode['dim1']
    for loop in barcode1:
        length = loop[0] - loop[1]
        if length != np.inf:
            dim1_bar_lengths.append(length)

f = plt.figure()
plt.title('Dimension 0')
plt.ylabel('Frequency')
plt.xlabel('Bar length')
plt.hist(dim0_bar_lengths, bins=20)
f.savefig('barcode0.png')

f = plt.figure()
plt.title('Dimension 1')
plt.ylabel('Frequency')
plt.xlabel('Bar length')
plt.hist(dim1_bar_lengths, bins=20)
f.savefig('barcode1.png')

