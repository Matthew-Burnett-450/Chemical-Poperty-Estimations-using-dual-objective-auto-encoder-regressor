import matplotlib.pyplot as plt
import numpy as np
import json
from collections import defaultdict

# load the data from json file
with open('HydrocarbonData.json', 'r') as infile:
    HydrocarbonData = json.load(infile)


def ShiftProcessing(data, binsize=1, maxshift=250):
    # Ensure data is a numpy array
    shift_list = np.array(data, dtype=np.float64)
    return shift_list







print(len(HydrocarbonData))
#run shift processing on all molecules
HydrocarbonData2 = HydrocarbonData.copy()
for element in HydrocarbonData2:
    shift_list = element['13C_shift']
    if len(shift_list) == 0:
        HydrocarbonData.remove(element)
        print('Removed:',element['MolName'])
        continue
    #check for if there are no positive values

    shift_list = ShiftProcessing(shift_list)

    #check for NaN or empty lists or inf or only 0
    if np.isnan(shift_list).any() or len(shift_list) == 0 or np.isinf(shift_list).any() or np.all(shift_list == 0):
        HydrocarbonData.remove(element)
        continue

    element['13C_shift'] = shift_list.tolist()


#save the data as json file
with open('HydrocarbonDataProcessed.json', 'w') as outfile:
    json.dump(HydrocarbonData, outfile)

