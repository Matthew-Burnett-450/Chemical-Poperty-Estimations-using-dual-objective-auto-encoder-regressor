import json
import numpy as np

# load the data from json file NMRData.json
with open('NMRData.json', 'r') as infile:
    NMRData = json.load(infile)

# filter out empty 13C shifts
NMRData = list(filter(lambda x: len(x['13C_shift']) > 0, NMRData))
print(len(NMRData))

# save the data as json file
with open('UnlabledNMRData.json', 'w') as outfile:
    json.dump(NMRData, outfile)