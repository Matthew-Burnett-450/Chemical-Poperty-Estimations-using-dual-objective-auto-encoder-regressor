import numpy as np


DataSet=[]
Mol_dictionary = {}
shift_list = []
firstiter = 0 

with open(r'NMRData\nmrshiftdb2withsignals.sd', 'r', encoding='utf-8') as file:
    # Iterate over each line in the file
    lastline = ''
    for line in file:
        line=line.strip(str('\n'))
        if firstiter == 0:
            if lastline.startswith('$$$$'):
                firstiter = 1
            else:
                lastline = line
                continue
        if lastline.startswith('$$$$'):
            if Mol_dictionary=={}:
                pass
            else:
                Mol_dictionary['13C_shift'] = shift_list
                shift_list = []
                DataSet.append(Mol_dictionary)
                Mol_dictionary = {}    
            Mol_dictionary['MolName'] = line

        if lastline.startswith('> <INChI key>'):
            Mol_dictionary['INChI key'] = line
        if lastline.startswith('> <INChI>'):
            Mol_dictionary['INChI'] = line

        if lastline.startswith('> <nmrshiftdb2 ID>'):
            Mol_dictionary['NMRShiftDB ID'] = line
        if lastline.startswith('> <Spectrum 13C'):
            for shift in line.split('|')[:-1]:
                [shift_val, _, shift_idx] = shift.split(';')
                shift_val, shift_idx = float(shift_val), int(shift_idx)
                shift_list.append([shift_val, shift_idx])



        lastline = line
        print(len(DataSet))

print(DataSet)

# save the data as json file
import json
with open('NMRData.json', 'w') as outfile:
    json.dump(DataSet, outfile)
        
        