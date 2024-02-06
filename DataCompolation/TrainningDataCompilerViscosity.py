import numpy as np
import json
from ThermoMlReader import ThermoMLParser
import os
from NistDataGrabberViscosity import TargetConstructor
import TDEEquations as TDEEq
# if HydrocarbonDataProcessed_In_NIST_TDE_Have_Viscosity does not exist, create it
if not os.path.exists('HydrocarbonDataProcessed_In_NIST_TDE_Have_Viscosity.json'):

    #load the data from json file
    with open('HydrocarbonDataProcessed.json', 'r') as infile:
        HydrocarbonData = json.load(infile)



    # Path to the DATA folder
    data_folder_path = 'NistData'
    # Check if the folder exists
    if not os.path.exists(data_folder_path):
        print(f"Folder '{data_folder_path}' does not exist.")
    else:
        # List to hold file names
        file_names = []

        # Iterate over each entry in the DATA folder
        for entry in os.listdir(data_folder_path):
            # Create full path
            full_path = os.path.join(data_folder_path, entry)

            # Check if it's a file and not a directory
            if os.path.isfile(full_path):
                #if ends in .xml
                if not entry.endswith('.xml'):
                    continue
                entry = entry.split('.')[0]
                file_names.append(entry)


    HydrocarbonData = [x for x in HydrocarbonData if x['MolName'] in file_names]

    #save the data as json file
    with open('HydrocarbonDataProcessed_In_NIST_TDE.json', 'w') as outfile:
        json.dump(HydrocarbonData, outfile)

    mask=[]

    for Mol in HydrocarbonData:
        filename = 'NistData/' + Mol['MolName'] + '.xml'
        parser = ThermoMLParser(filename)
        parser.extract_properties()
        parser.extract_equation_details()
        Properties = parser.get_properties()

        # Stack phase and property names
        AvailableProperties = np.column_stack((Properties['property_names'], Properties['property_phase']))
        AvailableProperties = [list(x) for x in AvailableProperties]
        if ['Viscosity, Pa*s', 'Liquid'] in AvailableProperties:
            print('True for file:', filename)
            mask.append(True)
        else:
            print('False for file:', filename)
            mask.append(False)

    print(len(HydrocarbonData))
    print(len(mask))
    #number of True values
    print(sum(mask))

    #delete all false values
    HydrocarbonData = [x for i,x in enumerate(HydrocarbonData) if mask[i]]

    print(len(HydrocarbonData))

        #generate file Paths and add them to the dictionary
    for Mol in HydrocarbonData:
        Path = 'NistData/' + Mol['MolName'] + '.xml'
        Mol['filename'] = Path

    #save the data as json file
    with open('HydrocarbonDataProcessed_In_NIST_TDE_Have_Viscosity.json', 'w') as outfile:
        json.dump(HydrocarbonData, outfile)
else:
    #load the data from json file
    with open('HydrocarbonDataProcessed_In_NIST_TDE_Have_Viscosity.json', 'r') as infile:
        HydrocarbonData = json.load(infile)

EqDict={'TDE.PPDS9':TDEEq.PPDS9,'TDE.NVCLSExpansion.Version1':TDEEq.ViscosityL}

Constructor=TargetConstructor(EqDict)

#grab list of all file paths
paths=[x['filename'] for x in HydrocarbonData]

A=[]
B=[]
print(len(paths))
for path in paths:
    A_,B_,_,_=Constructor.GenerateTargets(path)
    A.append(A_)
    B.append(B_)

print(len(A))
print(len(B))

#add A and B to the dictionary
for i,Mol in enumerate(HydrocarbonData):
    Mol['A']=A[i]
    Mol['B']=B[i]

#save the data as json file
with open('ViscosityTrainingData.json', 'w') as outfile:
    json.dump(HydrocarbonData, outfile)

#make an npz file with X and A,B
X=[x['13C_shift'] for x in HydrocarbonData]
A=np.array(A)
B=np.array(B)

