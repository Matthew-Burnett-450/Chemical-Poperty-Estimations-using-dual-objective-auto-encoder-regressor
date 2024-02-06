from rdkit import Chem
import json

def is_alkane(inchi):
    # Convert InChI to an RDKit molecule
    mol = Chem.MolFromInchi(inchi,sanitize=True)

    # Check if the molecule conversion was successful
    if mol is None:
        return False

    # Check all atoms are either carbon or hydrogen
    if not all(atom.GetAtomicNum() in [1, 6] for atom in mol.GetAtoms()):
        return False

    """        # Check for only single bonds in the molecule
        for bond in mol.GetBonds():
            if bond.GetBondType() != Chem.rdchem.BondType.SINGLE:
                return False"""

    return True

#load NMR data
with open('NMRData.json', 'r') as infile:
    NMRData = json.load(infile)

#filter out molecules that are not alkanes
HydrocarbonData = list(filter(lambda x: is_alkane(x['INChI']), NMRData))

print(len(HydrocarbonData))

#save the data as json file
with open('HydrocarbonData.json', 'w') as outfile:
    json.dump(HydrocarbonData, outfile)