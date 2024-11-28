from rdkit import Chem
from emp.sascore_helpers import sascorer as sa

def calculate_sascore(molecule):
    """
    Function to calculate the SAScore for a given molecule.
    Argument:
        molecule (Union[str, rdkit.Chem.rdchem.Mol]): Either a SMILES string or an RDKit Molecule object.
    Returns:
        sascore (float): SAScore, which represents the predicted difficulty of synthesizing a molecule.
                         Returns a value between 1 (easiest to synthesize) and 10 (hardest to synthesize).
                         Returns None if the calculation fails or the molecule is invalid.
    """
    try:
        # Convert SMILES string to RDKit Molecule object if necessary
        if isinstance(molecule, str):
            mol = Chem.MolFromSmiles(molecule)
            if mol is None:
                raise ValueError("Invalid SMILES string. Could not create RDKit Mol.")
        elif isinstance(molecule, Chem.rdchem.Mol):
            mol = molecule
        else:
            raise TypeError("Input must be either a SMILES string or an RDKit Mol object.")

        sascore = sa.calculateScore(mol)
    except Exception as e:
        print(f"Error calculating SAScore: {e}")
        sascore = None
    return sascore

