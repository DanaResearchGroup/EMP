from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import Descriptors
import re

def calculate_ob_percentage(molecule):
    """
    Function to calculate Oxygen Bonds percentage (OB%) for a given molecule.

    Arguments:
        molecule (Union[str, rdkit.Chem.rdchem.Mol]): Either a SMILES string or an RDKit Molecule object.

    Returns:
        ob1600_percentage, ob100_percentage (float): Oxygen bond percentage for both OB100 and OB1600 formulas.
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

        formula = rdMolDescriptors.CalcMolFormula(mol)

        # Regular expression to match elements and their counts
        element_counts = re.findall(r'([A-Z][a-z]*)(\d*)', formula)

        # Initialize counts for C, H, O, and M
        carbon_count = 0
        hydrogen_count = 0
        oxygen_count = 0
        metal_count = 0  # assuming 0 at the moment
        # Loop through the element counts and assign values to C, H, O
        for element, count in element_counts:
            count = int(count) if count else 1
            if element == 'C':
                carbon_count = count
            elif element == 'H':
                hydrogen_count = count
            elif element == 'O':
                oxygen_count = count

        # Calculate molecular weight
        mw = Descriptors.ExactMolWt(mol)
        if mw == 0:
            raise ValueError("Molecular weight cannot be zero.")

        # Calculate OB%
        n_atoms = carbon_count + hydrogen_count + oxygen_count + metal_count
        ob1600_percentage = (-1600 / mw) * (2 * carbon_count + hydrogen_count / 2 + metal_count - oxygen_count)
        ob100_percentage = (100 / n_atoms) * (oxygen_count - 2 * carbon_count - hydrogen_count / 2)
        return ob1600_percentage, ob100_percentage

    except Exception as e:
        print(f"Error calculating OB%: {e}")
        return None
