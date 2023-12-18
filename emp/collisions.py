from rdkit import Chem
from rdkit.Chem import AllChem
import itertools

def read_molecule_from_smiles(smiles):
    return Chem.MolFromSmiles(smiles)

def compute_collision_probability(molecule):
    # Compute atomic radii (for simplicity, assuming all atoms have the same radius)
    atomic_radii = {atom.GetAtomicNum(): AllChem.GetRvdw(atom.GetAtomicNum()) for atom in molecule.GetAtoms()}

    # Calculate collision probability based on pairwise atomic distances
    atoms = molecule.GetAtoms()
    num_collisions = 0
    for atom_pair in itertools.combinations(atoms, 2):
        dist = Chem.rdMolTransforms.GetBondLength(molecule.GetConformer(), atom_pair[0].GetIdx(), atom_pair[1].GetIdx())
        radius_sum = atomic_radii[atom_pair[0].GetAtomicNum()] + atomic_radii[atom_pair[1].GetAtomicNum()]
        if dist < radius_sum:
            num_collisions += 1

    total_possible_collisions = len(list(itertools.combinations(atoms, 2)))
    collision_probability = num_collisions / total_possible_collisions

    return collision_probability

if __name__ == "__main__":
    smiles = "CCO"
    molecule = read_molecule_from_smiles(smiles)

    if molecule:
        collision_probability = compute_collision_probability(molecule)
        print(f"Collision Probability: {collision_probability:.4f}")
    else:
        print("Failed to parse the molecule.")
