import numpy as np
import pandas as pd
import torch
from lightning import pytorch as pl
from pathlib import Path
from chemprop import data, featurizers, models
from rdkit import Chem

# Load path to the checkpoint file.
checkpoint_path = Path(__file__).resolve().parent.parent / "models/best_heat_of_formation_training_2.0.ckpt"
mpnn = models.MPNN.load_from_checkpoint(checkpoint_path)

def predict_heat_of_formation(molecule):
    """
    Function to predict the value of the solid phase heat of formation for a given molecule
    Argument:
        molecule (Union[str, rdkit.Chem.rdchem.Mol]): Either a SMILES string or an RDKit Molecule object.
    Returns:
        solid phase heat of formation prediction (float): Predicted value by the trained NN.
                                                Returns None if the calculation fails or the molecule is invalid.
    """
    try:
        # Convert RDKit Molecule object to SMILES string if necessary
        if isinstance(molecule, Chem.rdchem.Mol):
            smiles = Chem.MolToSmiles(molecule)
        elif isinstance(molecule, str):
            smiles = molecule
        else:
            raise TypeError("Input must be either a SMILES string or an RDKit Mol object.")
        test_datapoint = [data.MoleculeDatapoint.from_smi(smiles)]
        featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
        test_dset = data.MoleculeDataset(test_datapoint, featurizer)
        test_loader = data.build_dataloader(test_dset, shuffle=False)

        with torch.inference_mode():
            trainer = pl.Trainer(logger=None, enable_progress_bar=False, accelerator="auto", devices=1)
            test_preds = trainer.predict(mpnn, test_loader)

        test_preds = np.concatenate(test_preds, axis=0)
        heat_of_formation_pred = test_preds[0][0]
        return heat_of_formation_pred
    except Exception as e:
        print(f"Error predicting heat of formation: {e}")
        return None
