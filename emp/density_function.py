import numpy as np
import pandas as pd
import torch
from lightning import pytorch as pl
from pathlib import Path
from chemprop import data, featurizers, models
from rdkit import Chem

# Load path to the checkpoint file.
checkpoint_path = Path(__file__).resolve().parent.parent / "models/best_density_training.ckpt"
mpnn = models.MPNN.load_from_checkpoint(checkpoint_path)


def predict_density(molecule):
    """
    Function to predict the value of crystal phase density for a given molecule
    Argument:
        molecule (Union[str, rdkit.Chem.rdchem.Mol]): Either a SMILES string or an RDKit Molecule object.
    Returns:
        density prediction (float): Predicted value by the trained NN.
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
        density_pred = test_preds[0][0][0]
        return density_pred
    except Exception as e:
        print(f"Error predicting density: {e}")
        return None
