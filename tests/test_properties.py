#!/usr/bin/env python3
# encoding: utf-8

import unittest
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../emp')))
from density_function import density_prediction



class TestProperties(unittest.TestCase):

    def setUp(self):
        self.ethanol_smiles = MOLECULES_SMILES.get("Ethanol")  # Ethanol
        self.tnt_smiles = MOLECULES_SMILES.get("TNT")  # TNT
        self.invalid_smiles = "COCOCOCOCOC11##111OOO"  # invalid smile
        self.ethanol = Molecule(self.ethanol_smiles)
        self.tnt = Molecule(self.tnt_smiles)

    def test_density(self):




    def test_detonation_velocity(self):



    def test_detonation_pressure(self):




    def test_heat_of_formation(self):



if __name__ == "__main__":
    unittest.main()