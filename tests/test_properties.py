#!/usr/bin/env python3
# encoding: utf-8

import unittest
from constants import MOLECULES_SMILES
from emp.density_function import predict_density
from emp.heat_of_formation_function import predict_heat_of_formation
from emp.detonation_pressure_function import predict_detonation_pressure
from emp.detonation_velocity_function import predict_detonation_velocity
from emp.sascore import calculate_sascore
from emp.ob_function import calculate_ob_percentage
import numpy

class TestProperties(unittest.TestCase):

    def setUp(self):
        self.ethanol_smiles = MOLECULES_SMILES.get("Ethanol")  # Ethanol
        self.tnt_smiles = MOLECULES_SMILES.get("TNT")  # TNT
        self.invalid_smiles = "COCOCOCOCOC11##111OOO"  # invalid smile

    def test_density(self):
        for smi in [self.ethanol_smiles, self.tnt_smiles]:
            density_prediction = predict_density(smi)
            self.assertIsInstance(density_prediction, numpy.float32)
        invalid_density_prediction = predict_density(self.invalid_smiles)
        self.assertEqual(invalid_density_prediction, None)

    def test_heat_of_formation(self):
        for smi in [self.ethanol_smiles, self.tnt_smiles]:
            hof_prediction = predict_heat_of_formation(smi)
            self.assertIsInstance(hof_prediction, numpy.float32)
        invalid_hof_prediction = predict_heat_of_formation(self.invalid_smiles)
        self.assertEqual(invalid_hof_prediction, None)

    def test_detonation_pressure(self):
        for smi in [self.ethanol_smiles, self.tnt_smiles]:
            pressure_prediction = predict_detonation_pressure(smi)
            self.assertIsInstance(pressure_prediction, numpy.float32)
        invalid_pressure_prediction = predict_detonation_pressure(self.invalid_smiles)
        self.assertEqual(invalid_pressure_prediction, None)

    def test_detonation_velocity(self):
        for smi in [self.ethanol_smiles, self.tnt_smiles]:
            velocity_prediction = predict_detonation_velocity(smi)
            self.assertIsInstance(velocity_prediction, numpy.float32)
        invalid_velocity_prediction = predict_detonation_velocity(self.invalid_smiles)
        self.assertEqual(invalid_velocity_prediction, None)

    def test_ob(self):
        for smi in [self.ethanol_smiles, self.tnt_smiles]:
            ob_prediction = calculate_ob_percentage(smi)
            self.assertIsInstance(ob_prediction, tuple)
            for pred in ob_prediction:
                self.assertIsInstance(pred,float)
        invalid_ob_prediction = calculate_ob_percentage(self.invalid_smiles)
        self.assertEqual(invalid_ob_prediction, None)

    def test_sascore(self):
        for smi in [self.ethanol_smiles, self.tnt_smiles]:
            sascore_prediction = calculate_sascore(smi)
            self.assertIsInstance(sascore_prediction, float)
            self.assertGreaterEqual(sascore_prediction,1)
            self.assertLessEqual(sascore_prediction,10)
        invalid_sascore_prediction = calculate_sascore(self.invalid_smiles)
        self.assertEqual(invalid_sascore_prediction, None)



if __name__ == "__main__":
    unittest.main()