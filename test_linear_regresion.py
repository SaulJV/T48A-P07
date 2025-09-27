import unittest
import numpy as np
import pandas as pd
from linear_regresion import linear_regresion

class TestPandasExercises(unittest.TestCase):
    
    def test_linear_regresion(self):
        mse, r2 = linear_regresion() 
        assert np.isclose(mse, 4198.86,  1) and np.isclose(r2, 0.23, 0.2)
