import unittest
import import_ipynb
import pandas as pd
import pandas.testing as pd_testing

class Test(unittest.TestCase):
    def setUp(self):
        import Exercise_13_04_Logistic_Regression_Model_with_MSMOTE_v1_0
        self.exercises = Exercise_13_04_Logistic_Regression_Model_with_MSMOTE_v1_0		
        self.filename = '../Dataset/bank-full.csv'	
        self.bankData = pd.read_csv(self.filename, sep=";")        
        self.dataShape = self.bankData.shape 

    def test_file_url(self):
        self.assertEqual(self.exercises.filename, self.filename)       
        

    def test_shape(self):
        self.assertEqual(self.exercises.bankData.shape, self.dataShape)		


if __name__ == '__main__':
    unittest.main()
