import unittest
import import_ipynb
import pandas as pd
import pandas.testing as pd_testing

class Test(unittest.TestCase):
    def setUp(self):
        import Exercise_15_04_Ensemble_learning_Max_Voting_v1_0
        self.exercises = Exercise_15_04_Ensemble_learning_Max_Voting_v1_0
        self.filename = '../Dataset/crx.data'
        self.credData = pd.read_csv(self.filename,sep=",",header = None,na_values = "?")
        self.dataShape = self.credData.shape

    def test_file_url(self):
        self.assertEqual(self.exercises.filename, self.filename)
        

    def test_shape(self):
        self.assertEqual(self.exercises.credData.shape, self.dataShape)


if __name__ == '__main__':
    unittest.main()
