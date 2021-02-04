import unittest
import import_ipynb
import pandas as pd
import pandas.testing as pd_testing
import numpy.testing as np_testing
from sklearn.cluster import KMeans

class Test(unittest.TestCase):
	def setUp(self):
		import Exercise10_02
		self.exercises = Exercise10_02
		
		self.file_url = 'https://raw.githubusercontent.com/PacktWorkshops/The-Data-Science-Workshop/master/Chapter10/dataset/ames_iowa_housing.csv'
		self.df = pd.read_csv(self.file_url)
		self.obj_df = self.df.select_dtypes(include='object')
		self.obj_cols = self.obj_df.columns


	def test_file_url(self):
		self.assertEqual(self.exercises.file_url, self.file_url)

	def test_df(self):
		pd_testing.assert_frame_equal(self.exercises.df, self.df)

	def test_obj_df(self):
		pd_testing.assert_frame_equal(self.exercises.obj_df, self.obj_df)

	def test_obj_cols(self):
		np_testing.assert_array_equal(self.exercises.obj_cols, self.obj_cols)

if __name__ == '__main__':
	unittest.main()
