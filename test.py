import pytest as pt
import unittest
import pandas as pd
from model import SimpleDecisionTreeRegressor, SimpleRandomForestRegressor

class TestModels(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Create a sample dataset for testing
        cls.X = pd.DataFrame({
            'Total_Stops': [1],
            'Journey_day': [3],
            'Journey_month': [2],
            'Journey_year': [2024],
            'Dep_hour': [12],
            'Dep_min': [20],
            'Arrival_hour': [14],
            'Arrival_min': [20],
            'Duration_hours': [2],
            'Duration_mins': [0],
            'Airline_Air India': [1],
            'Airline_Buddha Air': [0],
            'Airline_GoAir': [0],
            'Airline_IndiGo': [0],
            'Airline_Jet Airways': [0],
            'Airline_Multiple carriers': [0],
            'Airline_Shree Airlines': [0],
            'Airline_SpiceJet': [0],
            'Airline_Vistara': [0],
            'Airline_Yeti Airlines': [0],
            'Source_Bhadrapur': [0],
            'Source_Bhairahawa': [0],
            'Source_Bharatpur': [0],
            'Source_Biratnagar': [0],
            'Source_Chennai': [1],
            'Source_Delhi': [0],
            'Source_Dhangadi': [0],
            'Source_Janakpur': [0],
            'Source_Kathmandu': [0],
            'Source_Kolkata': [0],
            'Source_Mumbai': [0],
            'Source_Nepalgunj': [0],
            'Source_Pokhara': [0],
            'Source_Rajbiraj': [0],
            'Source_Simara': [0],
            'Destination_Bhadrapur': [0],
            'Destination_Bhairahawa': [0],
            'Destination_Bharatpur': [0],
            'Destination_Biratnagar': [1],
            'Destination_Cochin': [0],
            'Destination_Delhi': [0],
            'Destination_Dhangadi': [0],
            'Destination_Hyderabad': [0],
            'Destination_Janakpur': [0],
            'Destination_Kathmandu': [0],
            'Destination_Kolkata': [0],
            'Destination_Nepalgunj': [0],
            'Destination_New Delhi': [0],
            'Destination_Pokhara': [0],
            'Destination_Rajbiraj': [0],
            'Destination_Simara': [0]
        })
        cls.y = pd.Series([1000])

    def test_dt_fit_predict(self):
        dt_regressor = SimpleDecisionTreeRegressor()
        dt_regressor.fit(self.X, self.y, max_depth=10, min_samples_split=10)
        predictions = dt_regressor.predict(self.X)
        self.assertEqual(len(predictions), len(self.y))


    def test_rf_fit_predict(self):

        rf_regressor = SimpleRandomForestRegressor(n_estimators=10, max_depth=10, min_samples_split=10)
        rf_regressor.fit(self.X, self.y)
        predictions = rf_regressor.predict(self.X)
        self.assertEqual(len(predictions), len(self.y))


if __name__ == '__main__':
    unittest.main()