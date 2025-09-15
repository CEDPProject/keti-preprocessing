import pandas as pd
import numpy as np
from prophet import Prophet  # prophet 패키지 사용
from sktime.transformations.series.impute import TimeSeriesImputer

class DLImputation():
    """
    This class handles deep learning-based imputation methods such as Prophet and sktime.
    """

    def __init__(self, data, method, parameter):
        """
        Initialize the DLImputation class with data, method, and parameters.

        Args:
            data (DataFrame): The input data with missing values.
            method (str): The deep learning imputation method to be used (e.g., 'Prophet' or 'sktime').
            parameter (dict): Additional parameters for the model.
        """
        self.data = data
        self.method = method
        self.parameter = parameter

    def getResult(self):
        """
        Returns the imputed data using the specified deep learning method.

        Returns:
            DataFrame: The imputed data.
        """
        if self.method == 'Prophet':
            return self.apply_prophet()
        elif self.method == 'sktime':
            return self.apply_sktime()
        else:
            print(f"Method {self.method} not implemented.")
            return self.data

    def apply_prophet(self):
        """
        Apply Prophet to impute missing values.

        Returns:
            DataFrame: The imputed data using Prophet.
        """
        # Reset index for Prophet compatibility and rename columns
        df = self.data.reset_index()
        df.columns = ['ds', 'y']
        
        # Fit Prophet model on the data
        model = Prophet()
        model.fit(df)
        
        # Make predictions to fill in missing values
        future = model.make_future_dataframe(periods=0, freq='D')
        forecast = model.predict(future)
        
        # Replace missing values in original data with forecasted values
        forecast['yhat'] = np.where(self.data.isna(), forecast['yhat'], self.data.values.flatten())
        return forecast[['ds', 'yhat']].set_index('ds').rename(columns={'yhat': 'value'})

    def apply_sktime(self):
        """
        Apply sktime's TimeSeriesImputer to impute missing values.

        Returns:
            DataFrame: The imputed data using sktime.
        """
        # Initialize sktime imputer with nearest neighbor method
        imputer = TimeSeriesImputer(method="nearest")
        return imputer.fit_transform(self.data)
