import pandas as pd


class Interpolator:
    """
    Class responsible for handling missing data or outliers 
    by interpolation (linear, spline, etc.).
    """
    def __init__(self, method='linear'):
        self.method = method

    def interpolate(self, df: pd.DataFrame, columns) -> pd.DataFrame:
        for col in columns:
            if col not in df.columns:
                continue
            df[col] = df[col].interpolate(method=self.method, limit_direction='both')
        return df
