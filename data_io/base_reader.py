import pandas as pd
from abc import ABC, abstractmethod

class BaseReader(ABC):
    """
    Abstract base class for all keypoint data readers.
    Define the interface that every reader must implement.
    """
    @abstractmethod
    def load_json(self, input_path: str) -> pd.DataFrame:
        """Load raw data from a specific file format and return a standardized DataFrame."""
        pass
    
    @abstractmethod
    def load_csv(self, input_path: str) -> pd.DataFrame:
        """Load raw data from a CSV file and return a standardized DataFrame."""
        return pd.read_csv(input_path)
