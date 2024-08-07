import pandas as pd
from sklearn.model_selection import train_test_split
from src.exception import CustomException
from src.logger import logger
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    raw_data_path: str          # Path to the raw data CSV file
    train_data_path: str        # Path to save training data CSV file
    test_data_path: str         # Path to save test data CSV file
    test_size: float = 0.3      # Proportion of data to use for testing
    random_state: int = 42      # Seed for reproducibility

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def ingest_data(self):
        """
        Method to ingest data from a CSV file, split it into training and test sets, and save them.
        """
        try:
            # Load data from CSV
            data = pd.read_csv(self.config.raw_data_path)
            logger.info(f"Data loaded successfully from {self.config.raw_data_path}")

            # Perform train-test split
            train_data, test_data = train_test_split(data, test_size=self.config.test_size, random_state=self.config.random_state)
            
            # Save train and test data
            train_data.to_csv(self.config.train_data_path, index=False)
            test_data.to_csv(self.config.test_data_path, index=False)
            
            logger.info(f"Training data saved to {self.config.train_data_path}")
            logger.info(f"Test data saved to {self.config.test_data_path}")
            logger.info(f"Data split into training and test sets with test size={self.config.test_size}")
            logger.info(f"Training set size: {train_data.shape[0]}")
            logger.info(f"Test set size: {test_data.shape[0]}")
            
        except FileNotFoundError:
            logger.error(f"The file {self.config.raw_data_path} was not found.")
            raise CustomException(f"The file {self.config.raw_data_path} was not found.")
        except pd.errors.EmptyDataError:
            logger.error(f"The file {self.config.raw_data_path} is empty.")
            raise CustomException(f"The file {self.config.raw_data_path} is empty.")
        except pd.errors.ParserError:
            logger.error(f"Error parsing the file {self.config.raw_data_path}.")
            raise CustomException(f"Error parsing the file {self.config.raw_data_path}.")
        except Exception as e:
            logger.error(f"An unexpected error occurred while loading data: {str(e)}")
            raise CustomException(f"An unexpected error occurred while loading data: {str(e)}")

if __name__ == "__main__":
    # Configuration for data ingestion
    config = DataIngestionConfig(
        raw_data_path=r'C:\Users\abyac\OneDrive\Documents\GitHub\ML_PROJECT_2\notebook\data\dataset.csv',          # Path to the raw data CSV file
        train_data_path=r'C:\Users\abyac\OneDrive\Documents\GitHub\ML_PROJECT_2\notebook\data\train_data.csv',     # Path where the training data CSV will be saved
        test_data_path=r'C:\Users\abyac\OneDrive\Documents\GitHub\ML_PROJECT_2\notebook\data\test_data.csv',       # Path where the test data CSV will be saved
        test_size=0.3,                            # Proportion of data to be used for testing
        random_state=42                           # Seed for reproducibility
    )
    
    # Initialize and run data ingestion
    data_ingestion = DataIngestion(config)
    try:
        data_ingestion.ingest_data()
    except CustomException as e:
        print(e)


# Basic checks
'''
    assert len(train_data) > 0, "Training data is empty"
    assert len(test_data) > 0, "Test data is empty"
    assert len(train_data) + len(test_data) == len(pd.read_csv(config.raw_data_path)), "Mismatch in data size"
    
    print("Data ingestion test passed.")

if __name__ == "__main__":
    test_data_ingestion()
    '''