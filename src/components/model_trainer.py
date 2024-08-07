import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from src.exception import CustomException
from src.logger import logger
from dataclasses import dataclass
import joblib
import json

@dataclass
class ModelTrainerConfig:
    transformed_train_data_path: str  # Path to the transformed training data CSV file
    transformed_test_data_path: str   # Path to the transformed test data CSV file
    model_save_path: str               # Path to save the trained model

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train_model(self):
        try:
            # Load the transformed training and test data
            train_data = pd.read_csv(self.config.transformed_train_data_path)
            test_data = pd.read_csv(self.config.transformed_test_data_path)
            logger.info(f"Transformed training data loaded from {self.config.transformed_train_data_path}")
            logger.info(f"Transformed test data loaded from {self.config.transformed_test_data_path}")

            # Separate features and target variable
            X_train = train_data.drop(columns=['math score'])
            y_train = train_data['math score']
            X_test = test_data.drop(columns=['math score'])
            y_test = test_data['math score']

            # Initialize and train the model
            model = LinearRegression()
            model.fit(X_train, y_train)
            logger.info("Model training completed")

            # Make predictions
            predictions = model.predict(X_test)
            mse = mean_squared_error(y_test, predictions)
            logger.info(f"Mean Squared Error on test data: {mse}")

            # Save the model
            joblib.dump(model, self.config.model_save_path)
            logger.info(f"Model saved to {self.config.model_save_path}")

        except FileNotFoundError as e:
            logger.error(f"File not found: {str(e)}")
            raise CustomException(f"File not found: {str(e)}")
        except pd.errors.EmptyDataError:
            logger.error(f"Data file is empty.")
            raise CustomException(f"Data file is empty.")
        except pd.errors.ParserError:
            logger.error(f"Error parsing data file.")
            raise CustomException(f"Error parsing data file.")
        except Exception as e:
            logger.error(f"An unexpected error occurred: {str(e)}")
            raise CustomException(f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    try:
        # Load configuration from JSON file
        with open('config.json', 'r') as config_file:
            config_data = json.load(config_file)

        # Initialize ModelTrainerConfig with loaded data
        config = ModelTrainerConfig(
            transformed_train_data_path=config_data['transformed_train_data_path'],
            transformed_test_data_path=config_data['transformed_test_data_path'],
            model_save_path=config_data['model_save_path']
        )

        # Initialize and run model training
        model_trainer = ModelTrainer(config)
        model_trainer.train_model()
    except CustomException as e:
        print(e)
    except Exception as e:
        logger.error(f"An error occurred while reading the configuration file: {e}")
        print(f"An error occurred while reading the configuration file: {e}")
