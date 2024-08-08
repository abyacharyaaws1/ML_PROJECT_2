import json
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from src.exception import CustomException
from src.logger import logger
from dataclasses import dataclass
import joblib

@dataclass
class ModelTrainerConfig:
    transformed_train_data_path: str
    transformed_test_data_path: str
    model_output_path: str
    evaluation_metrics_path: str

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def load_data(self):
        """
        Load transformed training and test data from CSV files.
        """
        try:
            train_data = pd.read_csv(self.config.transformed_train_data_path)
            test_data = pd.read_csv(self.config.transformed_test_data_path)
            logger.info(f"Data loaded successfully from {self.config.transformed_train_data_path} and {self.config.transformed_test_data_path}")
            return train_data, test_data
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise CustomException(f"Error loading data: {e}")

    def train_and_evaluate_models(self):
        """
        Train multiple models and evaluate their performance.
        """
        try:
            # Load data
            train_data, test_data = self.load_data()

            # Separate features and target
            X_train = train_data.drop(columns=['math score'])
            y_train = train_data['math score']
            X_test = test_data.drop(columns=['math score'])
            y_test = test_data['math score']

            # Check if target variable is continuous or discrete
            if y_train.nunique() > 10:  # Assuming more than 10 unique values indicate a regression task
                logger.info("Regression task detected.")
                models = {
                    'LinearRegression': LinearRegression()
                }
                best_model = None
                best_r2_score = float('-inf')  # For regression, we use R2 score

                # Train and evaluate models
                for name, model in models.items():
                    logger.info(f"Training {name}...")
                    model.fit(X_train, y_train)
                    predictions = model.predict(X_test)
                    r2 = r2_score(y_test, predictions)
                    logger.info(f"{name} R2 Score: {r2}")

                    if r2 > best_r2_score:
                        best_r2_score = r2
                        best_model = model
                        logger.info(f"New best model found: {name} with R2 Score: {r2}")

                if best_model:
                    # Save the best model
                    joblib.dump(best_model, self.config.model_output_path)
                    logger.info(f"Best model saved to {self.config.model_output_path}")

                    # Save evaluation metrics
                    metrics = {
                        'best_model': best_model.__class__.__name__,
                        'r2_score': best_r2_score
                    }
                    with open(self.config.evaluation_metrics_path, 'w') as f:
                        json.dump(metrics, f)
                    logger.info(f"Evaluation metrics saved to {self.config.evaluation_metrics_path}")

            else:
                logger.info("Classification task detected.")
                models = {
                    'KNeighborsClassifier': KNeighborsClassifier(),
                    'AdaBoostClassifier': AdaBoostClassifier(),
                    'XGBClassifier': XGBClassifier(use_label_encoder=False),
                    'CatBoostClassifier': CatBoostClassifier(learning_rate=0.1, iterations=500, depth=10, verbose=0)
                }

                best_model = None
                best_accuracy = 0  # For classification, we use accuracy score

                # Train and evaluate models
                for name, model in models.items():
                    logger.info(f"Training {name}...")
                    model.fit(X_train, y_train)
                    predictions = model.predict(X_test)
                    accuracy = accuracy_score(y_test, predictions)
                    logger.info(f"{name} Accuracy: {accuracy}")

                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_model = model
                        logger.info(f"New best model found: {name} with Accuracy: {accuracy}")

                if best_model:
                    # Save the best model
                    joblib.dump(best_model, self.config.model_output_path)
                    logger.info(f"Best model saved to {self.config.model_output_path}")

                    # Save evaluation metrics
                    metrics = {
                        'best_model': best_model.__class__.__name__,
                        'accuracy': best_accuracy
                    }
                    with open(self.config.evaluation_metrics_path, 'w') as f:
                        json.dump(metrics, f)
                    logger.info(f"Evaluation metrics saved to {self.config.evaluation_metrics_path}")

        except Exception as e:
            logger.error(f"Error in model training and evaluation: {e}")
            raise CustomException(f"Error in model training and evaluation: {e}")

if __name__ == "__main__":
    try:
        # Load configuration from JSON file
        with open('config.json', 'r') as config_file:
            config_data = json.load(config_file)

        # Initialize ModelTrainerConfig with loaded data
        config = ModelTrainerConfig(
            transformed_train_data_path=config_data['transformed_train_data_path'],
            transformed_test_data_path=config_data['transformed_test_data_path'],
            model_output_path=config_data['model_output_path'],
            evaluation_metrics_path=config_data['evaluation_metrics_path']
        )

        # Initialize and run model trainer
        model_trainer = ModelTrainer(config)
        model_trainer.train_and_evaluate_models()
    except CustomException as e:
        logger.error(e)
        print(e)
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        print(f"An unexpected error occurred: {e}")


'''
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
'''
'''
import pandas as pd
import json
import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report
from src.exception import CustomException
from src.logger import logger
from dataclasses import dataclass
import sys

@dataclass
class ModelTrainerConfig:
    transformed_train_data_path: str  # Path to the transformed training data CSV file
    transformed_test_data_path: str   # Path to the transformed test data CSV file
    model_save_path: str               # Path to save the trained model

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def load_data(self):
        """
        Load the transformed training and test data.
        """
        try:
            train_data = pd.read_csv(self.config.transformed_train_data_path)
            test_data = pd.read_csv(self.config.transformed_test_data_path)
            logger.info(f"Transformed training data loaded from {self.config.transformed_train_data_path}")
            logger.info(f"Transformed test data loaded from {self.config.transformed_test_data_path}")
            return train_data, test_data
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise CustomException(f"Error loading data: {e}", sys)

    def train_and_evaluate(self, model, param_grid, X_train, y_train, X_test, y_test):
        """
        Train and evaluate the model with hyperparameter tuning.
        """
        try:
            grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy', cv=3, n_jobs=-1, verbose=1)
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            predictions = best_model.predict(X_test)
            report = classification_report(y_test, predictions)
            logger.info(f"Model: {model.__class__.__name__}")
            logger.info(f"Best Parameters: {grid_search.best_params_}")
            logger.info(f"Classification Report:\n{report}")
            return best_model
        except Exception as e:
            logger.error(f"Error during model training or evaluation: {e}")
            raise CustomException(f"Error during model training or evaluation: {e}", sys)

    def save_model(self, model):
        """
        Save the trained model.
        """
        try:
            joblib.dump(model, self.config.model_save_path)
            logger.info(f"Model saved to {self.config.model_save_path}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise CustomException(f"Error saving model: {e}", sys)

    def train_model(self):
        """
        Train multiple models with hyperparameter tuning and save the best model.
        """
        try:
            # Load data
            train_data, test_data = self.load_data()

            # Separate features and target variable
            X_train = train_data.drop(columns=['math score'])
            y_train = train_data['math score']
            X_test = test_data.drop(columns=['math score'])
            y_test = test_data['math score']

            # Define models and hyperparameters
            models = {
                'RandomForest': (RandomForestClassifier(), {
                    'n_estimators': [100, 200],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5]
                }),
                'DecisionTree': (DecisionTreeClassifier(), {
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5]
                }),
                'GradientBoosting': (GradientBoostingClassifier(), {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.01, 0.1, 1],
                    'max_depth': [3, 5, 7]
                }),
                'LogisticRegression': (LogisticRegression(max_iter=1000), {
                    'penalty': ['l2'],
                    'C': [0.1, 1, 10]
                }),
                'KNeighbors': (KNeighborsClassifier(), {
                    'n_neighbors': [3, 5, 7],
                    'weights': ['uniform', 'distance']
                }),
                'XGBClassifier': (XGBClassifier(), {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.01, 0.1, 1],
                    'max_depth': [3, 5, 7]
                }),
                'CatBoost': (CatBoostClassifier(verbose=0), {
                    'iterations': [100, 200],
                    'depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1, 1]
                }),
                'AdaBoost': (AdaBoostClassifier(), {
                    'n_estimators': [50, 100],
                    'learning_rate': [0.01, 0.1, 1]
                })
            }

            best_model = None
            best_score = 0

            # Train and evaluate each model
            for name, (model, param_grid) in models.items():
                logger.info(f"Training {name}...")
                trained_model = self.train_and_evaluate(model, param_grid, X_train, y_train, X_test, y_test)
                score = trained_model.score(X_test, y_test)
                if score > best_score:
                    best_score = score
                    best_model = trained_model

            # Save the best model
            if best_model:
                self.save_model(best_model)
                logger.info(f"Best model saved with accuracy: {best_score}")
            else:
                logger.error("No model was trained.")
        except Exception as e:
            logger.error(f"Error in training models: {e}")
            raise CustomException(f"Error in training models: {e}", sys)

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
'''