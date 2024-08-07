import os
import pandas as pd
import json
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from src.exception import CustomException
from src.logger import logger
from dataclasses import dataclass
import joblib
import sys

@dataclass
class DataTransformationConfig:
    train_data_path: str                     # Path to the training data CSV file
    test_data_path: str                      # Path to the test data CSV file
    transformed_train_data_path: str         # Path to save the transformed training data
    transformed_test_data_path: str          # Path to save the transformed test data
    transformer_object_path: str             # Path to save the transformer object

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def get_data(self, file_path):
        """
        Load data from a CSV file.
        """
        try:
            data = pd.read_csv(file_path)
            logger.info(f"Data loaded successfully from {file_path}")
            return data
        except Exception as e:
            logger.error(f"Error loading data from {file_path}: {e}")
            raise CustomException(f"Error loading data from {file_path}: {e}", sys)

    def save_data(self, data, file_path):
        """
        Save data to a CSV file.
        """
        try:
            data.to_csv(file_path, index=False)
            logger.info(f"Data saved successfully to {file_path}")
        except Exception as e:
            logger.error(f"Error saving data to {file_path}: {e}")
            raise CustomException(f"Error saving data to {file_path}: {e}", sys)

    def save_transformer(self, transformer):
        """
        Save the transformer object using joblib.
        """
        try:
            joblib.dump(transformer, self.config.transformer_object_path)
            logger.info(f"Transformer saved successfully to {self.config.transformer_object_path}")
        except Exception as e:
            logger.error(f"Error saving transformer object: {e}")
            raise CustomException(f"Error saving transformer object: {e}", sys)

    def load_transformer(self):
        """
        Load the transformer object using joblib.
        """
        try:
            transformer = joblib.load(self.config.transformer_object_path)
            logger.info(f"Transformer loaded successfully from {self.config.transformer_object_path}")
            return transformer
        except Exception as e:
            logger.error(f"Error loading transformer object: {e}")
            raise CustomException(f"Error loading transformer object: {e}", sys)

    def get_transformer(self):
        """
        Create a transformer pipeline for numerical and categorical features.
        """
        try:
            # Define the columns for transformation
            numerical_features = ['reading score', 'writing score']
            categorical_features = ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']

            # Define the pipeline for numerical features
            numerical_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='mean')),  # Replace missing values with the mean
                ('scaler', StandardScaler())                  # Standardize features by removing the mean and scaling to unit variance
            ])

            # Define the pipeline for categorical features
            categorical_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),  # Replace missing values with the most frequent value
                ('onehot', OneHotEncoder(handle_unknown='ignore'))     # Apply one-hot encoding to categorical features
            ])

            # Combine both pipelines into a ColumnTransformer
            transformer = ColumnTransformer(
                transformers=[
                    ('num', numerical_pipeline, numerical_features),  # Apply numerical pipeline to numerical features
                    ('cat', categorical_pipeline, categorical_features)  # Apply categorical pipeline to categorical features
                ]
            )

            logger.info("Transformer pipeline created successfully")
            return transformer
        except Exception as e:
            logger.error(f"Error creating transformer pipeline: {e}")
            raise CustomException(f"Error creating transformer pipeline: {e}", sys)

    def transform_data(self):
        """
        Transform the data using the created transformer pipeline.
        """
        try:
            # Load training and testing data
            train_data = self.get_data(self.config.train_data_path)
            test_data = self.get_data(self.config.test_data_path)

            # Separate features and target for training data
            X_train = train_data.drop(columns=['math score'])
            y_train = train_data['math score']

            # Separate features and target for testing data
            X_test = test_data.drop(columns=['math score'])
            y_test = test_data['math score']

            # Get the transformer
            transformer = self.get_transformer()

            # Fit the transformer to the training data and transform it
            X_train_transformed = transformer.fit_transform(X_train)

            # Transform the test data
            X_test_transformed = transformer.transform(X_test)

            # Combine transformed features and target back into DataFrames
            transformed_train_data = pd.DataFrame(X_train_transformed)
            transformed_train_data['math score'] = y_train.values

            transformed_test_data = pd.DataFrame(X_test_transformed)
            transformed_test_data['math score'] = y_test.values

            # Save the transformed data
            self.save_data(transformed_train_data, self.config.transformed_train_data_path)
            self.save_data(transformed_test_data, self.config.transformed_test_data_path)

            # Save the transformer object for later use
            self.save_transformer(transformer)

            logger.info("Data transformation completed successfully")
        except Exception as e:
            logger.error(f"Error in data transformation process: {e}")
            raise CustomException(f"Error in data transformation process: {e}", sys)

if __name__ == "__main__":
    try:
        # Load configuration from JSON file
        with open('config.json', 'r') as config_file:
            config_data = json.load(config_file)

        # Initialize DataTransformationConfig with loaded data
        config = DataTransformationConfig(
            train_data_path=config_data['train_data_path'],
            test_data_path=config_data['test_data_path'],
            transformed_train_data_path=config_data['transformed_train_data_path'],
            transformed_test_data_path=config_data['transformed_test_data_path'],
            transformer_object_path=config_data['transformer_object_path']
        )

        # Initialize and run data transformation
        data_transformation = DataTransformation(config)
        data_transformation.transform_data()
    except CustomException as e:
        print(e)
    except Exception as e:
        logger.error(f"An error occurred while reading the configuration file: {e}")
        print(f"An error occurred while reading the configuration file: {e}")
