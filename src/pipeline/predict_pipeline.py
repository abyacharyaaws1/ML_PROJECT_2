import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

class PredictPipeline:
    def __init__(self, transformer_path, model_path):
        self.transformer = joblib.load(transformer_path)
        self.model = joblib.load(model_path)

    def predict(self, input_data: pd.DataFrame):
        """
        Transform input data and make predictions using the trained model.
        :param input_data: DataFrame containing the input features.
        :return: Model predictions.
        """
        transformed_data = self.transformer.transform(input_data)
        predictions = self.model.predict(transformed_data)
        return predictions

def create_pipeline():
    config_path = 'config.json'
    import json
    with open(config_path, 'r') as file:
        config = json.load(file)
    
    pipeline = PredictPipeline(
        transformer_path=config['transformer_object_path'],
        model_path=config['model_output_path']
    )
    return pipeline
