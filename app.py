from flask import Flask, render_template, request
import pandas as pd
from src.pipeline.predict_pipeline import create_pipeline

app = Flask(__name__)

# Initialize the prediction pipeline
pipeline = create_pipeline()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/home')
def home():
    return render_template('home.html', results="")

@app.route('/predict_datapoint', methods=['POST'])
def predict_datapoint():
    try:
        # Extract form data
        gender = request.form.get('gender')
        ethnicity = request.form.get('ethnicity')
        parental_level_of_education = request.form.get('parental_level_of_education')
        lunch = request.form.get('lunch')
        test_preparation_course = request.form.get('test_preparation_course')
        reading_score = int(request.form.get('reading_score'))
        writing_score = int(request.form.get('writing_score'))

        # Create a DataFrame with the input data
        input_data = pd.DataFrame({
            'gender': [gender],
            'race/ethnicity': [ethnicity],
            'parental level of education': [parental_level_of_education],
            'lunch': [lunch],
            'test preparation course': [test_preparation_course],
            'reading score': [reading_score],
            'writing score': [writing_score]
        })

        # Predict using the pipeline
        prediction = pipeline.predict(input_data)[0]

        return render_template('home.html', results=f"Predicted Math Score: {prediction}")

    except Exception as e:
        return render_template('home.html', results=f"An error occurred: {e}")

if __name__ == '__main__':
    app.run(debug=True)
