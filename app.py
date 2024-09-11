from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        try:
            data = CustomData(
                total_sqft=request.form.get('total_sqft'),
                bath=request.form.get('bath'),
                BHK=request.form.get('BHK'),
                location=request.form.get('Location')
            )
            pred_df = data.get_data_as_data_frame()
            print(pred_df)
            predict_pipeline = PredictPipeline()
            results = predict_pipeline.predict(pred_df)
            return render_template('home.html', results=results[0])
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            return "Internal Server Error", 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
