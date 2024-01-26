from flask import Flask, render_template, request
from src.exception import CustomException
from src.pipelines.predict_pipeline import CustomData, PredictionPipeline

application = Flask(__name__)
app = application
# ['Cement', 'Blast_Furnace_Slag', 'Fly_Ash', 'Water', 'Superplasticizer', 'Coarse_Aggregate', 'Fine_Aggregate', 'Age','Concrete_strength']
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_data', methods=['GET', 'POST'])
def predict_data():
    if request.method == 'GET':
        return render_template('head1.html')
    else:
        data = CustomData(
            Cement=int(request.form.get('Cement')),
            Blast_Furnace_Slag=int(request.form.get('Blast_Furnace_Slag')),
            Fly_Ash=int(request.form.get('Fly_Ash')),
            Water=int(request.form.get('Water')),
            Superplasticizer=int(request.form.get('Superplasticizer')),
            Coarse_Aggregate=int(request.form.get('Coarse_Aggregate')),
            Fine_Aggregate=int(request.form.get('Fine_Aggregate')),
            Age=int(request.form.get('Age'))
        )

        pred_df = data.get_data_to_df()
        print(pred_df)

        processor = PredictionPipeline()
        print("Before Prediction")
        results = processor.predict(pred_df)
        print("after Prediction")
        return render_template('head1.html', results=results[0])

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
