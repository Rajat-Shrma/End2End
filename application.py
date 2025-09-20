from flask import Flask,render_template,request,url_for

from src.pipeline.predict_pipeline import PredictPipeline,CustomDataset

applicaton=Flask(__name__)

app=applicaton

@app.route('/')
def welcome():
    return render_template('index.html')

@app.route("/form", methods=['GET','POST'])
def form():
    if request.method=='GET':
        return render_template('form.html')

    else:
        gender=request.form.get('gender')
        race_ethnicity=request.form.get('race_ethnicity')
        parental_level_of_education=request.form.get('parental_level_of_education')
        lunch=request.form.get('lunch')
        test_preparation_course=request.form.get('test_preparation_course')
        reading_score=float(request.form.get('reading_score'))
        writing_score=float(request.form.get('writing_score'))

        data=CustomDataset(gender,race_ethnicity,parental_level_of_education,lunch,test_preparation_course,reading_score,writing_score)

        features=data.get_dataframe()

        prediction_pipeline=PredictPipeline()

        pred=prediction_pipeline.predict(features)

        return render_template('form.html',prediction=pred[0])
    

if __name__=="__main__":
    app.run(host='0.0.0.0')



