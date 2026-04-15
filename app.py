from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

#load model and scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        features = [
            float(request.form["MonthlyIncome"]),
            float(request.form["OverTime"]),
            float(request.form["Age"]),
            float(request.form["DailyRate"]),
            float(request.form["TotalWorkingYears"]),
            float(request.form["HourlyRate"]),
            float(request.form["MonthlyRate"]),
            float(request.form["YearsAtCompany"]),
            float(request.form["DistanceFromHome"])
        ]
        
        final_input = np.array(features).reshape(1, -1)
        final_input = scaler.transform(final_input)
        
        prediction = model.predict(final_input)[0]
        probability = model.predict_proba(final_input)[0][1]
        
        if prediction == 1:
            result = f"⚠️ Employee is likely to leave ({probability*100:.2f}%)"
        else:
            result = f"✅ Employee is likely to stay ({(1-probability)*100:.2f}%)"
            
        return render_template("index.html", prediction_text=result)
    
    except Exception as e:
        return render_template("index.html", predicition_text=f"Error: {e}")
 
if __name__=="__main__":
    app.run()
