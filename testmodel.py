import joblib
model = joblib.load('linear_regression_model.joblib')
print(model.predict([[650]])[0])
