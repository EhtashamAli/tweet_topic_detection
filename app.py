import numpy as np
from flask import Flask, request, jsonify
import pickle
import json
from flask_cors import CORS


app = Flask(__name__)
CORS(app)

linear_model = pickle.load(open('./Model/model_linear_model.pkl','rb'))
naive_bayes = pickle.load(open('./Model/model_naive_bayes.pkl','rb'))
svm = pickle.load(open('./Model/model_svm.pkl','rb'))
count_vect = pickle.load(open('./Model/count_vect.pkl','rb'))
encoder = pickle.load(open('./Model/encoder.pkl','rb'))

@app.route('/api',methods=['POST'])
def predict():
    data = request.get_json(force=True)
    # print(data['tweet'])
    # tweet = "This is a tweet about Science and Technology, wow!"
    # print("Predicting tweet: {}".format(tweet))
    prediction1 = linear_model.predict(count_vect.transform([data['tweet']]))
    linear_model_result = encoder.inverse_transform(prediction1)
    prediction2 = naive_bayes.predict(count_vect.transform([data['tweet']]))
    naive_bayes_result = encoder.inverse_transform(prediction2)
    prediction3 = svm.predict(count_vect.transform([data['tweet']]))
    svm_result = encoder.inverse_transform(prediction3)

    results = { "svm": svm_result[0] , "naive_bayes": naive_bayes_result[0] , "linear_model": linear_model_result[0]}
    # results = { "svm": svm_result.tolist() , "naive_bayes": naive_bayes_result.tolist() , "linear_model": linear_model_result.tolist()}
    y = json.dumps(results)
    # class MyClass:
    #     svm = svm_result
    #     naive_bayes = naive_bayes_result
    #     linear_model = linear_model_result
    # print(prediction1)
    # print("**********")
    # print(prediction2)
    # print("**********")
    # print(prediction3)
    # print("**********")
    return jsonify(y)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=3000, debug=True, threaded=True)