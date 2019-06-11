import numpy as np
from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

linear_model = pickle.load(open('./Model/model_linear_model.pkl','rb'))
naive_bayes = pickle.load(open('./Model/model_naive_bayes.pkl','rb'))
svm = pickle.load(open('./Model/model_svm.pkl','rb'))


@app.route('/api',methods=['POST'])
def predict():
    data = request.get_json(force=True)
    print(data['tweet'])
    prediction1 = linear_model.predict([[np.array(data['tweet'])]])
    prediction2 = naive_bayes.predict([[np.array(data['tweet'])]])
    prediction3 = svm.predict([[np.array(data['tweet'])]])
    print(prediction1)
    print("**********")
    print(prediction2)
    print("**********")
    print(prediction3)
    print("**********")
    return jsonify(prediction1)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True, threaded=True)