import numpy as np
from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

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
    result = encoder.inverse_transform(prediction1)
    # prediction2 = naive_bayes.predict([[np.array(data['tweet'])]])
    # prediction3 = svm.predict([[np.array(data['tweet'])]])
    # print(prediction1)
    # print("**********")
    # print(prediction2)
    # print("**********")
    # print(prediction3)
    # print("**********")
    return jsonify(result.tolist())

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True, threaded=True)