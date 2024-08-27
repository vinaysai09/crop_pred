from flask import Flask,render_template,request,redirect,url_for
import pickle
import numpy as np

app = Flask("_name__")

rf = pickle.load(open('crop.pkl','rb'))

@app.route('/')
def hello_world():
    return render_template("index.html")

@app.route('/predict',methods = ['POST','GET'])

def predict():
    sg = request.form['n']
    htn = (request.form['p'])
    hemo = (request.form['k'])
    dm = (request.form['temp'])
    al = (request.form['hum'])
    appet = (request.form['ph'])
    rc = (request.form['rain'])
    values = np.array([[sg, htn, hemo, dm, al, appet, rc]])

    # int_features=[int(x) for x in request.form.values()]
    # values=[np.array(int_features)]
    # values = np.array([[99,15,27,27.41,56.6,6.08,127.92]])

    predict=rf.predict(values)
    print(predict)

    return render_template('result.html',predict=predict)


if __name__ == '__main__':
    app.run(debug=True)