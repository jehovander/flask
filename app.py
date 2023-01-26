from flask import Flask , render_template , request , jsonify

from chat import get_response
 import nltk
 nltk.download('punkt')
app = Flask ( __name__ )



@app.get('/')
def index_get():
    return render_template('base.html')


@app.post('/predict')
def predict():
    text = request.get_json().get('message')
    #TODO:chequea el texto si es valido 
    response=get_response(text)
    message= {'answer': response}
    return jsonify(message)





if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0",port=5000)
