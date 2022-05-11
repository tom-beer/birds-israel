from flask import Flask, render_template, request, jsonify
from bird_app import BirdApp

app = Flask(__name__)
bird_app = BirdApp()


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        if file is not None:
            return jsonify(bird_app.predict(file))


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def upload_file():
    uploaded_file = request.files['file']
    # if uploaded_file.filename != '':
    #     uploaded_file.save(uploaded_file.filename)

    return jsonify(bird_app.predict(uploaded_file))


if __name__ == '__main__':
    app.run(port=5001)
