from flask import Flask, request, jsonify
from model import predict
app = Flask(__name__)


@app.route("/api/v1/chat", methods=['POST'])
def chat():
    try:
        r = request.get_json(silent=True)
        text = r.get("text")
    except:
        msg = {'response': "could not get json"}
        return jsonify(msg)

    prediction = predict(text)
    return jsonify(prediction)

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=8888)

