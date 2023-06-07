from flask import Flask, request, jsonify
from flask_cors import CORS
import chatbot

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["https://chatbot-cat-ia.vercel.app", "http://localhost:3000"]}})

@app.route('/chat', methods=['GET'])
def default():
    return jsonify({'response': 'Bienvenido, mi nombre es Angelina. En que puedo ayudarlo?'})
    
@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    message = data['message']
    response = chatbot.get_response(chatbot.model, message)
    return jsonify({'response': response})

if __name__ == '__main__':
    chatbot.load_intents()
    chatbot.load_model_files()
    model = chatbot.model
    app.run()
