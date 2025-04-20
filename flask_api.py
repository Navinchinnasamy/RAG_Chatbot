from flask import Flask, request, jsonify
from modules.chatbot import Chatbot
from modules.config_loader import load_config

app = Flask(__name__)

# Initialize chatbot
config = load_config()
chatbot = Chatbot(config)

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    if not data or "query" not in data:
        return jsonify({"error": "Missing 'query' in request body"}), 400
    
    query = data["query"]
    session_id = data.get("session_id", "default")  # Optional: For tracking user sessions

    # Process query
    try:
        response = chatbot.process_query(query)
        return jsonify({"response": response, "session_id": session_id})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/reset", methods=["POST"])
def reset():
    data = request.get_json()
    session_id = data.get("session_id", "default")
    
    # Reset context
    try:
        chatbot.reset_context()
        return jsonify({"message": "Conversation reset", "session_id": session_id})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)