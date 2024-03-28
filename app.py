from flask import Flask, request, jsonify
import boto3
import json

app = Flask(__name__)

bedrock = boto3.client(service_name="bedrock-runtime")

model_id = "meta.llama2-70b-chat-v1"

@app.route('/generate_poem', methods=['POST'])
def generate_poem():
    user_query = request.json.get('query')
    payload = {
        "prompt": "[INST]" + user_query + "[/INST]",
        "max_gen_len": 512,
        "temperature": 0.5,
        "top_p": 0.9
    }
    body = json.dumps(payload)
    response = bedrock.invoke_model(
        body=body,
        modelId=model_id,
        accept="application/json",
        contentType="application/json"
    )
    response_body = json.loads(response.get("body").read())
    generated_poem = response_body['generation']
    return jsonify({'generated_poem': generated_poem})

if __name__ == '__main__':
    app.run(debug=True)
