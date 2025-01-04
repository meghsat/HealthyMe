import torch
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from datasets import load_dataset, Dataset
from peft import PeftModel
import pandas as pd
import gc
from flask_cors import CORS

gc.collect()
torch.cuda.empty_cache()
if torch.cuda.is_available():
    torch.device('cuda:0')
    print("CUDA is available. Using GPU.")
else:
    raise RuntimeError("No GPU found. A GPU is needed for quantization.")

# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.bfloat16,
# )
device = "cuda" if torch.cuda.is_available() else "cpu"
# base_model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

base_model_id = "LiteAI/Hare-1.1B-base"
tokenizer = AutoTokenizer.from_pretrained(base_model_id)
model = AutoModelForCausalLM.from_pretrained(base_model_id)
model.to(device)
# base_model = AutoModelForCausalLM.from_pretrained(
#     base_model_id,  # Llama 2 7B, same as before
#     quantization_config=bnb_config,  # Same quantization config as before
#     device_map="auto",
#     trust_remote_code=True,
#     token="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"
# )
# tokenizer = AutoTokenizer.from_pretrained(base_model_id, add_bos_token=True, trust_remote_code=True, token="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")
# ft_model = PeftModel.from_pretrained(base_model, "checkpoints/no_news/checkpoint-300")
# pipeline = pipeline(
#     "text-generation",
#     model=ft_model,
#     tokenizer = tokenizer,
#     token="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"
# )
# terminators = [
#     pipeline.tokenizer.eos_token_id,
#     pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
# ]
app = Flask(_name_)
CORS(app)

@app.route('/generate-text', methods=['POST'])
def generate_text():
    try:
        data = request.json
        prompt = data.get('prompt', '')
        tokens = tokenizer(prompt, add_special_tokens=True, return_tensors='pt').to(device)
        output = model.generate(**tokens,max_new_tokens=128)
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        # data = request.json
        # prompt_text = data.get('prompt', '')

        # messages = [
        #     {"role": "system", "content": "You are a helpful AI assistant who is an expert analyst that will attend the earnings call with Skyworks Solutions Inc. to ask questions about the company's financial health and goals based on the earnings report and prepared remarks."},
        #     {"role": "user", "content": f"Generate exactly 5 questions an analyst might ask related to the given input:{prompt_text} based on the knowledge you gained from earnings report and prepared text of the previous quarters."},
        # ]
        # prompt = pipeline.tokenizer.apply_chat_template(
        #     messages, 
        #     tokenize=False, 
        #     add_generation_prompt=True
        # )

        # Generate text using the pipeline
        # outputs = pipeline(
        #     prompt,
        #     max_new_tokens=6000,
        #     eos_token_id=terminators
        # )
        # llm_questions = outputs[0]["generated_text"][len(prompt):]
        # # start = llm_questions.find("{")
        # # end = llm_questions.find("}") + 1
        # generated_text = llm_questions

        return jsonify({'generated_text': generated_text})
    except Exception as e:
        return jsonify({'error': str(e)})

if _name_ == '_main_':
    app.run(debug=True)


        #     def generate():
        #     messages = [
        #     {"user": "Alice", "message": "Hello, anyone here?"},
        #     {"user": "Bob", "message": "Hi Alice, I'm here!"},
        #     {"user": "Charlie", "message": "Hey folks, what's up?"}
        #     ]
        #     for message in messages:
        #         yield f"data: {json.dumps(message)}\n\n"
        #         time.sleep(1)  # Simulate delay between messages
        # return Response(stream_with_context(generate()), content_type='text/event-stream')