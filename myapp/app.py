from flask import Flask, render_template, request
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

app = Flask(__name__)
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/get-response', methods=['POST'])
def get_response():
    user_input = request.form['user_input']
    input_ids = tokenizer.encode(user_input, return_tensors='pt')
    output = model.generate(input_ids, max_length=100, num_beams=5, no_repeat_ngram_size=2, early_stopping=True)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return {'response': response}

if __name__ == '__main__':
    app.run(debug=True)
