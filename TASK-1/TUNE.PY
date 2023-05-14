from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

# Load the GPT-2 tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Encode the preprocessed chat data
encoded_data = tokenizer.encode(' '.join(filtered_tokens), add_special_tokens=True, return_tensors='pt')

# Fine-tune the model on the chat data
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
model.zero_grad()

loss = model(encoded_data, labels=encoded_data)[0]
loss.backward()
optimizer.step()
