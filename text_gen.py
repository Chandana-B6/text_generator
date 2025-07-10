from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load pre-trained model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Set the model to evaluation mode (faster inference)
model.eval()

# Take user input
topic = input("Enter a topic for paragraph generation: ")

# Encode the input
input_ids = tokenizer.encode(topic, return_tensors='pt')

# Generate output using GPT2
output = model.generate(
    input_ids,
    max_length=150,          # length of paragraph
    num_return_sequences=1,  # how many outputs
    no_repeat_ngram_size=2,  # avoid repeating phrases
    do_sample=True,          # add randomness
    top_k=50,                # limits to top 50 words
    top_p=0.95,              # nucleus sampling
    temperature=0.8          # controls creativity
)

# Decode the output
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

# Show result
print("\nGenerated Paragraph:\n")
print(generated_text)
