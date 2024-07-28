from transformers import AutoTokenizer
 
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
 
text = input("Enter text: ")
tokenized_text = tokenizer.tokenize(text)
print(tokenized_text)
 
token_ids = tokenizer.encode(text)
print(token_ids)
 
tokens = tokenizer.convert_ids_to_tokens(token_ids)
print(tokens)