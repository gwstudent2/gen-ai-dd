from transformers import AutoTokenizer, AutoModel
import torch
 
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
model = AutoModel.from_pretrained("bert-base-cased")
 
term1 = str(input("Enter first term: "))
term2 = input("Enter second term: ")
term3 = input("Enter third term: ")
 
# get the embedding vector for each term
term1_token_id = tokenizer.convert_tokens_to_ids(term1)
term1_embedding = model.embeddings.word_embeddings(torch.tensor([term1_token_id]))
term2_token_id = tokenizer.convert_tokens_to_ids(term2)
term2_embedding = model.embeddings.word_embeddings(torch.tensor([term2_token_id]))
term3_token_id = tokenizer.convert_tokens_to_ids(term3)
term3_embedding = model.embeddings.word_embeddings(torch.tensor([term3_token_id]))
 
print('Dimensions for ', term1, term1_embedding.shape)
print('First 10 dimensions for ', term1, ' : ', term1_embedding [0][:10])
print('Dimensions for ', term2, term2_embedding.shape)
print('First 10 dimensions for ', term2, ' : ', term2_embedding [0][:10])
print('Dimensions for ', term3, term3_embedding.shape)
print('First 10 dimensions for ', term3, ' : ', term3_embedding [0][:10])
 

cos = torch.nn.CosineSimilarity(dim=1)
similarity1to2 = cos(term1_embedding, term2_embedding)
print('Similarity measure between ', term1, ' and ', term2, ' is ', similarity1to2[0])
similarity2to3 = cos(term2_embedding, term3_embedding)
print('Similarity measure between ', term2, ' and ', term3, ' is ', similarity2to3[0])
similarity1to3 = cos(term1_embedding, term3_embedding)
print('Similarity measure between ', term1, ' and ', term3, ' is ', similarity1to3[0])
 