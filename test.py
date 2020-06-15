import torch
from transformers import BertTokenizer, BertModel
# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging
#logging.basicConfig(level=logging.INFO)

# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
print(tokenizer.convert_tokens_to_ids(['UNK']))
# print(list(tokenizer.vocab.keys())[10000:20000])

# # Define a new example sentence with multiple meanings of the word "bank"
# text = "After stealing money from the bank vault, the bank robber was seen " \
#        "fishing on the Mississippi river bank."
# text_2= "[CLS] Here is the sentence I want embeddings for. [SEP]"
# # Add the special tokens.
# marked_text = "[CLS] " + text + " [SEP]"

# # Split the sentence into tokens.
# tokenized_text = tokenizer.tokenize(marked_text)

# print(tokenized_text)
# # Map the token strings to their vocabulary indeces.
# indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

# # Display the words with their indeces.
# # for tup in zip(tokenized_text, indexed_tokens):
# #     print('{:<12} {:>6,}'.format(tup[0], tup[1]))
# # Mark each of the 22 tokens as belonging to sentence "1".
# segments_ids = [1] * len(tokenized_text)
# # Convert inputs to PyTorch tensors

# # Load pre-trained model (weights)
# model = BertModel.from_pretrained('bert-base-uncased',
#                                   output_hidden_states = True, # Whether the model returns all hidden-states.
#                                   )

# # Put the model in "evaluation" mode, meaning feed-forward operation.
# model.eval()
# model.cuda()
# # Run the text through BERT, and collect all of the hidden states produced
# # from all 12 layers. 
# with torch.no_grad():
#     outputs = model(torch.tensor([indexed_tokens]).cuda(),torch.tensor([segments_ids]).cuda())
#     # https://huggingface.co/transformers/model_doc/bert.html#bertmodel
#     hidden_states = outputs[2][12]
# print(hidden_states)
# # Remove dimension 1, the "batches".
# # Concatenate the tensors for all layers. We use `stack` here to
# # create a new dimension in the tensor.
# token_embeddings = torch.stack(hidden_states, dim=0)



# print(token_embeddings.size())
