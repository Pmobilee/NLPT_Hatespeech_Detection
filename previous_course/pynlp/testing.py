import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from pytorch_pretrained_bert import BertTokenizer, BertConfig
from pytorch_pretrained_bert import BertAdam, BertForSequenceClassification
from tqdm import tqdm, trange
import pandas as pd
import io
import numpy as np
import matplotlib.pyplot as plt
# % matplotlib inline
torch.cuda.empty_cache()
# specify GPU device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
torch.cuda.get_device_name(0)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

model = torch.load('E:/VU/subjectivity_mining/eigenBert/model/model.pt')


# test = pd.read_csv("E:/VU/subjectivity_mining/eigenBert/twitterdata/filled_data/ManualTag_Misogyny.csv", sep=';', encoding = "ISO-8859-1")
test = pd.read_csv("E:/VU/subjectivity_mining/eigenBert/own_dataset.csv", sep='\t', encoding = "ISO-8859-1")

# load test data
# sentences = ["[CLS] " + query + " [SEP]" for query in test["text"]]
# labels = test["label"]

sentences = ["[CLS] " + query + " [SEP]" for query in test["text"]]
labels = test["label"]

# tokenize test data
tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]
MAX_LEN = 128
# Pad our input tokens
input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                          maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
# Use the BERT tokenizer to convert the tokens to their index numbers in the BERT vocabulary
input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
# Create attention masks
attention_masks = []
# Create a mask of 1s for each token followed by 0s for padding
for seq in input_ids:
  
  seq_mask = [float(i>0) for i in seq]
  attention_masks.append(seq_mask) 


# create test tensors
prediction_inputs = torch.tensor(input_ids)
prediction_masks = torch.tensor(attention_masks)
prediction_labels = torch.tensor(labels.to_numpy(dtype=np.float64), dtype=torch.long)
batch_size = 50  
prediction_data = TensorDataset(prediction_inputs, prediction_masks, prediction_labels)
prediction_sampler = SequentialSampler(prediction_data)
prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)

## Prediction on test set
# Put model in evaluation mode
model.eval()
# Tracking variables 
predictions , true_labels = [], []
# Predict 
for batch in prediction_dataloader:
  # Add batch to GPU
  batch = tuple(t.to(device) for t in batch)
  # Unpack the inputs from our dataloader
  b_input_ids, b_input_mask, b_labels = batch
  # Telling the model not to compute or store gradients, saving memory and speeding up prediction
  with torch.no_grad():
    # Forward pass, calculate logit predictions
    logits = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
  # Move logits and labels to CPU
  logits = logits.detach().cpu().numpy()
  label_ids = b_labels.to('cpu').numpy()  
  # Store predictions and true labels
  predictions.append(logits)
  true_labels.append(label_ids)
  
# Import and evaluate each test batch using Matthew's correlation coefficient
from sklearn.metrics import matthews_corrcoef
matthews_set = []
for i in range(len(true_labels)):
  matthews = matthews_corrcoef(true_labels[i],
                 np.argmax(predictions[i], axis=1).flatten())
  matthews_set.append(matthews)
  
# Flatten the predictions and true values for aggregate Matthew's evaluation on the whole dataset
flat_predictions = [item for sublist in predictions for item in sublist]
flat_predictions = np.argmax(flat_predictions, axis=1).flatten()
flat_true_labels = [item for sublist in true_labels for item in sublist]

from sklearn.metrics import classification_report
print(classification_report(flat_true_labels, flat_predictions))

predictions_list = []
text_list = []


# a = 0
# for i in range(len(flat_predictions)):
#   if flat_predictions[i] == 0:
#     print('jup')
#     predictions_list.append('Sexism')
#     text_list.append(test['Text'][i])

# print(predictions_list)
# print(text_list)

# a = pd.DataFrame({'text' : text_list, 'prediction' : predictions_list})
# a.to_csv('sexism_on_offense.csv', sep=';')

#print('Classification accuracy using BERT Fine Tuning: {0:0.2%}'.form)