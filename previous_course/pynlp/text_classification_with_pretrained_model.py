import re
import emoji
import csv
import os, sys
from datetime import datetime
import logging
from sklearn.metrics import classification_report
import torch
from sklearn import preprocessing
from keras.preprocessing.sequence import pad_sequences
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler, TensorDataset
from transformers import BertForSequenceClassification, AdamW, BertConfig
import numpy as np
import datetime
from _datetime import datetime as dt
from transformers import (
    WEIGHTS_NAME,
    AdamW,
    BertConfig,
    BertForMaskedLM,
    BertTokenizer,
    CamembertConfig,
    CamembertForMaskedLM,
    CamembertTokenizer,
    DistilBertConfig,
    DistilBertForMaskedLM,
    DistilBertTokenizer,
    GPT2Config,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    OpenAIGPTConfig,
    OpenAIGPTLMHeadModel,
    OpenAIGPTTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    RobertaConfig,
    RobertaForMaskedLM,
    RobertaTokenizer,
    get_linear_schedule_with_warmup,
)

# Logger stuff
logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
logger = logging.getLogger(__name__)


def load_unlabeled_data(tokenizer, test_file_path):

    '''
    Function to load unlabeled data. The input format is one tweet per line:
    tweet_id , tweet text , labels
    :param tokenizer: BERT tokenizer, output of the training code
    :return: the list of
        test_input_ids, test_attention_masks, test_tweet_ids
        which stand for the list of tokenized tweet texts, the list of attention masks, and the list of input tweet ids respectively
        note that the list test_tweet_ids will be used for prediction output
    '''
    # List of all tweets text
    test_tweets = []
    test_tweet_ids = []
    test_labels = []
    # Test Set
    with open(test_file_path) as input_file:
        reader = csv.reader(input_file, delimiter=",")
        # For each tweet
        # for line in csv.reader(input_file, delimiter="\t"):
        for line in reader:
        #    line = line.split(",")
            if line[0] != 'id':
                full_line = line[1]
                full_line = re.sub(r'#([^ ]*)', r'\1', full_line)
                full_line = re.sub(r'https.*[^ ]', 'URL', full_line)
                full_line = re.sub(r'http.*[^ ]', 'URL', full_line)
                full_line = re.sub(r'@([^ ]*)', '@USER', full_line)
                full_line = emoji.demojize(full_line)
                full_line = re.sub(r'(:.*?:)', r' \1 ', full_line)
                full_line = re.sub(' +', ' ', full_line)

                # Save tweet's text and label
                test_tweets.append(full_line)
                test_tweet_ids.append(line[0])
                test_labels.append(line[2])

    # List of all tokenized tweets
    test_input_ids = []

    # For every tweet in the test set
    for sent in test_tweets:
        # `encode` will:
        #   (1) Tokenize the sentence.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        encoded_sent = tokenizer.encode(
            sent,  # Sentence to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            max_length=100,

            # This function also supports truncation and conversion
            # to pytorch tensors, but we need to do padding, so we
            # can't use these features :( .
            # max_length = 128,          # Truncate all sentences.
            # return_tensors = 'pt',     # Return pytorch tensors.
        )

        # Add the encoded tweet to the list.
        test_input_ids.append(encoded_sent)

    # # Pad our input tokens with value 0.
    # # "post" indicates that we want to pad and truncate at the end of the sequence,
    # # as opposed to the beginning.
    test_input_ids = pad_sequences(test_input_ids, maxlen=100, dtype="long",
                                    value=tokenizer.pad_token_id, truncating="pre", padding="pre")

    # Create attention masks
    # The attention mask simply makes it explicit which tokens are actual words versus which are padding
    test_attention_masks = []

    # For each tweet in the test set
    for sent in test_input_ids:
        # Create the attention mask.
        #   - If a token ID is 0, then it's padding, set the mask to 0.
        #   - If a token ID is > 0, then it's a real token, set the mask to 1.
        att_mask = [int(token_id > 0) for token_id in sent]

        # Store the attention mask for this sentence.
        test_attention_masks.append(att_mask)

    # Return the list of encoded tweets, the list of labels and the list of attention masks
    return test_input_ids, test_attention_masks, test_tweet_ids, test_labels



def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


# ======================================================================================================================
# Part of the code comes from https://mccormickml.com/2019/07/22/BERT-fine-tuning/
# ======================================================================================================================
# ---------------------------- Main ----------------------------

# Directory where the pretrained model can be found
model_dir = sys.argv[1]
test_file_path = sys.argv[2]

output_file = '/path2folder/output.csv'
print(test_file_path)
print(output_file)
# Returns a datetime object containing the local date and time
dateTimeObj = str(dt.now()).replace(" ", "_")

# Log stuff: print logger on file
# Make dir for model serializations
os.makedirs(os.path.dirname('path2folder'), exist_ok=True)


logging.basicConfig(filename='/path2folder/prediction_log' + model_dir.split("/")[-1].split("_")[-1] + '.log', level=logging.DEBUG)

# Log stuff: print logger also on stderr
consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
logger.addHandler(consoleHandler)

# -----------------------------
# Load Pre-trained BERT model
# -----------------------------
config_class, model_class, tokenizer_class = (BertConfig, BertForSequenceClassification, BertTokenizer)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load a trained model and vocabulary pre-trained for specific language
logger.info("Loading model from: '" + model_dir + "', it may take a while...")

# Load pre-trained Tokenizer from directory, change this to load a tokenizer from ber package
tokenizer = tokenizer_class.from_pretrained(model_dir)

# Load Bert for classification 'container'
model = BertForSequenceClassification.from_pretrained(
    model_dir, # Use pre-trained model from its directory, change this to use a pre-trained model from bert
    num_labels = 2, # The number of output labels--2 for binary classification.
                    # You can increase this for multi-class tasks.
    output_attentions = False, # Whether the model returns attentions weights.
    output_hidden_states = False, # Whether the model returns all hidden-states.
)

# Set the model to work on CPU if no GPU is present
model.to(device)
logger.info("Bert for classification model has been loaded!")

# --------------------------------------------------------------------
# -------------------------- Load test data --------------------------
# --------------------------------------------------------------------

# The loading eval data return:
# - input_ids:         the list of all tweets already tokenized and ready for bert (with [CLS] and [SEP)
# - labels:            the list of labels, the i-th index corresponds to the i-th position in input_ids
# - attention_masks:   a list of [0,1] for every input_id that represent which token is a padding token and which is not
# input_ids, labels, attention_masks = load_eval_data(language, tokenizer)

# Load Offenseval 2018, Train,Test already divided into Train/Test set

prediction_inputs, prediction_masks, tweet_ids, tweet_labels = load_unlabeled_data(tokenizer, test_file_path)


#print(tweet_labels)
# Tweets
prediction_inputs = torch.tensor(prediction_inputs)

# Attention masks
prediction_masks = torch.tensor(prediction_masks)


label_encoder = preprocessing.LabelEncoder()
targets = label_encoder.fit_transform(tweet_ids)
# targets: array([0, 1, 2, 3])

prediction_ids = torch.as_tensor(targets)
# targets: tensor([0, 1, 2, 3])

# Set the batch size.
batch_size = 32

# Create the DataLoader.
prediction_data = TensorDataset(prediction_inputs, prediction_masks, prediction_ids)
prediction_sampler = SequentialSampler(prediction_data)
prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)

print('Predicting labels for {:,} test sentences...'.format(len(prediction_inputs)))

# Put model in evaluation mode
model.eval()

# Tracking variables
predictions = []

# Predict
for batch in prediction_dataloader:
    # Add batch to GPU
    batch = tuple(t.to(device) for t in batch)

    # Unpack the inputs from our dataloader
    b_input_ids, b_input_mask, b_tweet_ids = batch

    # Telling the model not to compute or store gradients, saving memory and
    # speeding up prediction
    with torch.no_grad():
        # Forward pass, calculate logit predictions
        outputs = model(b_input_ids, token_type_ids=None,
                        attention_mask=b_input_mask)

    logits = outputs[0]

    # Move logits and labels to CPU
    logits = logits.detach().cpu().numpy()

    flat_logits = np.argmax(logits, axis=1).flatten()
    # Get tweet ids for prediction output
    ids = label_encoder.inverse_transform(b_tweet_ids.cpu().numpy())
    # Store predictions and true labels
    predictions.extend(list(zip(ids, flat_logits)))

# Print the list of prediction & store for evaluation
with open(output_file, 'w') as out_file:
    # store predictions in list for eval
    pred_ = []
    # Get each tweet id
    for tweet_id_prediction in predictions:
        # Print the prediction todo: debug to remove
        #print(str(tweet_id_prediction[0]) + "\t" + str(tweet_id_prediction[1]))
        pred_.append(str(tweet_id_prediction[1]).replace('0', 'NOT').replace('1','ABU'))
        # Append the tweet id along with predicted label on file
        label = 'ABU'
        if str(tweet_id_prediction[1]) == '0':
            label = 'NOT'
#            pred_.append(label)
#        pred_.append(label)
        out_file.write(str(tweet_id_prediction[0]) + "," + label + '\n')

logger.info(len(pred_))
logger.info(len(tweet_labels))
logger.info(classification_report(tweet_labels,pred_, digits=4))



print('    DONE.')
