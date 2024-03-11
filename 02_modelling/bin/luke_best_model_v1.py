#!/usr/bin/env python
# coding: utf-8

# # Luke for Entity Pair Classification with classification of type of regulation  
# ## Training with final hyperparameters and merging train and validation dataset

# ## Libraries 

# In[1]:


import pandas as pd
#For spliting training, validation and test groups
from sklearn.model_selection import train_test_split
# Download existing tokenizer for the selected model
from transformers import LukeTokenizer
# For creating the batches for training 
from torch.utils.data import Dataset, DataLoader
# Deep learning framework
import torch
#from transformers import LukeTokenizer, TrainingArguments
import numpy as np
# Download the pretrained model with a classification head on top
from transformers import LukeForEntityPairClassification
# Optimizer for the fine-tunning steps
from transformers import AdamW
# Model construction framework
import pytorch_lightning as pl
from itertools import islice
# Live performance metrics of the model
import wandb
# Meassuring performance of the model F1-score
from torchmetrics.classification import MulticlassF1Score
# Meassuring actual vs predicted groups
from torchmetrics.classification import ConfusionMatrix
# Meassuring Multiclass precision 
from torchmetrics.classification import MulticlassPrecision
# Meassuring Multiclass recall
from torchmetrics.classification import MulticlassRecall
# Meassuring Mathews correlation coeficient
from torchmetrics.classification import MulticlassMatthewsCorrCoef
from sklearn.metrics import classification_report
import torchmetrics
import numpy as np
# Heatmap for confusion matrix 
import seaborn as sn
# Plotting confusion matrix 
import matplotlib.pyplot as plt


# In[2]:


# Setting a seed for random operations reproducibility 
torch.manual_seed(12222)


# In[3]:


# complete data with multi regultor - multi regulated
#data = pd.read_pickle("../../01_preprocessing/results/ECO/curated_master_dataset/ECO_dataset_master_curated_whole_tagging_info_with_span_data_serialized_4_v2.pkl")

# for ascend server running 
data = pd.read_pickle("../data/ECO_dataset_master_curated_whole_tagging_info_with_span_data_serialized_4_v2.pkl")


# In[4]:


data.columns


# ### Extracting from the dataset the 2 columns that we need
# 1. sentence tagged with @TF$ and @Regulated$
# 2. span_regulator_regulated:  [[168, 171], [73, 77]]
# 3. Labels: 'regulator' 'no_relation' 'activator' 'repressor'

# In[5]:


# Creating a smaller dataframe with the 3 columns that we need 
df = data.filter(['SENTENCE','span_regulator_regulated','NORMALIZED_EFFECT'], axis=1)

df.rename(columns = {"SENTENCE":"sentence","span_regulator_regulated":"entity_spans","NORMALIZED_EFFECT":"label"},
               inplace = True)
print(df)


# ### Groups summary 

# In[6]:


print("Sentence example: \n {}".format(df.sentence[0]))
print("Labels: \n {}".format(df.label.unique()))
print("Label distribution: \n {}".format(df.label.value_counts()))


# ### Changing data type of entity spans to numpy array

# In[7]:


#print(data.dtypes)
print(type(df.iloc[:,0][0]))
print(type(df.iloc[:,1][1]))
print(type(df.iloc[:,2][2]))


# In[8]:


df['entity_spans'] = df['entity_spans'].apply(lambda x: np.array(list(x)))


# In[9]:


print(df)


# In[10]:


#print(data.dtypes)
print(type(df.iloc[:,0][0]))
print(type(df.iloc[:,1][1]))
print(type(df.iloc[:,2][2]))


# In[11]:


print(df)


# In[12]:


print(df.shape)
print(df.dtypes)


# ## Creating 2 dictionaries 
# We have 4 labels: activator, repressor, regulator, no_relation
# 1. id2label : maps each label to a unique integer 
# 2. label2id: maps a unique integer to each label 

# In[13]:


id2label = dict()
for idx, label in enumerate(df.label.value_counts().index):
  id2label[idx] = label
print(id2label)


# In[14]:


# For test confusion matrix
class_conf_labels = []
for keys, values in id2label.items():
    class_conf_labels.append(values)

print(class_conf_labels)


# In[15]:


label2id = {v:k for k,v in id2label.items()}
print(label2id)


# # Define the PyTorch dataset and dataloaders 
# In PyTorch you need to define a Dataset class and Dataloaders.
# **Dataset class**: This class stores the samples and their corresponding labels.
# **Dataloaders class**: wraps an iterable around the Dataset to enable easy access to the samples.
# 
# 
# 
# ## 3 Methods must be implemented 
# 1. *init* method: itnitializing the dataset with data.
# 2. *len* method: returns the number of elements in the dataset.
# 3. *getitem()* method: returns a single item from the dataset. 
# 
# ## Each item of the dataset we will use must have 3 columns 
# 1. Sentence 
# 2. Spans of the 2 entities. Eg. [(0, 3), (70, 73)]
# 3. label of the relationship corresponding to the "normalized effect" column.  Eg. activator
# 
# # Tokenization 
# ## The expected input for the model consists of: 
# 1. input_ids
#    - Tensor: Indices of input sequence tokens in the vocabulary. They are numerical representations of tokens that build the input sequence. 
#      - The shape of the tensor is [batch_size, sequence_length].
# 2. entity_ids (particular from Luke)
# 3. attention_mask
#   - Indicates whether a token should be attended to or not.
# 4. entity_attention_mask (particular from Luke)
# 5. entity_position_ids (particular from Luke)

# In[16]:


# We use this tokenizer from transformer library to turn the dataset into the inputs expected by the model 
tokenizer = LukeTokenizer.from_pretrained("studio-ousia/luke-base", task="entity_pair_classification")


class RelationExtractionDataset(Dataset):
    """Relation extraction dataset."""

    def __init__(self, data):
        """
        Args:
            data : Pandas dataframe.
        """
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]

        sentence = item.sentence
        entity_spans = [tuple(x) for x in item.entity_spans]
        
#### Tokenization step
# return_tensors: Specify the type of tensors we want to get back (PyTorch, TensorFlow, or plain NumPy)
# padding: Tensors, the input of the model need to have a uniform shape but the sentences are not the same length with this we add a special token to the sentences that are shorter to ensure tensors are rectangular
# truncation: Sometimes a sequence may be too long for a model to handle. In this case we truncate the sentence to a shorter length. True, truncate a sequence to the maximum length accepted by the model 
        encoding = tokenizer(sentence, entity_spans=entity_spans, padding="max_length", truncation=True, return_tensors="pt")

        for k,v in encoding.items():
          encoding[k] = encoding[k].squeeze()

        encoding["label"] = torch.tensor(label2id[item.label])

        return encoding


# ### Spliting training and test groups (in this model we just train with 80% of data and evaluate with 20%) and making the instances of the RelationExtractionDataset class

# In[17]:


# 80% train and 20% test
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)

print("length train dataset : {}".format(len(train_df)))
print("length test dataset: {}".format(len(test_df)))

# define the dataset
train_dataset = RelationExtractionDataset(data=train_df)
test_dataset = RelationExtractionDataset(data=test_df)


# In[18]:


print("train_dataset: {}".format(train_dataset))
print("test_dataset: {}".format(test_dataset) )


# In[19]:


print("train_dataset.keys: {}".format(train_dataset[0].keys()))
print("input_ids lenght: {}".format(len(train_dataset[0]["input_ids"])))
print("train_dataset.values: {}".format(train_dataset[0].values()))

print("length train dataset : {}".format(len(train_dataset)))
print("length test dataset: {}".format(len(test_dataset)))


# ## Checking the tokenizer vocabulary to understand the input_ids

# In[20]:


print("LukeTokenizer information: \n {}".format(tokenizer))
print("LukeTokenizer vocabulary size: {}".format(tokenizer.vocab_size))


# In[21]:


# Get the first 10 elements of the tokenizer vocabulary 
vocab = tokenizer.get_vocab()
print("vocab keys are the words of the vocabulary: {}".format(list(vocab.keys())[:10]))
print("vocab values are the input_ids: {}".format(list(vocab.values())[:10]))
# Checking if a word is in the vocabulary 
print('Is "ArgP" in vocabulary?: {}'.format("ArgP" in vocab.keys()))
print('Is "this" in vocabulary?: {}'.format("this" in vocab.keys()))
print('input_id of "this": {}'.format(vocab["this"]))


# # Defining the dataloaders 
# 
# While training a model, we typically want to pass samples in “minibatches”, reshuffle the data at every epoch to reduce model overfitting, and use Python’s multiprocessing to speed up data retrieval.
# 
# Dataloaders wrap an iterable around the Dataset to enable easy access to the samples.
# 
# ## Batch size 
# **The batch size is a hyperparameter that defines the number of samples processed before updating the internal model parameters.**
# 
# 1. Batch Gradient Descent. Batch Size = Size of Training Set
# - All training samples are used to create one batch 
# 
# 3. Stochastic Gradient Descent. Batch Size = 1
# - Batch size is the size of one sample 
# 
# 4. Mini-Batch Gradient Descent. 1 < Batch Size < Size of Training Set
# - batch size is more than one sample and less than the size of the training dataset
# - popular are 32, 64 and 128
# - If the dataset does not divide evenly by the batch size, it simply means that the final batch has fewer samples than the other batches.
#   - You can remove some samples from the dataset or change the batch size such that the number of samples in the dataset does divide evenly by the batch size.
# 
# 
# ## Epoch
# 
# **The number of epochs is a hyperparameter that defines the number times that the learning algorithm will work through the entire training dataset.**
# 
# - One epoch means that each sample in the training dataset has had an opportunity to update the internal model parameters.
# - The number of epochs is traditionally large, ranging from 0 to infinite, allowing the learning algorithm to run until the error from the model has been sufficiently minimized.
#   -  Eg. of the number of epochs in the literature and in tutorials set to 10, 100, 500, 1000, and larger
# - Learning curves: create line plots that show epochs along the x-axis as time and the error or skill of the model on the y-axis.
# - Criteria besides a fixed number of epochs, such as a change (or lack of change) in model error over time can be implemented
# 
# ## Example 
# 
# Assume you have a dataset with 200 samples (rows of data) and you choose a batch size of 5 and 1,000 epochs.
# 
# ```
# samples = 200 
# batch size = 5 
# epochs = 1000
# ```
# 
# This means that the dataset will be divided into 40 batches, each with five samples. The model weights will be updated after each batch of five samples.
# 
# ```
# 200 samples / 5 samples per batch  = 40 batches
# ```
# 
# This also means that one epoch will involve 40 batches or 40 updates to the model.
# With 1,000 epochs, the model will be exposed to or pass through the whole dataset 1,000 times. That is a total of 40,000 batches during the entire training process.
# 
# ```
# 1 epoch will invovle 40 updates 
# 1000 epochs will involve 40,000 updates
# ```
# 
# 
# 
# 

# In[22]:


# We will use 10,16,32 and 64
# Shuffle = True indicates that the data will ensure a different order of samples in each epoch
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32)


# In[23]:


print("length train dataloader : {}".format(len(train_dataloader)))
print("length test dataloader: {}".format(len(test_dataloader)))


# In[24]:


## Exploring the dataloaders
# Iterate over the dataloader
for batch in test_dataloader:
    print(batch.keys())
    # Access the data samples in the batch
    input_ids = batch['input_ids']
    attention_mask = batch['attention_mask']
    labels = batch['label']
    
    # Print or inspect the data samples
    print("Input IDs:", input_ids)
    print("Attention Mask:", attention_mask)
    print("Label:", label)
    
    # Additional exploration or analysis of the data
    # ...

    # Break the loop after inspecting one batch if needed
    break


# ## Testing the model and why we import as AutomModelForSequenceClassification

# In[25]:


model = LukeForEntityPairClassification.from_pretrained("studio-ousia/luke-base", num_labels=len(label2id))
print(model)


# # Defining a PyTorch LightningModule 
# PyTorch Lightning is the deep learning framework
# This method will organize our pytorch code and let us develop different complexity layer for the training 
# **It has over 20 hooks to keep flexibility**
# 
# ## `__init__`
# - Model architecture goes in here which in this case is luke-base 
# 
# ## `forward` 
# - Defines the prediction /inference actions
# 
# ## `configure_optimizers`
# - optimizers go in here
# - `self.parameters()` will contain parameters for encoder and decoder
# 
# ## `training_step`
# - The training logic goes in here 
# - Use `self.log` to send any metric to TensorBoard or your preffered logger
#   - TensorBoard provides the visualization and tooling needed for machine learning experimentation
# - in `self.log(on_epoch= True)` add param on_epoch= True to calculate epoch-level metrics
# - `return loss` if if you want to use them latter 
# 
# ## `validation_step`
# - Here goes the validation logic 
# - `self.log` calling it will automatically accumulate and log at the end of the epoch
# 
# ## Remove any `.cuda` or device calls
# - lightningModule is device agonistic 
# 

# In[26]:


class LUKE(pl.LightningModule):

    def __init__(self, num_classes):
        super().__init__()
        self.model = LukeForEntityPairClassification.from_pretrained("studio-ousia/luke-base", num_labels=len(label2id))
        # Multiclass classification loss
        # Cross Entropy loss combines softmax activation function and negative log likelihood loss into a single operation
        # Softmax will ensure that the sum of the probabilities of each class is equal to 1 by doing 3 things:
        # Calculates the exponential of each logit 
        # Sum up the exponential values of each logit for getting the denominator of the division 
        # Divides each exponential by the sum of the exponential values to ensure the probabilites of each class sum up to 1
        self.criterion = torch.nn.CrossEntropyLoss()
        # Accuracy in every step 
        self.train_acc = torchmetrics.Accuracy(task = 'multiclass', num_classes = num_classes)
        self.test_acc = torchmetrics.Accuracy(task = 'multiclass', num_classes = num_classes)
        # Multi class F1 score in every step 
        self.train_f1 = MulticlassF1Score(num_classes = num_classes, average='macro')
        self.test_f1 = MulticlassF1Score(num_classes = num_classes, average='macro')
        # Multi class precision
        self.train_precision = MulticlassPrecision(num_classes = num_classes, average='macro')
        self.test_precision = MulticlassPrecision(num_classes = num_classes, average ='macro')
        # Multiclass recall
        self.train_recall = MulticlassRecall(num_classes = num_classes, average = 'macro')
        self.test_recall = MulticlassRecall(num_classes = num_classes, average = 'macro')
        # Multiclass matthews correlation coeficient
        self.train_matthews_corr_coef = MulticlassMatthewsCorrCoef(num_classes = num_classes)
        self.test_matthews_corr_coef = MulticlassMatthewsCorrCoef(num_classes = num_classes)
        # Class names for classification report and confusion matrix
        self.class_conf_labels = class_conf_labels
        # Classification report
#        self.test_classification_report = classification_report(target_names =self.class_conf_labels,output_dict=True)
        # Confusion matrix 
        self.confmat = ConfusionMatrix(task="multiclass", num_classes=4)
        self.class_names = label2id 

        self.test_preds_list = []
        self.test_targets_list = []

    def forward(self, input_ids, entity_ids, entity_position_ids, attention_mask, entity_attention_mask):     
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, entity_ids=entity_ids, entity_attention_mask=entity_attention_mask, entity_position_ids=entity_position_ids)
        return outputs.logits

### training_step
# The training logic goes in here 
# Use self.log to send any metric to TensorBoard or your preffered logger for visualization and tooling needed for machine learning experimentation
# in self.log(on_epoch= True) add param on_epoch= True to calculate epoch-level metrics
# return loss  or a dictionary of predictions if you want to use them latter in training
    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        train_labels = batch['label']
        # particular parameters for LUKE model
        entity_ids = batch['entity_ids']
        entity_attention_mask = batch['entity_attention_mask']
        entity_position_ids = batch['entity_position_ids']

        train_logits = self.forward(input_ids=input_ids, attention_mask=attention_mask, entity_ids=entity_ids, entity_attention_mask=entity_attention_mask, entity_position_ids=entity_position_ids)
        # Quantifies the dissimilarity between the predicted probabilities and the target distribution
        train_loss = self.criterion(train_logits, train_labels)
        # argmax operation is performed along the last dimension of the tensor which is typically the class dimension
        # it returns the indices of the maximum values along the specified dimension in this case, the index of the class with the highest logit value
        train_preds = train_logits.argmax(-1)
        # Log metrics
        self.log("training_loss", train_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("training_accuracy", self.train_acc(train_preds, train_labels), on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_f1", self.train_f1(train_preds,train_labels), on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_precision",self.train_precision(train_preds,train_labels), on_step = True, on_epoch = True, prog_bar = True)
        self.log("train_recall",self.train_recall(train_preds,train_labels), on_step =True, on_epoch = True, prog_bar = True)
        self.log("train_matthews_corr_coef",self.train_matthews_corr_coef(train_preds, train_labels),on_step =True, on_epoch =True, prog_bar =True)

        return train_loss

    def test_step(self, batch, batch_idx):
        #self.test_preds_list = []
        #self.test_targets_list = []
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        test_labels = batch['label']
        # particular parameters for LUKE model
        entity_ids = batch['entity_ids']
        entity_attention_mask = batch['entity_attention_mask']
        entity_position_ids = batch['entity_position_ids']

        test_logits = self.forward(input_ids=input_ids, attention_mask=attention_mask, entity_ids=entity_ids, entity_attention_mask=entity_attention_mask, entity_position_ids=entity_position_ids)
        # Getting the predictions
        test_preds = test_logits.argmax(-1)
        # Calculating metrics
        test_loss = self.criterion(test_logits, test_labels)
        self.test_acc(test_preds, test_labels)
        self.test_f1(test_preds,test_labels)
        self.test_precision(test_preds, test_labels)
        self.test_recall(test_preds,test_labels)
        self.test_matthews_corr_coef(test_preds,test_labels)
        # Log metrics
        self.log("test_loss", test_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("test_accuracy", self.test_acc, on_step=True, on_epoch=True, prog_bar=True)   
        self.log("test_f1", self.test_f1, on_step=True, on_epoch=True, prog_bar=True)
        self.log("test_precision", self.test_precision, on_step=True, on_epoch=True, prog_bar = True)
        self.log("test_recall", self.test_recall, on_step =True, on_epoch = True, prog_bar = True)
        self.log("test_matthews_corr_coef",self.test_matthews_corr_coef, on_step =True, on_epoch =True, prog_bar =True)

        self.test_preds_list.extend(test_preds.cpu().detach().numpy())
        self.test_targets_list.extend(test_labels.cpu().detach().numpy())
        return test_loss 

    def on_test_epoch_end(self):
        wandb.log({"confusion_matrix-test":wandb.plot.confusion_matrix(probs=None, y_true= self.test_targets_list, preds = self.test_preds_list, class_names=self.class_conf_labels)})
        classification_report_dict = classification_report(self.test_preds_list, self.test_targets_list,target_names =self.class_conf_labels,output_dict=True)
        classification_report_df = pd.DataFrame(classification_report_dict).transpose()
        print(classification_report_df)
        classification_report_df.to_csv("../results/luke_best_model_classification_report.csv",sep=",",index=True)
        classification_table = wandb.Table(dataframe=classification_report_df)
        wandb.log({"classification_report":classification_table})


    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=0.00001)
        return optimizer

    def train_dataloader(self):
        return train_dataloader

    def test_dataloader(self):
        return test_dataloader


# The initial loss should be around -ln(1/number of classes) = -ln(1/4) = 1.3862943611
# ## Meassure of loss 
# 
# Cross-entropy is a measure of the difference between two probability distributions for a given random variable or set of events.

# # Train the model 

# In[27]:


wandb.login()


# In[28]:


from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping

wandb_logger = WandbLogger(name='luke_best_model', project='deep_for_bio_nlp')
model = LUKE(num_classes=len(label2id))
#trainer = Trainer(logger=wandb_logger, accelerator= 'cpu',devices= 1, max_steps = 2)

# For Ascend server running and correcting the step logging error due to batch size  If the logging interval is larger than the number of training batches, then logs will not be printed for every training epoch. 
trainer = Trainer(logger=wandb_logger, accelerator= 'auto',devices= 'auto', max_steps = 415, log_every_n_steps =1)
#trainer = Trainer(logger=wandb_logger, accelerator= 'auto',devices= 'auto', callbacks=[EarlyStopping(monitor='validation_loss')])
# Enable the following Trainer arguments to run on Apple silicon gpus (MPS devices).
# The MPSAccelerator only supports 1 device at a time. Currently there are no machines with multiple MPS-capable GPUs.
# params to run on mac but bugging 
#trainer = Trainer(logger=wandb_logger, accelerator= 'mps',devices= 1, callbacks=[EarlyStopping(monitor='validation_loss')])

# original trainer params to run local 
#trainer = Trainer(logger=wandb_logger, accelerator= 'cpu',devices= 1, callbacks=[EarlyStopping(monitor='validation_loss')])

trainer.fit(model)


# In[29]:


trainer.test()

