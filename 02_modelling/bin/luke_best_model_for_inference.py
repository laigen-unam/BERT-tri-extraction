#!/usr/bin/env python
# coding: utf-8

# # Error analysis of the best model 
# - model: LUKE
# - batch_size: 32
# - learning_rate: 0.00001
# - epoch: 12

# In[8]:


# Deep learning framework
import torch
import pandas as pd
import numpy as np
# Download existing tokenizer for the selected model
from transformers import LukeTokenizer
# For creating the batches for training 
from torch.utils.data import Dataset, DataLoader
#For spliting training, validation and test groups
from sklearn.model_selection import train_test_split
from pytorch_lightning import Trainer


# In[9]:


# Model construction framework
import pytorch_lightning as pl
# Download the pretrained model with a classification head on top
from transformers import LukeForEntityPairClassification
# Meassuring performance of the model F1-score
from torchmetrics.classification import MulticlassF1Score
# Meassuring actual vs predicted groups
from torchmetrics.classification import ConfusionMatrix
import torchmetrics
# Live performance metrics of the model
import wandb
from pytorch_lightning.loggers import WandbLogger


# ## Seeding everything as in the training of the model for comparisson porpouses 

# In[10]:


# Setting a seed for random operations reproducibility 
torch.manual_seed(12222)
## Getting the dataset splits 


# ## Getting the dataset splits 

# In[38]:


# complete data with multi regultor - multi regulated
#data = pd.read_pickle("../../01_preprocessing/results/ECO/curated_master_dataset/ECO_dataset_master_curated_whole_tagging_info_with_span_data_serialized_4_v2.pkl")
#data.columns


# In[11]:


# complete data with multi regultor - multi regulated
# Local Alfredo
#data = pd.read_pickle("../../../Data-sets/STM_data_set_articles_for_LUKE.pkl")
# server 
data = pd.read_pickle("/export/storage/users/avarela/deep_learning_models/Data-sets/STM_data_set_dl_articles_4600_preprocessed_for_luke.pkl")
data.columns


# In[12]:


print(data)


# In[13]:


# Creating a smaller dataframe with the 3 columns that we need 
df = data.filter(['SENTENCE','span_regulator_regulated','NORMALIZED_EFFECT','sentence_tagged','REGULATOR','REGULATED'], axis=1)

df.rename(columns = {"SENTENCE":"sentence","sentence_tagged":"sentence_tagged","REGULATOR":"regulator","REGULATED":"regulated","span_regulator_regulated":"entity_spans","NORMALIZED_EFFECT":"label"},
               inplace = True)
print(df)


# In[6]:


print("Sentence example: \n {}".format(df.sentence[0]))
print("Labels: \n {}".format(df.label.unique()))
print("Label distribution: \n {}".format(df.label.value_counts()))


# ### Changing data type of entity spans to numpy array

# In[67]:


#print(data.dtypes)
print(type(df.iloc[:,0][0]))
print(type(df.iloc[:,1][1]))
print(type(df.iloc[:,2][2]))


# In[14]:


print(data.dtypes)
print(type(data.iloc[:,0][0]))
print(type(data.iloc[:,1][1]))
print(data.iloc[:,1][1])
print(type(data.iloc[:,2][2]))


# In[8]:


df['entity_spans'] = df['entity_spans'].apply(lambda x: np.array(list(x)))


# In[15]:


data['entity_spans'] = data['entity_spans'].apply(lambda x: np.array(list(x)))


# In[28]:


print(df)


# In[16]:


print(data)


# In[17]:


#print(data.dtypes)
print(type(df.iloc[:,0][0]))
print(type(df.iloc[:,1][1]))
print(type(df.iloc[:,2][2]))


# In[18]:


print(type(data.iloc[:,0][0]))
print(type(data.iloc[:,1][1]))
print(type(data.iloc[:,2][2]))


# ## Creating 2 dictionaries 
# We have 4 labels: activator, repressor, regulator, no_relation
# 1. id2label : maps each label to a unique integer 
# 2. label2id: maps a unique integer to each label 

# In[11]:


id2label = dict()
for idx, label in enumerate(df.label.value_counts().index):
  id2label[idx] = label
print(id2label)


# In[19]:


id2label = dict()
label = ["activator", "no_relation", "repressor", "regulator"]
for idx, label in enumerate(label):
    id2label[idx] = label
print(id2label)


# In[20]:


# For test confusion matrix
class_conf_labels = []
for keys, values in id2label.items():
    class_conf_labels.append(values)

print(class_conf_labels)


# In[21]:


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

# In[22]:


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
        #sentence_tagged = item.sentence_tagged
        regulator = item.regulator
        regulated = item.regulated
        entity_spans = [tuple(x) for x in item.entity_spans]
        
#### Tokenization step
# return_tensors: Specify the type of tensors we want to get back (PyTorch, TensorFlow, or plain NumPy)
# padding: Tensors, the input of the model need to have a uniform shape but the sentences are not the same length with this we add a special token to the sentences that are shorter to ensure tensors are rectangular
# truncation: Sometimes a sequence may be too long for a model to handle. In this case we truncate the sentence to a shorter length. True, truncate a sequence to the maximum length accepted by the model 
        encoding = tokenizer(sentence, entity_spans=entity_spans, padding="max_length", truncation=True, return_tensors="pt")

        for k,v in encoding.items():
          encoding[k] = encoding[k].squeeze()

        #encoding["label"] = torch.tensor(label2id[item.label])

       # Return both the tokenized data and the original sentence
        return {
            "tokenized_data": encoding,
            # store the original sentence
            "sentence": sentence,
            #"sentence_tagged":sentence_tagged,
            "regulator":regulator,
            "regulated":regulated
            
        } 


# ### Making the instances of the RelationExtractionDataset class

# In[23]:


# 80% train and 20% test
#train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)

print("length test dataset: {}".format(len(data)))

# define the dataset
infer_dataset = RelationExtractionDataset(data=data)


# In[24]:


print("test_dataset: {}".format(infer_dataset) )


# In[25]:


# We will use 10,16,32 and 64
# Shuffle = True indicates that the data will ensure a different order of samples in each epoch
infer_dataloader = DataLoader(infer_dataset, batch_size=32)


# In[26]:


print("length infer dataloader: {}".format(len(infer_dataloader)))


# In[27]:


# Access the real sentences
for idx in range(len(infer_dataset)):
    sample = infer_dataset[idx]
    sentence = sample["sentence"]
#    sentence_tagged = sample["sentence_tagged"]
    sentence_regulator = sample["regulator"]
    sentence_regulated = sample["regulated"]

    # Now you have access to the original sentence
    print("Original Sentence:", sentence)
#    print("Original Sentence tagged:", sentence_tagged)
    print("Original regulator:", sentence_regulator)
    print("Original regulated:", sentence_regulated)


# In[28]:


for batch in infer_dataloader:
    input_ids = batch['tokenized_data']['input_ids']
    sentences = batch['sentence'] 
    #sentences_tagged = batch['sentence_tagged']
    sentences_regulators = batch['regulator']
    sentences_regulateds = batch['regulated'] 

    # Print the data in the current batch
    print("Input IDs:", input_ids)
    print("Sentences:", sentences)
    #print("Sentences tagged",sentences_tagged)
    print("Regulators",sentences_regulators)
    print("Regulateds",sentences_regulateds)
    
    # You can process and print more details about the data as needed

    # Break the loop if you only want to inspect the first batch
    break


# # Loading trained model

# In[29]:


# Adding get_test_predictions
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
        self.val_acc = torchmetrics.Accuracy(task = 'multiclass', num_classes = num_classes)
        self.test_acc = torchmetrics.Accuracy(task = 'multiclass', num_classes = num_classes)
        # Multi class F1 score in every step 
        self.train_f1 = MulticlassF1Score(num_classes = num_classes, average='macro')
        self.val_f1 =  MulticlassF1Score(num_classes = num_classes, average='macro')
        self.test_f1 = MulticlassF1Score(num_classes = num_classes, average='macro')

        self.confmat = ConfusionMatrix(task="multiclass", num_classes=4)
        self.class_names = label2id 
        self.class_conf_labels = class_conf_labels

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
        ####

        return train_loss

### validation_step
# Here goes the validation logic 
# self.log calling it will automatically accumulate and log at the end of the epoch
    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        val_labels = batch['label']
        # particular parameters for LUKE model
        entity_ids = batch['entity_ids']
        entity_attention_mask = batch['entity_attention_mask']
        entity_position_ids = batch['entity_position_ids']

        val_logits = self.forward(input_ids=input_ids, attention_mask=attention_mask, entity_ids=entity_ids, entity_attention_mask=entity_attention_mask, entity_position_ids=entity_position_ids)
        # Getting the predictions 
        val_preds = val_logits.argmax(-1)

        # Calculating metrics 
        val_loss = self.criterion(val_logits, val_labels)
        self.val_acc(val_preds, val_labels)
        self.val_f1(val_preds, val_labels)

        # Log metrics
        self.log("validation_loss", val_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("validation_accuracy", self.val_acc, on_step=True, on_epoch=True, prog_bar=True)
        self.log("validation_f1", self.val_f1, on_step=True, on_epoch=True, prog_bar=True)
        
        self.confmat.update(val_preds, val_labels)

        return val_loss

    def on_validation_epoch_end(self):
            confmat = self.confmat.compute()
            class_names = self.class_names
            df_cm = pd.DataFrame(confmat.cpu().numpy() , index = [i for i in class_names], columns = [i for i in class_names])
            print('Num of val samples: {}. Check this aligns with the numbers from the dataloader'.format(df_cm.sum(axis=1).sum()))
            #log to wandb
            f, ax = plt.subplots(figsize = (15,10)) 
            sn.heatmap(df_cm, annot=True, ax=ax)
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.title("Confusion Matrix - Validation")
            wandb.log({"plot": wandb.Image(f) })
            # For not stacking the results after each epoch 
            self.confmat.reset() 

    def test_step(self, batch, batch_idx):
        #self.test_preds_list = []
        #self.test_targets_list = []
        input_ids = batch['tokenized_data']['input_ids']
        attention_mask = batch['tokenized_data']['attention_mask']
        #test_labels = batch['tokenized_data']['label']
        # particular parameters for LUKE model
        entity_ids = batch['tokenized_data']['entity_ids']
        entity_attention_mask = batch['tokenized_data']['entity_attention_mask']
        entity_position_ids = batch['tokenized_data']['entity_position_ids']
        sentences = batch['sentence']
        #sentences_tagged = batch['sentence_tagged']
        sentences_regulators = batch['regulator']
        sentences_regulateds = batch['regulated']

        test_logits = self.forward(input_ids=input_ids, attention_mask=attention_mask, entity_ids=entity_ids, entity_attention_mask=entity_attention_mask, entity_position_ids=entity_position_ids)
        # Getting the predictions
        test_preds = test_logits.argmax(-1)
        # Calculating metrics
        #test_loss = self.criterion(test_logits, test_labels)
        #self.test_acc(test_preds, test_labels)
        #self.test_f1(test_preds,test_labels)
        # Log metrics
        #self.log("test_loss", test_loss, on_step=True, on_epoch=True, prog_bar=True)
        #self.log("test_accuracy", self.test_acc, on_step=True, on_epoch=True, prog_bar=True)   
        #self.log("test_f1", self.test_f1, on_step=True, on_epoch=True, prog_bar=True)

        self.test_preds_list.extend(test_preds.cpu().detach().numpy())
        #self.test_targets_list.extend(test_labels.cpu().detach().numpy())
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        #    "test_labels":test_labels,
            "test_preds":test_preds,
            "sentences":sentences,
        #    "sentences_tagged":sentences_tagged,
            "sentences_regulators":sentences_regulators,
            "sentences_regulateds":sentences_regulateds
        }

    def get_test_predictions(self, test_loader):
        self.test_preds_analysis_list = []
#        self.test_targets_analysis_list = []
        self.test_sentences_analysis_list = []
#        self.test_sentences_tagged_analysis_list = []
        self.test_sentences_regulators_analysis_list = []
        self.test_sentences_regulateds_analysis_list = []



        self.eval()

        for batch in test_loader:
            results = self.test_step(batch, 0)  # 0 is a placeholder for batch index
            self.test_preds_analysis_list.extend(results["test_preds"].cpu().detach().numpy())
            #self.test_targets_analysis_list.extend(results["test_labels"].cpu().detach().numpy())

           # Include the original sentences and entity positions in the results
            self.test_sentences_analysis_list.extend(results["sentences"])
            #self.test_sentences_tagged_analysis_list.extend(results["sentences_tagged"])
            self.test_sentences_regulators_analysis_list.extend(results["sentences_regulators"])
            self.test_sentences_regulateds_analysis_list.extend(results["sentences_regulateds"])


        #return self.test_sentences_analysis_list,self.test_sentences_tagged_analysis_list,self.test_sentences_regulators_analysis_list,self.test_sentences_regulateds_analysis_list, self.test_targets_analysis_list, self.test_preds_analysis_list
        return self.test_sentences_analysis_list,self.test_sentences_regulators_analysis_list,self.test_sentences_regulateds_analysis_list, self.test_preds_analysis_list


    #def on_test_epoch_end(self):
        #wandb.log({"confusion_matrix-test":wandb.plot.confusion_matrix(probs=None, y_true= self.test_targets_list, preds = self.test_preds_list, class_names=self.class_conf_labels)})

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=0.00001)
        return optimizer

    def train_dataloader(self):
        return train_dataloader

    def val_dataloader(self):
        return valid_dataloader

    def test_dataloader(self):
        return test_dataloader


# In[30]:

# Local Alfredo 
#model = LUKE.load_from_checkpoint(
#    checkpoint_path="deep_for_bio_nlp/7gzyvoli/checkpoints/epoch=10-step=415.ckpt", num_classes = 4
#)

# Server pakal
model = LUKE.load_from_checkpoint(
    checkpoint_path="/export/storage/users/avarela/deep_learning_models/models/deep_for_bio_nlp/7gzyvoli/checkpoints/epoch=10-step=415.ckpt", num_classes = 4
)


# In[28]:


# Extract real sentences and predictions from the results
#input_sentences, input_sentences_tagged,input_regulators,input_regulateds, true_labels, predicted_labels = model.get_test_predictions(infer_dataloader)






# In[32]:


# Extract real sentences and predictions from the results without tagged sentences
input_sentences,input_regulators,input_regulateds, predicted_labels = model.get_test_predictions(infer_dataloader)


# In[33]:


print(input_sentences)
#print(input_sentences_tagged)
print(input_regulators)
print(input_regulateds)
#print(true_labels)
print(predicted_labels)


# In[34]:


print(len(input_sentences))
#print(len(input_sentences_tagged))
print(len(input_regulators))
print(len(input_regulateds))
#print(len(true_labels))
print(len(predicted_labels))


# In[35]:


id2label


# In[36]:


#true_labels_words = [id2label[label] for label in true_labels]
predicted_labels_words = [id2label[predicted] for predicted in predicted_labels]
#print(true_labels)
#print(true_labels_words)
print(predicted_labels)
print(predicted_labels_words)


# In[37]:


import csv 
# Local Alfredo
#with open("../../02_modelling/inference/luke_stm_predictions_all_dataset.tsv","w") as output:
# Server pakal
with open("/export/storage/users/avarela/deep_learning_models/inference/luke_stm_4600_predictions_all_dataset.tsv","w") as output:

    writer = csv.writer(output, delimiter='\t')
#    writer.writerow(["sentence","sentence_tagged","regulator","regulated", "true_label", "predicted_label"])
    writer.writerow(["regulator","regulated","predicted_label"])

    for regulator, regulated, predicted_label in zip(input_regulators,input_regulateds, predicted_labels_words):
        writer.writerow([regulator, regulated, predicted_label])


# In[38]:

# Local Alfredo
#with open("../../02_modelling/inference/luke_stm_predictions_with_sentence_all_dataset.tsv","w") as output:
# Server pakal
with open("/export/storage/users/avarela/deep_learning_models/inference/luke_stm_4600_predictions_with_sentence_all_dataset.tsv","w") as output:
    writer = csv.writer(output, delimiter='\t')
#    writer.writerow(["sentence","sentence_tagged","regulator","regulated", "true_label", "predicted_label"])
    writer.writerow(["sentence","regulator","regulated","predicted_label"])

    for sentence, regulator, regulated, predicted_label in zip(input_sentences,input_regulators,input_regulateds, predicted_labels_words):
        writer.writerow([sentence,regulator, regulated, predicted_label])


# In[42]:

# Local Alfredo 
#all_predictions_tsv = pd.read_csv("../../02_modelling/inference/luke_stm_predictions_all_dataset.tsv", delimiter= "\t")
# Sever pakal
all_predictions_tsv = pd.read_csv("/export/storage/users/avarela/deep_learning_models/inference/luke_stm_4600_predictions_all_dataset.tsv")
filtered_df = all_predictions_tsv[all_predictions_tsv['predicted_label'] != 'no_relation']

# Save the filtered data to a new tsv file
# Local Alfredo
#filtered_df.to_csv('../../02_modelling/inference/luke_stm_predictions_264_dataset_without_no_relation.tsv', sep ="\t", index=False)
# Server pakal
filtered_df.to_csv('/export/storage/users/avarela/deep_learning_models/inference/luke_stm_predictions_4600_dataset_without_no_relation.tsv', sep ="\t", index=False)

