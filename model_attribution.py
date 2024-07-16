import json
import torch
import codecs
import operator
import re
import time
import os
import math
from random import shuffle
import torch.nn as nn
import torch.nn.functional as F
from torch import nn as nn
from sklearn.datasets import load_digits
import numpy as np
from sklearn.metrics import classification_report, mean_absolute_error, confusion_matrix
from torch.utils.data import DataLoader, Dataset
from memory_supcontrast import SupConLoss
import random
import argparse
import pandas as pd
from transformers import XLMRobertaTokenizer,XLMRobertaConfig,XLMRobertaModel,AutoTokenizer, AutoModel, AutoModelWithLMHead,XLMRobertaForMaskedLM
from transformers import RobertaTokenizer,RobertaConfig,RobertaModel, get_linear_schedule_with_warmup
from transformers import BertTokenizer,BertConfig,BertModel
from typing import Union, Iterable
from sentence_transformers import SentenceTransformer, util
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
import seaborn as sns



num_regex = re.compile('^[+-]?[0-9]+\.?[0-9]*$')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    print ('GPU will be used')
else:
    print ("CPU will be used")
 
def xlm_r_train_valid_test(data, tokenizer, max_length, domain_label_map):
    """
    obtain pretrained xlm_r representations
    """
    # X_train, X_valid, X_domain_train, X_domain_valid, y_train, y_valid = train_test_split(data['synthetic misinformation'], data['generation_approach'], data['generated_by'], test_size=0.2, random_state=42)
    # X_train, X_domain_train, y_train = data['synthetic misinformation'], data['generation_approach'], data['generated_by']
    
    # Split data into train and combined validation-test sets
    x_train, x_valid_test, x_domain_train, x_domain_valid, y_train, y_valid_test = train_test_split(data['synthetic misinformation'], data['generation_approach'], data['generated_by'], test_size=0.2, random_state=42)

    # Split combined validation-test set into validation and test sets
    x_valid, x_test, y_valid, y_test = train_test_split(x_valid_test, y_valid_test, test_size=0.5, random_state=42)


    # Use LabelEncoder to encode the labels
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_valid_encoded = label_encoder.fit_transform(y_valid)
    y_test_encoded = label_encoder.fit_transform(y_test)


    # Encode training data
    embedded_data_train = []
    for text, domain, label in zip(x_train, x_domain_train, y_train_encoded):
        encoded = torch.LongTensor(tokenizer.encode(text, max_length=max_length, truncation=True, pad_to_max_length=True)).to(device)
        domain_tensor = torch.tensor([float(domain_label_map[domain])]).long().to(device)
        label_tensor = torch.tensor([label]).long().to(device)
        embedded_data_train.append(torch.cat((domain_tensor, label_tensor, encoded), dim=0).unsqueeze(0))

    # Encode validation data
    embedded_data_valid = []
    for text, label in zip(x_valid, y_valid_encoded):
        encoded = torch.LongTensor(tokenizer.encode(text, max_length=max_length, truncation=True, pad_to_max_length=True)).to(device)
        label_tensor = torch.tensor([label]).long().to(device)
        embedded_data_valid.append(torch.cat((label_tensor, encoded), dim=0).unsqueeze(0))

    # Encode test data
    embedded_data_test = []
    for text, label in zip(x_test, y_test_encoded):
        encoded = torch.LongTensor(tokenizer.encode(text, max_length=max_length, truncation=True, pad_to_max_length=True)).to(device)
        label_tensor = torch.tensor([label]).long().to(device)
        embedded_data_test.append(torch.cat((label_tensor, encoded), dim=0).unsqueeze(0))

    embedded_data_train = torch.cat(embedded_data_train)
    embedded_data_valid = torch.cat(embedded_data_valid)
    embedded_data_test = torch.cat(embedded_data_test)

    return embedded_data_train, embedded_data_valid, embedded_data_test

def xlm_r_test(data, tokenizer, max_length):
    """
    obtain pretrained xlm_r representations
    """
    
    X_test, y_test = data['synthetic misinformation'], data['generated_by']

    # Use LabelEncoder to encode the labels
    label_encoder = LabelEncoder()
    y_test_encoded = label_encoder.fit_transform(y_test)

    # Encode testing data
    embedded_data_test = []
    for text, label in zip(X_test, y_test_encoded):
        encoded = torch.LongTensor(tokenizer.encode(text, max_length=max_length, truncation=True, pad_to_max_length=True)).to(device)
        label_tensor = torch.tensor([label]).long().to(device)
        embedded_data_test.append(torch.cat((label_tensor, encoded), dim=0).unsqueeze(0))

    embedded_data_test = torch.cat(embedded_data_test)

    return embedded_data_test
#######################################################################################################
def load_data(llms, dataset, generation_approach, human=True):
    data = pd.DataFrame()

    for llm in llms:
        for d in dataset:
            for g in generation_approach:
                df = pd.read_csv(path + 'filtered_llm/' + llm + '/' + d + '/' + 'synthetic-' + llm + '_' + d + '_' + g + '_filtered' + '.csv')
                # Add a 'generated_by' column to each dataset
                df['generated_by'] = llm
                # Add a 'generation_approach' column to each dataset
                df['generation_approach'] = g
                # Concatenate the two datasets
                data = pd.concat([data, df], ignore_index=True)
            
    if human:
        for d in dataset:
            df = pd.read_csv(path + 'filtered_human/' + d + '/' + d + '_human_filtered' + '.csv')
            # Add a 'generated_by' column to each dataset
            df['generated_by'] = 'human'
            # Add a 'generation_approach' column to each dataset
            df['generation_approach'] = 'human'
            # Concatenate the two datasets
            data = pd.concat([data, df], ignore_index=True)

    # Remove rows where 'synthetic misinformation' has NaN values
    data = data.dropna(subset=['synthetic misinformation'])
    # Shuffle the combined dataset
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)

    print('--------------------------------------------------------------')
    print('LLMs: {}'.format(llms))
    print('Dataset: {}'.format(dataset))
    print('Generation Approach: {}'.format(generation_approach))
    print('Using Human data: {}'.format(human))
    print('--------------------------------------------------------------')
    print(data['generated_by'].value_counts())
    print('--------------------------------------------------------------')
    print(data['generation_approach'].value_counts())
    print('--------------------------------------------------------------')

    return data

#######################################################################################################
# Function to compute semantic similarity
def compute_similarity(text1, text2):
    embeddings1 = sim_model.encode(text1, convert_to_tensor=True)
    embeddings2 = sim_model.encode(text2, convert_to_tensor=True)
    return util.pytorch_cos_sim(embeddings1, embeddings2).item()

# Conditional application based on 'generated_by' column
def conditional_similarity(row):
    if row['generated_by'] == 'gpt-3.5-turbo':
        return compute_similarity(row['news_text'], row['synthetic misinformation'])
    else:
        return compute_similarity(row['human'], row['synthetic misinformation'])

def filter_data(df):
    # Apply the function to each row
    df['similarity'] = df.apply(conditional_similarity, axis=1)

    grouped_means = df.groupby(['generated_by', 'generation_approach'])['similarity'].mean().reset_index()
    print(grouped_means)
    print('--------------------------------------------------------------')

    # Merge the mean similarity scores back into the original DataFrame
    df_with_means = pd.merge(df, grouped_means, on=['generated_by', 'generation_approach'], suffixes=('', '_mean'))

    # Filter rows where the similarity is greater than or equal to the group mean
    filtered_df = df_with_means[df_with_means['similarity'] >= df_with_means['similarity_mean']]

    # Now, filtered_df contains only the rows where the similarity score 
    # is equal to or higher than the mean for its 'generated_by' and 'generation_approach' group.
    print(filtered_df['generated_by'].value_counts())
    print('--------------------------------------------------------------')
    print(filtered_df['generation_approach'].value_counts())
    print('--------------------------------------------------------------')

    return filtered_df

#######################################################################################################

def train(train_data, total_valid, model, fa_module, classifier, gradient_accumulate_step, epoch_num):
    """
    function for training the classifier
    """

    global global_step
    global start
    global results_table
    global class_centroids
    global momentum_embedding
    global memory_bank
    # Train the model
    model.train()
    train_loss = 0
    contrast_loss = 0
    intra_class_loss = 0
    train_acc = 0
    domain_acc = 0
    optimizer.zero_grad()

    best_val_accuracy = 0
    best_model_state = None
    
    # print(train_data.size())
    data = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    n_batches = len(data) 
    for i, pairs in enumerate(data):
        # print(pairs)
        # print (pairs.size())
        text_a = pairs[ :, 2:params.max_length+2].to(device)
        index = pairs[ :, 1:2].to(device)
        domain_label = pairs[ :, :1].to(device)
        text_a=torch.squeeze(text_a)
        index = torch.squeeze(index)
        domain_label = torch.squeeze(domain_label)
        # optim.zero_grad()
        #optimizer.zero_grad()
        text_a,  index = text_a.to(device),  index.to(device)
        #last_hidden=model.extract_features(text)
        last_hidden_a=model(text_a)[0]
        # print(last_hidden_a.size())
        last_hidden_cls = last_hidden_a[:,0,:]#Classifier representation
        last_hidden_avg = torch.mean(last_hidden_a, dim=1)#Classifier representation
        last_hidden_cls = fa_module(last_hidden_cls)
        #last_hidden_avg = fa_module(last_hidden_avg)
      
        #print(index)
        #hinge_loss = 0
           #print(cls_centroids[cls_id]['centroid'].shape)
        #first_token_hidden = last_hidden[:,0,:]
        #contrast_input = torch.stack([last_hidden_cls, last_hidden_cls],dim=1)
        contrast_input = last_hidden_cls
        output = classifier(last_hidden_cls)

       
        #supcon_loss = contrast_criterion(last_hidden_cls,index)
        ce_loss = criterion(output, index)
        divisor = 24
       
        batch_prediction = output.argmax(1)
        inter_loss = 0 
        mmd_loss = 0 
        intra_loss = 0
        batch_cluster_loss = 0

        #moco_loss = MocoLoss(batch_mean_feature_list, class_centroids, moco_distance)     
        divisor = 24
        #if torch.unique(index, return_counts=True)[1].min()<=2:
        #  supcon_loss = 0
        if memory_bank == None:
          #supcon_loss = 0
          supcon_loss = contrast_criterion(contrast_input,index)
        elif memory_bank != None:
          #supcon_loss = moco_loss(batch_example, memory_bank, params.nclass)
          memory_label = memory_bank[:,:1].squeeze()
          memory_feature = memory_bank[:, 1:]
          supcon_loss = contrast_criterion(contrast_input,  index, memory_feature, memory_label)
     
        batch_example = torch.cat([index.unsqueeze(1), last_hidden_cls], dim = 1 )
        enqueue_and_dequeue(batch_example, params.memory_bank_size) 
        #contrast_loss = moco_loss(batch_example, memory_bank, params.nclass)  
        #if torch.unique(index, return_counts=True)[1].min() <=2:
        #  contrast_loss = 0
        #else:
        #  contrast_loss = contrast_criterion(contrast_input, index)
        
        composite_loss =  params.lambda_ce  * ce_loss + params.lambda_moco * supcon_loss 
        #composite_loss =  params.lambda_ce  * ce_loss
        train_loss  +=  ce_loss.item()  
        contrast_loss += supcon_loss 

        # if global_step%100 == 0:     
        #    print(supcon_loss.requires_grad)
        #    print(f'CE train loss: {params.lambda_ce  * ce_loss: .4f}')
        #    print(f'Contrast loss: {supcon_loss: .4f}')
           
        #composite_loss.backward(retain_graph=True)
        composite_loss.backward()
        if (i+1)%gradient_accumulate_step == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), params.grad_clip_norm)
            torch.nn.utils.clip_grad_norm_(fa_module.parameters(), params.grad_clip_norm)
            #torch.nn.utils.clip_grad_norm_(projection.parameters(), params.grad_clip_norm)
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), params.grad_clip_norm)
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
            scheduler.step()
        # optim.step()
            
        batch_acc = (output.argmax(1) == index).sum().item() / BATCH_SIZE
        valid_loss, valid_acc = test(total_valid, model, FA_module, classifier)
        update_lists(params, ce_loss, supcon_loss, batch_acc, valid_loss, valid_acc)
        
        train_acc += (output.argmax(1) == index).sum().item()

        # Save the model if it has the best validation accuracy so far
        if valid_acc > best_val_accuracy:
            best_val_accuracy = valid_acc
            best_model_state = model.state_dict()
            torch.save(best_model_state, 'best_model.pth')
            
    return train_loss / n_batches, contrast_loss / n_batches ,train_acc / (1 * len(train_data))

def test(test_data, model, fa_module, classifier):
    """
    function for evaluating the classifier
    """

    loss = 0
    acc = 0
    data = DataLoader(test_data, batch_size=BATCH_SIZE)
    pred = []
    pred_probs = []
    ground = []
    n_batches = len(data)
    model.eval()
    #fa_module.eval()
    classifier.eval()
    for i, pairs in enumerate(data):
        text = pairs[ :, 1:].to(device)
        index = pairs[ :, :1].to(device)
        text = torch.squeeze(text)
        index = torch.squeeze(index)
        index = index.long()
        
        text, index = text.to(device), index.to(device)
        with torch.no_grad():

            last_hidden = model(text)[0]
            last_hidden_cls = last_hidden[:,0,:]
            last_hidden_cls = fa_module(last_hidden_cls)
            output = classifier(last_hidden_cls)
            output = softmax(output)

            l = criterion(output, index)
            loss += l.item()
            acc += (output.argmax(1) == index).sum().item()
            pred_probs.append(output.detach().cpu().numpy())
            pred.append(output.argmax(1).detach().cpu().numpy())           
            ground.append(index.detach().cpu().numpy())
    return loss / n_batches, acc / len(test_data)

def cosine_sim(a,b):
    return np.dot(a,b) / ( (np.dot(a,a) **.5) * (np.dot(b,b) ** .5) )

class FAM(nn.Module):
    def __init__(self, embed_size, hidden_size, hidden_dropout_prob):
        super().__init__()
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.fc = nn.Linear(embed_size, hidden_size)
        #self.init_weights()
    def init_weights(self):
        initrange = 0.2
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()


    def forward(self, text):#, return_att = False):
        batch,  dim = text.size()
        # print(text.size())

        feat = self.fc(torch.tanh(self.dropout(text.view(batch, dim))))
        feat = F.normalize(feat, dim=1)
        return feat

class SupConHead(nn.Module):
    """backbone + projection head"""
    def __init__(self, head='mlp', dim_in=1024, feat_dim=256):
        super(SupConHead, self).__init__()
        #model_fun, dim_in = model_dict[name]
        #self.encoder = model_fun()
        
        if head == 'linear':
            self.head = nn.Linear(dim_in, feat_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                #nn(inplace=True),
                nn.Tanh(),
                nn.Linear(dim_in, feat_dim)
            )
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))

    def forward(self, x):
        #feat = self.encoder(x)
        feat = F.normalize(self.head(x), dim=1)
        return feat

class Projection(nn.Module):
    def __init__(self, hidden_size, projection_size):
        super().__init__()
        self.fc = nn.Linear(hidden_size, projection_size)
        self.ln = nn.LayerNorm(projection_size)
        self.bn = nn.BatchNorm1d(projection_size)
        self.init_weights()
    def init_weights(self):
        initrange = 0.01
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()


    def forward(self, text):#, return_att = False):
        #text = text.view()
        batch,  dim = text.size()

        return self.ln(self.fc(torch.tanh(text.view(batch, dim))))

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)

class CosineDistance(nn.Module):
    def __init__(self):
        super().__init__()
        self.cos = nn.CosineSimilarity(dim=1) 

    def forward(self, feature, centroid):
        cos_distance = 1 - self.cos(feature, centroid)
        #cos_distance =  self.cos(feature, centroid)
        #print(cos_distance.size())
        cos_distance = torch.mean(cos_distance, dim=0)    
        return cos_distance

class L2Distance(nn.Module):
    def __init__(self):
        super().__init__()
        #self.cos = nn.CosineSimilarity(dim=1)

    def forward(self, feature, centroid):
        L2_distance = (feature - centroid)**2
        #cos_distance =  self.cos(feature, centroid)
        #print(cos_distance.size())
        L2_distance = torch.sum(L2_distance, dim=1)
        L2_distance = torch.mean(L2_distance, dim=0)
        return L2_distance

class InterCosineSimilarity(nn.Module):
    def __init__(self):
        super().__init__()
        self.cos = nn.CosineSimilarity(dim=1)

    def forward(self, feature, centroid):
        cos_distance = 1 + self.cos(feature, centroid)
        #cos_distance =  self.cos(feature, centroid)
        #print(cos_distance.size())
        cos_distance = torch.mean(cos_distance, dim=0)
        return cos_distance

class PositiveContrastLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.cos = nn.CosineSimilarity(dim=1)

    def forward(self, feature, centroid, temp=1.0):
        feature_norm = torch.norm(feature, dim=1)
        centroid_norm = torch.norm(centroid, dim=1)
        batch_dot_product = torch.bmm(feature.view(feature.size()[0], 1, feature.size()[1]), centroid.view(centroid.size()[0], centroid.size()[1], 1))
        batch_dot_product =  batch_dot_product.squeeze()/(feature_norm * centroid_norm)
        batch_dot_product = torch.mean(batch_dot_product)     
        
        return  1 - 1*batch_dot_product

class MocoLoss(nn.Module):
    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature
    def forward(self, batch_examples, memory_bank, nclass):
        loss = 0
        for class_id in range(nclass):
          feature_index = torch.where(batch_examples[:,:1] == class_id)
          
          feature = batch_examples[feature_index[0]][:,1:]
          if feature.size()[0] <= 1 :
            continue
          feature_norm = torch.norm(feature, dim=1)
          feature = feature / feature_norm.unsqueeze(1)
          mask = torch.diag(torch.ones((feature.size()[0])))
          #print(mask)
          mask = (1 - mask).to(device)
          #print(mask)
          feature_dot_feature = torch.div(torch.matmul(feature, feature.T), self.temperature)
          if torch.isnan(feature_dot_feature).any():
            print("feature:")
            print(feature)
            exit()
          #logits_max, _ = torch.max(feature_dot_feature, dim=1)
          #feature_dot_feature = feature_dot_feature * mask.to(device)
          #feature_dot_feature = feature_dot_feature
          negative_index = torch.where(memory_bank[:,:1] != class_id)
          #print(negative_index)
          #exit() 
          negative_examples = memory_bank[negative_index[0]][:,1:].detach()
          negative_norm = torch.norm(negative_examples, dim=1)
          negative_examples = negative_examples / negative_norm.unsqueeze(1)
          
          feature_dot_negative = torch.div(torch.matmul(feature, negative_examples.T), self.temperature)
          logits_mask_positive = torch.ones(feature_dot_negative.size()).to(device)
          logits_mask_negative = torch.zeros(feature_dot_negative.size()).to(device)
          logits = torch.cat([feature_dot_feature,feature_dot_negative],dim=1)
          positive_mask = torch.cat([mask, logits_mask_negative], dim=1)
          logits_mask = torch.cat([mask, logits_mask_positive], dim=1)
          logits = logits * logits_mask
          #print(logits_mask)
          #print(positive_mask)
          logits_max, _ = torch.max(logits, dim=1, keepdim=True)
          logits = (logits - logits_max.detach()).to(device)
          #logits = logits_copy
          
          exp_logits = torch.exp(logits)
          
          c_loss = -1 * (logits*positive_mask).sum(1)/(logits*logits_mask).sum(1)
          #c_loss = -1 * (exp_logits*positive_mask).sum(1)/(exp_logits*logits_mask).sum(1)
          
          sums = exp_logits.sum(1,keepdims=True)
          log_prob = logits - torch.log(sums ) 
         
          mean_log_prob_pos = (positive_mask * log_prob).sum(1) / positive_mask.sum(1)
          #loss += c_loss.mean()
          loss += -1 * (0.07) * mean_log_prob_pos.mean()
          #print(log_pro)
          #print(positive_mask)
          #print(loss)
          
        return  loss

class NegativeContrastLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.cos = nn.CosineSimilarity(dim=1)

    def forward(self, feature, centroid, temp=1.0):
        feature_norm = torch.norm(feature, dim=1)
        centroid_norm = torch.norm(centroid, dim=1)
        batch_dot_product = torch.bmm(feature.view(feature.size()[0], 1, feature.size()[1]), centroid.view(centroid.size()[0], centroid.size()[1], 1))
        batch_dot_product = batch_dot_product.squeeze()/(feature_norm * centroid_norm)
        batch_dot_product = torch.mean(batch_dot_product)     
        #dot_product = torch.div( torch.matmul(feature, centroid.T), torch.matmul(feature_norm, centroid_norm.T))
        #dot_product = torch.div( torch.matmul(feature, centroid.T), temp)
        #batch_dot_product = torch.exp(batch_dot_product)
        #batch_dot_product = torch.log(batch_dot_product)
        #batch_dot_product = torch.mean(batch_dot_product.size())
        
        return batch_dot_product

class PairwiseCosineDistance(nn.Module):
    def __init__(self):
        super().__init__()
        self.cos = nn.CosineSimilarity(dim=1)
 
    def forward(self, feature):
        mean_distance=[]
        for cls in range(params.nclass):
            access = params.nclass * [True]
            access[cls] = False
            cls_cos_dist = 1 + self.cos(torch.stack((params.nclass - 1)* [feature[cls]],dim=0), feature[access])
            mean_distance.append(torch.mean(cls_cos_dist, dim=0)/2)
        mean_distance = torch.stack(mean_distance) 
        
        mean_distance = torch.mean(mean_distance, dim=0)
        return mean_distance

class PairwiseContrastLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.cos = nn.CosineSimilarity(dim=1)

    def forward(self, feature):
        nclass = feature.size()[0]
        feature_norm = torch.norm(feature, dim = 1)
        inter_class_mask = 1-torch.diag(torch.ones((nclass)))
        norm_dot_norm = torch.matmul(feature_norm,feature_norm.T)
        class_dot_class = torch.matmul(feature,feature.T)
        class_dot_class = torch.div(class_dot_class, norm_dot_norm)
        class_dot_class = inter_class_mask.to(device) * class_dot_class
        mean_distance = torch.sum(class_dot_class) / (nclass**2 - nclass)
        mean_distance = torch.log(torch.exp(mean_distance))
        return mean_distance

class CosineSimilarity(nn.Module):
    def __init__(self):
        super().__init__()
        self.cos = nn.CosineSimilarity(dim=1)

    def forward(self, feature, centroid, dummy):
        cos_similarity = self.cos(feature, centroid)
        #print(cos_distance.size())
        #exit()
        cos_similarity = torch.mean(cos_similarity, dim=0)
        return cos_similarity

def BatchMeanFeature(feature, index, nclass):
    batch_mean_feature_list = []
    batch_mean_feature_list_detach = []
    #batch_mean_feature_tensor = torch.zeros((nclass, feature.size()[1]), requires_grad=True)
    batch_mean_feature_tensor = []
    for cls_id in range(nclass):
       cls_id_idx = torch.where(index == cls_id)
       if len(cls_id_idx) > 0:
           batch_centroid = feature[cls_id_idx].detach()
           batch_mean_feature_list.append(feature[cls_id_idx])
           batch_mean_feature_list_detach.append(batch_centroid)
           batch_mean_feature_tensor.append(torch.mean(feature[cls_id_idx], dim=0))   ##mean feature of class cls_id within one batch
       else:
           batch_mean_feature_list.append(None)
    batch_mean_feature_tensor=torch.stack(batch_mean_feature_tensor, dim=0)
    return batch_mean_feature_tensor, batch_mean_feature_list, batch_mean_feature_list_detach

def InitMeanFeature(momentum_embedding, nclass):
    global class_centroids
    temp_tensor = []
    for cls_id in range(nclass):
        class_centroids[cls_id] = [x for x in class_centroids[cls_id] if x != None]
        print(torch.cat(class_centroids[cls_id], dim=0).size())
        init_mean_feat = torch.mean(torch.cat(class_centroids[cls_id], dim=0),dim=0)
        temp_tensor.append(init_mean_feat)
        class_centroids[cls_id] = []
    temp_tensor = torch.stack(temp_tensor, dim=0)
    momentum_embedding.requires_grad = False
    momentum_embedding.weight.data = temp_tensor
    return momentum_embedding

def IntraClassLoss(batch_mean_feature_list, momentum_embedding, distance_metric):
    nclass = len(batch_mean_feature_list)
    b_intra = 0
    for cls_id in range(nclass):
        if  batch_mean_feature_list[cls_id].size()[0] > 0:
            num_samples = batch_mean_feature_list[cls_id].size()[0]
            #print(num_samples)
            #print(momentum_embedding.weight.data[cls_id].size())
            b_intra += distance_metric(batch_mean_feature_list[cls_id],  torch.stack(num_samples * [momentum_embedding.weight.data[cls_id]], dim = 0) )/nclass
    return b_intra

def InterClassLoss(batch_mean_feature_list, momentum_embedding, inter_distance_metric):
    nclass = len(batch_mean_feature_list)
    b_inter = 0
    for cls_id in range(nclass):
        if batch_mean_feature_list[cls_id].size()[0]>0:
            access = torch.ones((nclass))
            access[cls_id] = 0
            inter_class = torch.where(access == 1)[0].numpy().tolist()
            num_samples = batch_mean_feature_list[cls_id].size()[0]
            for i_cls in  inter_class:
                b_inter += inter_distance_metric(batch_mean_feature_list[cls_id],  torch.stack(num_samples * [momentum_embedding.weight.data[i_cls]], dim = 0) )/(nclass*len(inter_class))
    return b_inter

class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, dim=257, K=1000, m=0.999, T=0.07):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo, self).__init__()
        self.K = K
        self.m = m
        self.T = T
        self.register_buffer("queue", torch.randn(dim, K))

def enqueue_and_dequeue(batch_examples, memory_bank_size):
    global memory_bank
    if memory_bank==None:
      memory_bank = batch_examples
    else:
      memory_bank = torch.cat([memory_bank, batch_examples.detach()] , dim=0)
      if memory_bank.size()[0] > memory_bank_size:
        memory_bank = memory_bank[-memory_bank_size:,:]
        
def UpdateMomentum(momentum_embedding, ema_updater, nclass):
    global class_centroids
    temp_tensor = []
    for cls_id in range(nclass):
        class_centroids[cls_id] = [x for x in class_centroids[cls_id] if x != None]
        init_mean_feat = torch.mean(torch.cat(class_centroids[cls_id], dim=0),dim=0)
        temp_tensor.append(init_mean_feat)
        class_centroids[cls_id] = []
    temp_tensor = torch.stack(temp_tensor, dim=0)
    update_embedding.requires_grad = False
    update_embedding.weight.data = temp_tensor
    update_moving_average(ema_updater, momentum_embedding, update_embedding)        
    return momentum_embedding

def set_seed(seed):
    """
    Helper function for reproducible behavior to set the seed in ``random``, ``numpy``, ``torch`` and/or ``tf`` (if
    installed).
    Args:
        seed (:obj:`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
        
class Classifier(nn.Module):
    def __init__(self, hidden_size, num_class, hidden_dropout_prob):
        super().__init__()
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.fc = nn.Linear(hidden_size, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.02
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, feature):
        return self.fc(torch.tanh(feature))

def get_parser():
    # parse parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", type=str, help="Experiment dump path")
    parser.add_argument("--nclass", type = int, default = 3)
    parser.add_argument("--nepochs", type = int, default = 10)
    parser.add_argument("--train_limited", type = bool, default = False)
    parser.add_argument("--save_model", type = bool, default = False)
    parser.add_argument("--batch_size", type = int, default = 12)
    parser.add_argument("--gradient_acc_step", type = int, default = 1)
    parser.add_argument("--projection_size", type = int, default = 512)
    parser.add_argument("--hidden_size", type = int, default = 256)
    parser.add_argument("--embedding_size", type = int, default = 1024)
    parser.add_argument("--max_length", type = int, default = 180)
    parser.add_argument("--warmup_step", type = int, default = 600)
    parser.add_argument("--skip_step", type = int, default = 300)
    parser.add_argument("--eval_step", type = int, default = 100)
    parser.add_argument("--m_update_interval", type = int, default = 10)
    parser.add_argument("--topk", type = int, default = 10)
    parser.add_argument("--seed", type = int, default = 24)
    parser.add_argument("--topk_use", type = int, default = 10)
    parser.add_argument("--valid_size", type = int, default = 200)
    parser.add_argument("--memory_bank_size", type = int, default = 200)
    parser.add_argument("--train_num", type = int, default = 10000)
    parser.add_argument("--train_few_shot", type = int, default = 0)
    parser.add_argument("--temp", type = float, default = 0.07)
    parser.add_argument("--hidden_dropout_prob", type = float, default = 0.0)
    parser.add_argument("--lambda_intra", type = float, default = 0.0)
    parser.add_argument("--lambda_inter", type = float, default = 0.0)
    parser.add_argument("--centroid_inter_loss", type = float, default = 0.0)
    parser.add_argument("--lambda_ce", type = float, default = 1.0)
    parser.add_argument("--lambda_kl", type = float, default = 0.0)
    parser.add_argument("--lambda_nce", type = float, default = 0.0)
    parser.add_argument("--lambda_supcon", type = float, default = 0.0)
    parser.add_argument("--lambda_adv", type = float, default = 0.0)
    parser.add_argument("--lambda_mmd", type = float, default = 0.0)
    parser.add_argument("--lambda_moco", type = float, default = 0.0)
    parser.add_argument("--centroid_decay", type = float, default = 0.99)
    parser.add_argument("--weight_decay", type = float, default = 0.98)
    parser.add_argument("--grad_clip_norm", type = float, default = 1.0)
    parser.add_argument("--lr", type = float, default = 5e-6)
    parser.add_argument("--model_name", type = str)
    parser.add_argument("--source_domain", type = str)
    parser.add_argument("--language_model", type = str,default='xlmr',help="Pre-trained language model: xlmr|roberta")
    parser.add_argument("--test_path", type = str)
    parser.add_argument("--valid_path", type = str)
    parser.add_argument("--dataset", type = str,default='mtl-dataset')
    parser.add_argument("--log", type = str,default='multi-domain-log')

    return parser

def update_lists(params, ce_loss, supcon_loss, batch_acc, valid_loss, valid_acc):
    global ce_train_loss_list, contrast_loss_list, acc_train_list, valid_loss_list, valid_acc_list
    
    # Append values to the lists
    ce_train_loss_list.append(params.lambda_ce * ce_loss)
    contrast_loss_list.append(supcon_loss)
    acc_train_list.append(batch_acc * 100)
    valid_loss_list.append(valid_loss)
    valid_acc_list.append(valid_acc * 100)

def create_plots(filename='plots'):
    # Move tensors to CPU memory and detach from the computation graph
    ce_train_loss_list_cpu = [item.detach().cpu().numpy() for item in ce_train_loss_list]
    contrast_loss_list_cpu = [item.detach().cpu().numpy() for item in contrast_loss_list]

    plt.figure(figsize=(15, 10))  # Create a single figure for all plots

    # Plot CE train loss
    plt.subplot(4, 3, 1)
    plt.plot(ce_train_loss_list_cpu)
    plt.title('CE Train Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')

    # Plot Contrast loss
    plt.subplot(4, 3, 2)
    plt.plot(contrast_loss_list_cpu)
    plt.title('Contrast Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')

    # Plot Accuracy
    plt.subplot(4, 3, 3)
    plt.plot(acc_train_list)
    plt.title('Train Accuracy')
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')

    # Plot Validation Loss
    plt.subplot(4, 3, 4)
    plt.plot(valid_loss_list)
    plt.title('Validation Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')

    # Plot Validation Accuracy
    plt.subplot(4, 3, 5)
    plt.plot(valid_acc_list)
    plt.title('Validation Accuracy')
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')

    # # Plot In-Domain Test Loss
    # plt.subplot(4, 3, 6)
    # plt.plot(test_loss_list)
    # plt.title('In-Domain Loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')

    # # Plot In-Domain Test Accuracy
    # plt.subplot(4, 3, 7)
    # plt.plot(test_acc_list)
    # plt.title('In-Domain Accuracy')
    # plt.xlabel('Epoch')
    # plt.ylabel('Accuracy')

    # Create subplots for Out-of-Domain test loss and accuracy within the same figure
    for j, test_domain in enumerate(test_domains):
        # Plot Out-of-Domain test loss
        plt.subplot(4, 3, 6 + j*2)  # Adjust subplot index
        plt.plot(test_out_loss_per_domain[j], label=f'Domain {test_domain} Loss')
        plt.title(f'Out-of-Domain Test Loss - {test_domain}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.tight_layout()

        # Plot Out-of-Domain test accuracy
        plt.subplot(4, 3, 6 + j*2 + 1)  # Adjust subplot index
        plt.plot(test_out_acc_per_domain[j], label=f'Domain {test_domain} Accuracy')
        plt.title(f'Out-of-Domain Test Accuracy - {test_domain}')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.tight_layout()

    # Save the combined plots to a PDF file
    plt.tight_layout()
    plt.savefig(filename + '.pdf')
    plt.show()

class NewsDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

def get_embeddings(sentences, model, tokenizer, device, batch_size=16):
    model.to(device)
    embeddings_list = []
    
    for i in range(0, len(sentences), batch_size):
        batch_sentences = sentences[i:i + batch_size]
        inputs = tokenizer(batch_sentences, padding=True, truncation=True, return_tensors='pt').to(device)
        with torch.no_grad():
            outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
        embeddings = outputs[0]  # Get the last_hidden_state
        sentence_embeddings = embeddings.mean(dim=1)
        embeddings_list.append(sentence_embeddings.cpu())

    return torch.cat(embeddings_list)

def sample_data(df, sample_size):
    sampled_df = df.groupby('generation_approach', group_keys=False).apply(lambda x: x.sample(min(len(x), sample_size), random_state=42))
    return sampled_df

def get_color(label, domain):
    base_colors = {
        'gpt-3.5-turbo': 'green', 
        'llama2_70b': 'red', 
        'vicuna-v1.3_33b': 'blue',
        # Add more label base colors if necessary
    }
    domain_shades = {
        'paraphrase_generation': 0.8, 
        'rewrite_generation': 0.5,
        'open_ended_generation': 0.3, 
        # Add more domain shades if necessary
    }
    base_color = base_colors.get(label, 'gray')  # Default to gray if label not found
    shade = domain_shades.get(domain, 0.5)       # Default to a mid shade if domain not found
    color = sns.light_palette(base_color, input="rgb", n_colors=10)[int(shade * 10)]
    return color

def vis(data, model, tokenizer):
    sampled_data_vis = sample_data(data, 500)
    
    X_sampled_data_vis, y_sampled_data_vis = sampled_data_vis['synthetic misinformation'], sampled_data_vis['generated_by']
    domains = sampled_data_vis['generation_approach']

    label_encoder = LabelEncoder()
    y_sampled_data_vis_encoded = label_encoder.fit_transform(y_sampled_data_vis)  # Use the same encoder for visualization
    domain_encoder = LabelEncoder()
    domains_encoded = domain_encoder.fit_transform(domains)

    vis_encodings = tokenizer(list(X_sampled_data_vis), truncation=True, padding=True, return_tensors='pt')
    vis_encodings = {k: v.to(device) for k, v in vis_encodings.items()}
    vis_labels = torch.tensor(y_sampled_data_vis_encoded).to(device)


    model.eval()

    # Get embeddings for the vis data
    sentence_embeddings = get_embeddings(list(X_sampled_data_vis), model, tokenizer, device)

    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    test_embeddings_2d = tsne.fit_transform(sentence_embeddings.cpu().numpy())

    # Create a DataFrame for easy plotting
    df = pd.DataFrame({
        'x': test_embeddings_2d[:, 0],
        'y': test_embeddings_2d[:, 1],
        'label': y_sampled_data_vis,
        'domain': domains
    })

    # Get colors
    colors = [get_color(label, domain) for label, domain in zip(df['label'], df['domain'])]

    # Plotting
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(df['x'], df['y'], c=colors, s=50, alpha=0.7)
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.title('t-SNE Visualization of SCL Embeddings')

    # Create legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='ChatGPT - Paraphrase', markerfacecolor=sns.light_palette("blue", input="rgb", n_colors=10)[8], markersize=10),
        Line2D([0], [0], marker='o', color='w', label='ChatGPT - Rewrite', markerfacecolor=sns.light_palette("blue", input="rgb", n_colors=10)[5], markersize=10),
        Line2D([0], [0], marker='o', color='w', label='ChatGPT - Open_ended', markerfacecolor=sns.light_palette("blue", input="rgb", n_colors=10)[3], markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Llama2 - Paraphrase', markerfacecolor=sns.light_palette("red", input="rgb", n_colors=10)[8], markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Llama2 - Rewrite', markerfacecolor=sns.light_palette("red", input="rgb", n_colors=10)[5], markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Llama2 - Open_ended', markerfacecolor=sns.light_palette("red", input="rgb", n_colors=10)[3], markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Vicuna - Paraphrase', markerfacecolor=sns.light_palette("green", input="rgb", n_colors=10)[8], markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Vicuna - Rewrite', markerfacecolor=sns.light_palette("green", input="rgb", n_colors=10)[5], markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Vicuna - Open_ended', markerfacecolor=sns.light_palette("green", input="rgb", n_colors=10)[3], markersize=10),
    ]
    plt.legend(handles=legend_elements, loc='best')

    plt.tight_layout()
    plt.savefig('SCL_tSNE_Visualization.pdf')
    plt.show()

def run_code(data, source_domains, test_domains):
    global params 
    global BATCH_SIZE
    global optimizer
    global scheduler
    global criterion
    global contrast_criterion
    global FA_module
    global update_embedding
    global softmax
    global memory_bank
    global global_step
    global ce_train_loss_list
    ce_train_loss_list = []
    global contrast_loss_list
    contrast_loss_list = []
    global acc_train_list
    acc_train_list = []

    global valid_loss_list
    valid_loss_list = []
    global valid_acc_list
    valid_acc_list = []

    # global test_loss_list = []
    # global test_acc_list = []

    global test_out_loss_per_domain
    test_out_loss_per_domain = []
    global test_out_acc_per_domain
    test_out_acc_per_domain = []
    
    parser = get_parser()
    params = parser.parse_args()
    print(params)
    set_seed(params.seed)

    model_name = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name).to(device)
    # Freeze the first 9 layers of the encoder
    # for layer in model.encoder.layer[:9]:
    #     for param in layer.parameters():
    #         param.requires_grad = False
    # model.train()
    # print(type(model))

    max_length = params.max_length

    test_out_loss_per_domain = [[] for _ in range(len(test_domains))]
    test_out_acc_per_domain = [[] for _ in range(len(test_domains))]

    domain_label_map = {}
    i = 0
    for domain in source_domains:
      domain_label_map[domain] = i
      i += 1
    print(domain_label_map)

    train_data = {}
    valid_data = {}
    test_data = {}
    total_valid = []
    total_train = []
    total_test = []

    test_out_data = {}
    
    for domain in source_domains:
      train_data[domain], valid_data[domain], test_data[domain]  = xlm_r_train_valid_test(data[data['generation_approach'] == domain], tokenizer,max_length, domain_label_map)
      total_train.append(train_data[domain])
      total_valid.append(valid_data[domain])
      total_test.append(test_data[domain])


    total_train = torch.cat(total_train, dim=0)
    print(f'Total Training Examples: {total_train.size()[0]:d}')
    total_valid = torch.cat(total_valid, dim=0)
    print(f'Total Validation Examples: {total_valid.size()[0]:d}')
    total_test = torch.cat(total_test, dim=0)
    print(f'Total In-Domain Test Examples: {total_test.size()[0]:d}')


    for test_domain in test_domains:
      test_out_data[test_domain] = xlm_r_test(data[data['generation_approach'] == test_domain], tokenizer, max_length)    
 
    ema_updater = EMA(params.centroid_decay)
    BATCH_SIZE=params.batch_size
    gradient_accumulate_step = params.gradient_acc_step
    FA_module = FAM(params.embedding_size, params.hidden_size, params.hidden_dropout_prob).to(device)
    projection = Projection(params.hidden_size, params.projection_size).to(device)
    classifier = Classifier(params.hidden_size, params.nclass, params.hidden_dropout_prob).to(device)
    domain_classifier = Classifier(params.hidden_size, len(source_domains), params.hidden_dropout_prob).to(device)
    softmax = nn.Softmax(dim=1)
    l1_criterion = torch.nn.SmoothL1Loss(reduction='mean').to(device)
    hinge_criterion = nn.HingeEmbeddingLoss(reduction = 'mean').to(device)
    #distance_metric = nn.CosineSimilarity(dim = 1)
    #distance_metric = L2Distance()
    #distance_metric = nn.MSELoss()
    #distance_metric = nn.MSELoss(reduction='mean').to(device)
    distance_metric = PositiveContrastLoss()
    moco_loss = MocoLoss(params.temp)
    #pairwise_dist = PairwiseCosineDistance()
    pairwise_dist = PairwiseContrastLoss()
    inter_distance_metric = NegativeContrastLoss()
    cos_embedding_loss = nn.CosineEmbeddingLoss(margin = 0.0, reduction='mean').to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    domain_criterion = torch.nn.CrossEntropyLoss().to(device)
    

    momentum_embedding = nn.Embedding(params.nclass, params.hidden_size)
    update_embedding = nn.Embedding(params.nclass, params.hidden_size)

    contrast_criterion = SupConLoss(temperature=params.temp).to(device)
    optimizer_grouped_parameters = [{"params": [p for n, p in model.named_parameters()  if p.requires_grad == True], 'weight_decay': 0.0 } ,{'params': domain_classifier.parameters()},{'params': FA_module.parameters()}, {'params': classifier.parameters()}]
    optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=params.lr)
    #optimizer = torch.optim.Adam([ {'params': classifier.parameters()},  {'params':model.parameters()}], lr=params.lr)
    t_total = len(DataLoader(total_train, batch_size=BATCH_SIZE, shuffle=True))* params.nepochs / params.gradient_acc_step 
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=params.warmup_step, num_training_steps=t_total)
    index_map_3_class = {0:0, 1:0, 2:1, 3:2, 4:2}   
    memory_bank = None
    results_table = pd.DataFrame(columns=['test_domain','epoch','test_in_domain_loss','test_in_domain_acc','test_out_of_domain_loss','test_out_of_domain_acc'])

    start = time.time()

    #global_step
    global_step = 0 
    for i in range(params.nepochs):
      train_loss, contrast_loss, train_acc = train(total_train, total_valid, model, FA_module, classifier, gradient_accumulate_step, i)
      valid_loss, valid_acc = test(total_valid, model, FA_module, classifier)
      test_loss, test_acc = test(total_test, model, FA_module, classifier)
    #   test_loss_list.append(test_loss)
    #   test_acc_list.append(test_acc)
  
      end = time.time()
    
      print(f'\tEpoch: {i+1:.4f}\t|\tTime Elapsed: {end-start:.1f}s')
      print(f'\tCE: {train_loss:.4f}(train) \t| Contrast_loss: {contrast_loss:.4f} \tAcc: {train_acc * 100:.2f}%(train) \t|\t')
      print(f'\tLoss: {valid_loss:.4f}(valid_loss)\t|\tAcc: {valid_acc * 100:.2f}%(valid_acc)')
      
      print(f'\tLoss: {test_loss:.4f}(test_loss)\t|\tAcc: {test_acc * 100:.2f}%(test_acc)')

      for j, test_domain in enumerate(test_domains):
        test_out_loss, test_out_acc = test(test_out_data[test_domain],model, FA_module, classifier)
        print(f'\tDomain: {test_domain:s} | Loss: {test_out_loss:.4f}(test_loss)\t|\tAcc: {test_out_acc * 100:.2f}%(test_acc)')
        test_out_loss_per_domain[j].append(test_out_loss)
        test_out_acc_per_domain[j].append(test_out_acc)
        if (i+1) >= 0:
            new_row_data = {'test_domain': test_domain,'epoch':i+1,'test_in_domain_loss':test_loss, 'test_in_domain_acc':test_acc,'test_out_of_domain_loss':test_out_loss,'test_out_of_domain_acc':test_out_acc}
            results_table = results_table._append(new_row_data, ignore_index=True)
      start = time.time()

    for domain_test in test_domains:
      domain_result = results_table[results_table['test_domain']==domain_test].copy()
      print(domain_result.sort_values(by=['test_out_of_domain_acc'], ascending=False).head())


    # Load the best model
    model.load_state_dict(torch.load('best_model.pth'))
    test_loss, test_acc = test(total_test, model, FA_module, classifier)
    print('Test Accuracy on the best Validation Score')
    print(f'\tLoss: {test_loss:.4f}(test_loss)\t|\tAcc: {test_acc * 100:.2f}%(test_acc)')

    for j, test_domain in enumerate(test_domains):
        test_out_loss, test_out_acc = test(test_out_data[test_domain],model, FA_module, classifier)
        print(f'\tDomain: {test_domain:s} | Loss: {test_out_loss:.4f}(test_loss)\t|\tAcc: {test_out_acc * 100:.2f}%(test_acc)')


    # Concatenate all items with underscore
    plot_name = '_'.join(source_domains)
    create_plots('FREEZE_'+ 'DG' + plot_name)

    vis(data, model, tokenizer)

    # if params.save_model:
    #   torch.save({'model': model, 'fam': FA_module, 'classifier': classifier}, 'model_{}.pt'.format(domain_test))


if __name__ == '__main__':
    sim_model = SentenceTransformer('all-MiniLM-L6-v2')

    # Load datasets
    path = './data/'
    llms = ['gpt-3.5-turbo', 'llama2_70b', 'vicuna-v1.3_33b']
    dataset = ['coaid', 'gossipcop', 'politifact']
    domains = ['open_ended_generation', 'paraphrase_generation', 'rewrite_generation']
    human = False
    
    data = load_data(llms, dataset, domains, human)
    data = filter_data(data)

    source_domains = ['open_ended_generation', 'rewrite_generation']
    test_domains = ['paraphrase_generation']
    run_code(data, source_domains, test_domains)

    # Generate combinations of different sizes, excluding the full set
    # for r in range(1, len(domains)):
    #     for subset in combinations(domains, r):
    #         source_domains = list(subset)
    #         test_domains = [x for x in domains if x not in subset]
    #         print('Train Generation Approach: {}'.format(source_domains))
    #         print('--------------------------------------------------------------')
    #         print('Test Generation Approach: {}'.format(test_domains))
    #         print('--------------------------------------------------------------')
    #         run_code(data, source_domains, test_domains)
    #         print('###############################')