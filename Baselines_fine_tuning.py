import pandas as pd
import torch.nn as nn
import numpy as np
import os
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import DebertaModel, DebertaTokenizer, BertModel, BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW 
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.model_selection import KFold
import itertools
from itertools import combinations
from sentence_transformers import SentenceTransformer, util
from transformers import DebertaTokenizer, DebertaForSequenceClassification, AdamW
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
import seaborn as sns

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
    embeddings1 = model.encode(text1, convert_to_tensor=True)
    embeddings2 = model.encode(text2, convert_to_tensor=True)
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

def fine_tune_BERT(data):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(data['synthetic misinformation'], data['generated_by'], test_size=0.2, random_state=42)

    # Use LabelEncoder to encode the labels
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    # Load pre-trained BERT model and tokenizer
    model_name = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=data['generated_by'].nunique())

    # Move the model to the GPU
    model.to(device)

    # Tokenize and encode the training data
    train_encodings = tokenizer(list(X_train), truncation=True, padding=True, return_tensors='pt')
    train_labels = torch.tensor(y_train_encoded).to(device)

    # Tokenize and encode the testing data
    test_encodings = tokenizer(list(X_test), truncation=True, padding=True, return_tensors='pt')
    test_labels = torch.tensor(y_test_encoded).to(device)

    # Create a PyTorch dataset
    class NewsDataset(Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: val[idx].clone().detach().to(device) for key, val in self.encodings.items()}
            item['labels'] = self.labels[idx]
            return item

        def __len__(self):
            return len(self.labels)


    train_dataset = NewsDataset(train_encodings, train_labels)
    test_dataset = NewsDataset(test_encodings, test_labels)

    # Set up DataLoader for training and testing
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    # Fine-tune the BERT model
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    for epoch in tqdm(range(10), desc='Training Epochs'):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

    # Evaluate the fine-tuned model on the test set
    model.eval()
    all_predictions = []

    with torch.no_grad():
        for batch in test_loader:
            outputs = model(**batch)
            predictions = torch.argmax(outputs.logits, dim=1)
            all_predictions.extend(predictions.cpu().tolist())

    # Convert predicted labels back to their original string form
    all_predictions_str = label_encoder.inverse_transform(all_predictions)

    # Evaluate the accuracy of the model
    accuracy = accuracy_score(y_test, all_predictions_str)
    print(f'Accuracy: {accuracy}')

    # You can also print a classification report for more detailed evaluation
    print(classification_report(y_test, all_predictions_str))

######################################################################################################

def fine_tune_BERT_with_kfold(data, k=5):
    # Initialize KFold
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    # Use LabelEncoder to encode the labels
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(data['generated_by'])

    # Convert to numpy for indexing in folds
    data_np = data['synthetic misinformation'].to_numpy()
    labels_encoded_np = labels_encoded

    # Prepare tokenizer and model
    model_name = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    accuracies = []

    for fold, (train_index, test_index) in enumerate(kf.split(data_np), 1):
        print(f"FOLD {fold}")
        print("-------------------------------")

        X_train, X_test = data_np[train_index], data_np[test_index]
        y_train_encoded, y_test_encoded = labels_encoded_np[train_index], labels_encoded_np[test_index]

        # Tokenize and encode the data
        train_encodings = tokenizer(list(X_train), truncation=True, padding=True, return_tensors='pt')
        train_labels = torch.tensor(y_train_encoded).to(device)

        test_encodings = tokenizer(list(X_test), truncation=True, padding=True, return_tensors='pt')
        test_labels = torch.tensor(y_test_encoded).to(device)

        # Create PyTorch datasets
        train_dataset = NewsDataset(train_encodings, train_labels)
        test_dataset = NewsDataset(test_encodings, test_labels)

        # DataLoader setup
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

        # Model and optimizer setup
        model = BertForSequenceClassification.from_pretrained(model_name, num_labels=data['generated_by'].nunique())
        model.to(device)
        optimizer = AdamW(model.parameters(), lr=1e-5)

        # Training loop
        for epoch in tqdm(range(10), desc=f'Training Epochs for fold {fold}'):
            model.train()
            for batch in train_loader:
                optimizer.zero_grad()
                outputs = model(**batch)
                loss = outputs.loss
                loss.backward()
                optimizer.step()

        # Evaluation
        model.eval()
        all_predictions = []

        with torch.no_grad():
            for batch in test_loader:
                outputs = model(**batch)
                predictions = torch.argmax(outputs.logits, dim=1)
                all_predictions.extend(predictions.cpu().tolist())

        all_predictions_str = label_encoder.inverse_transform(all_predictions)
        accuracy = accuracy_score(label_encoder.inverse_transform(y_test_encoded), all_predictions_str)
        accuracies.append(accuracy)
        print(f'Accuracy for fold {fold}: {accuracy}')
        print(classification_report(label_encoder.inverse_transform(y_test_encoded), all_predictions_str))

    print(f'Mean accuracy over {k} folds: {np.mean(accuracies)}')

class NewsDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach().to(device) for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)
    
######################################################################################################
# Function to evaluate the model
def evaluate_model(test_loader, model, label_encoder):
    model.eval()
    all_predictions = []
    all_true_labels = []

    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            predictions = torch.argmax(outputs.logits, dim=1)
            all_predictions.extend(predictions.cpu().tolist())
            all_true_labels.extend(batch['labels'].cpu().tolist())

    # Convert predictions
    all_predictions_str = label_encoder.inverse_transform(all_predictions)
    all_true_labels_str = label_encoder.inverse_transform(all_true_labels)

    # Evaluation
    accuracy = accuracy_score(all_true_labels_str, all_predictions_str)
    print(f'Accuracy: {accuracy}')
    print(classification_report(all_true_labels_str, all_predictions_str))

# Main function to fine-tune BERT and evaluate on domain test
def fine_tune_BERT_domain_test(data, train_generation_approach, test_generation_approach):
    # Filter data for training and testing
    train_data = data[data['generation_approach'].isin(train_generation_approach)]
    test_data = data[data['generation_approach'].isin(test_generation_approach)]
    
    # Split the training and testing data
    X_train, y_train = train_data['synthetic misinformation'], train_data['generated_by']
    X_test, y_test = test_data['synthetic misinformation'], test_data['generated_by']

    # Preprocess labels
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    # Load BERT tokenizer and model
    model_name = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=len(label_encoder.classes_))
    model.to(device)

    # Tokenize and encode the datasets
    train_encodings = tokenizer(list(X_train), truncation=True, padding=True, return_tensors='pt')
    test_encodings = tokenizer(list(X_test), truncation=True, padding=True, return_tensors='pt')
    
    train_labels = torch.tensor(y_train_encoded)
    test_labels = torch.tensor(y_test_encoded)

    # Create datasets and dataloaders
    train_dataset = NewsDataset(train_encodings, train_labels)
    test_dataset = NewsDataset(test_encodings, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    # Fine-tune the BERT model
    optimizer = AdamW(model.parameters(), lr=5e-5)
    for epoch in tqdm(range(10), desc='Training Epochs'):
        model.train()
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

    # Evaluate on the combined test dataset for overall performance
    print("Evaluating on all test generation approaches combined")
    evaluate_model(test_loader, model, label_encoder)
    print('***************************')

    # Prepare and evaluate the model on each test generation approach separately
    for approach in test_generation_approach:
        print(f"Evaluating on test generation approach: {approach}")
        approach_data = test_data[test_data['generation_approach'] == approach]
        X_approach, y_approach = approach_data['synthetic misinformation'], approach_data['generated_by']
        approach_encodings = tokenizer(list(X_approach), truncation=True, padding=True, return_tensors='pt')
        approach_labels = torch.tensor(label_encoder.transform(y_approach))

        approach_dataset = NewsDataset(approach_encodings, approach_labels)
        approach_loader = DataLoader(approach_dataset, batch_size=8, shuffle=False)

        evaluate_model(approach_loader, model, label_encoder)
        print('-----------------------------')

#######################################################################################################
def fine_tune_DeBERTa_with_kfold(data, k=5):
    # Initialize KFold
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    # Use LabelEncoder to encode the labels
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(data['generated_by'])

    # Convert to numpy for indexing in folds
    data_np = data['synthetic misinformation'].to_numpy()
    labels_encoded_np = labels_encoded

    # Prepare tokenizer and model
    model_name = 'microsoft/deberta-base'
    tokenizer = DebertaTokenizer.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    accuracies = []

    for fold, (train_index, test_index) in enumerate(kf.split(data_np), 1):
        print(f"FOLD {fold}")
        print("-------------------------------")

        X_train, X_test = data_np[train_index], data_np[test_index]
        y_train_encoded, y_test_encoded = labels_encoded_np[train_index], labels_encoded_np[test_index]

        # Tokenize and encode the data
        train_encodings = tokenizer(list(X_train), truncation=True, padding=True, return_tensors='pt')
        train_labels = torch.tensor(y_train_encoded).to(device)

        test_encodings = tokenizer(list(X_test), truncation=True, padding=True, return_tensors='pt')
        test_labels = torch.tensor(y_test_encoded).to(device)

        # Create PyTorch datasets
        train_dataset = NewsDataset(train_encodings, train_labels)
        test_dataset = NewsDataset(test_encodings, test_labels)

        # DataLoader setup
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

        # Model and optimizer setup
        model = DebertaForSequenceClassification.from_pretrained(model_name, num_labels=data['generated_by'].nunique())
        model.to(device)
        optimizer = AdamW(model.parameters(), lr=1e-5)

        # Training loop
        for epoch in tqdm(range(10), desc=f'Training Epochs for fold {fold}'):
            model.train()
            for batch in train_loader:
                optimizer.zero_grad()
                outputs = model(**batch)
                loss = outputs.loss
                loss.backward()
                optimizer.step()

        # Evaluation
        model.eval()
        all_predictions = []

        with torch.no_grad():
            for batch in test_loader:
                outputs = model(**batch)
                predictions = torch.argmax(outputs.logits, dim=1)
                all_predictions.extend(predictions.cpu().tolist())

        all_predictions_str = label_encoder.inverse_transform(all_predictions)
        accuracy = accuracy_score(label_encoder.inverse_transform(y_test_encoded), all_predictions_str)
        accuracies.append(accuracy)
        print(f'Accuracy for fold {fold}: {accuracy}')
        print(classification_report(label_encoder.inverse_transform(y_test_encoded), all_predictions_str))

    print(f'Mean accuracy over {k} folds: {np.mean(accuracies)}')

#######################################################################################################
def fine_tune_DeBERTa_domain_test(data, train_generation_approach, test_generation_approach):
    # Filter data for training and testing
    train_data = data[data['generation_approach'].isin(train_generation_approach)]
    test_data = data[data['generation_approach'].isin(test_generation_approach)]
    
    # Split the training and testing data
    X_train, y_train = train_data['synthetic misinformation'], train_data['generated_by']
    X_test, y_test = test_data['synthetic misinformation'], test_data['generated_by']

    # Preprocess labels
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    # Load DeBERTa tokenizer and model
    model_name = 'microsoft/deberta-base'
    tokenizer = DebertaTokenizer.from_pretrained(model_name)
    model = DebertaForSequenceClassification.from_pretrained(model_name, num_labels=len(label_encoder.classes_))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Tokenize and encode the datasets
    train_encodings = tokenizer(list(X_train), truncation=True, padding=True, return_tensors='pt')
    test_encodings = tokenizer(list(X_test), truncation=True, padding=True, return_tensors='pt')
    
    train_labels = torch.tensor(y_train_encoded).to(device)
    test_labels = torch.tensor(y_test_encoded).to(device)

    # Create datasets and dataloaders
    train_dataset = NewsDataset(train_encodings, train_labels)
    test_dataset = NewsDataset(test_encodings, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    # Fine-tune the DeBERTa model
    optimizer = AdamW(model.parameters(), lr=5e-5)
    for epoch in tqdm(range(10), desc='Training Epochs'):
        model.train()
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

    # Evaluate on the combined test dataset for overall performance
    print("Evaluating on all test generation approaches combined")
    evaluate_model(test_loader, model, label_encoder)
    print('***************************')

    # Prepare and evaluate the model on each test generation approach separately
    for approach in test_generation_approach:
        print(f"Evaluating on test generation approach: {approach}")
        approach_data = test_data[test_data['generation_approach'] == approach]
        X_approach, y_approach = approach_data['synthetic misinformation'], approach_data['generated_by']
        approach_encodings = tokenizer(list(X_approach), truncation=True, padding=True, return_tensors='pt')
        approach_labels = torch.tensor(label_encoder.transform(y_approach)).to(device)

        approach_dataset = NewsDataset(approach_encodings, approach_labels)
        approach_loader = DataLoader(approach_dataset, batch_size=8, shuffle=False)

        evaluate_model(approach_loader, model, label_encoder)
        print('-----------------------------')

#######################################################################################################
# def sample_data(df, sample_size):
#     unique_values = df['generation_approach'].unique()
#     sampled_df_list = []

#     for value in unique_values:
#         # Sample `sample_size` rows for each unique value
#         sampled_df = df[df['generation_approach'] == value].sample(n=sample_size, random_state=42)
#         sampled_df_list.append(sampled_df)

#     # Concatenate the sampled DataFrames
#     sampled_df = pd.concat(sampled_df_list)
#     return sampled_df

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

class BertForFineTuning(nn.Module):
    def __init__(self, model_name, num_labels):
        super(BertForFineTuning, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
    
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        logits = self.classifier(pooled_output)
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
        return loss, logits, outputs.last_hidden_state

def get_bert_embeddings(sentences, model, tokenizer, device, batch_size=16):
    model.to(device)
    embeddings_list = []
    
    for i in range(0, len(sentences), batch_size):
        batch_sentences = sentences[i:i + batch_size]
        inputs = tokenizer(batch_sentences, padding=True, truncation=True, return_tensors='pt').to(device)
        with torch.no_grad():
            _, _, embeddings = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
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

def fine_tune_BERT_vis(data, train_generation_approach, test_generation_approach):
    sampled_data_vis = sample_data(data, 500)
    train_data = data[data['generation_approach'].isin(train_generation_approach)]
    
    X_train, y_train = train_data['synthetic misinformation'], train_data['generated_by']
    X_sampled_data_vis, y_sampled_data_vis = sampled_data_vis['synthetic misinformation'], sampled_data_vis['generated_by']
    domains = sampled_data_vis['generation_approach']  # Assuming 'domain' column exists in your data

    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_sampled_data_vis_encoded = label_encoder.fit_transform(y_sampled_data_vis)  # Use the same encoder for visualization
    domain_encoder = LabelEncoder()
    domains_encoded = domain_encoder.fit_transform(domains)

    model_name = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForFineTuning(model_name, num_labels=len(label_encoder.classes_))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    train_encodings = tokenizer(list(X_train), truncation=True, padding=True, return_tensors='pt')
    train_labels = torch.tensor(y_train_encoded).to(device)
    train_dataset = NewsDataset(train_encodings, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

    vis_encodings = tokenizer(list(X_sampled_data_vis), truncation=True, padding=True, return_tensors='pt')
    vis_encodings = {k: v.to(device) for k, v in vis_encodings.items()}
    vis_labels = torch.tensor(y_sampled_data_vis_encoded).to(device)

    optimizer = AdamW(model.parameters(), lr=5e-5)
    for epoch in tqdm(range(10), desc='Training Epochs'):
        model.train()
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            loss, _, _ = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels'])
            loss.backward()
            optimizer.step()

    model.eval()

    # Get embeddings for the vis data
    sentence_embeddings = get_bert_embeddings(list(X_sampled_data_vis), model, tokenizer, device)

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
    plt.title('t-SNE Visualization of BERT Embeddings')

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
    plt.savefig('BERT_tSNE_Visualization.pdf')
    plt.show()

#######################################################################################################
class DebertaForFineTuning(nn.Module):
    def __init__(self, model_name, num_labels):
        super(DebertaForFineTuning, self).__init__()
        self.deberta = DebertaModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.deberta.config.hidden_size, num_labels)
    
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.deberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]  # Use the [CLS] token representation
        logits = self.classifier(pooled_output)
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
        return loss, logits, outputs.last_hidden_state

def get_deberta_embeddings(sentences, model, tokenizer, device, batch_size=16):
    model.to(device)
    embeddings_list = []
    
    for i in range(0, len(sentences), batch_size):
        batch_sentences = sentences[i:i + batch_size]
        inputs = tokenizer(batch_sentences, padding=True, truncation=True, return_tensors='pt').to(device)
        with torch.no_grad():
            _, _, embeddings = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
        sentence_embeddings = embeddings.mean(dim=1)
        embeddings_list.append(sentence_embeddings.cpu())

    return torch.cat(embeddings_list)

def fine_tune_DeBERTa_vis(data, train_generation_approach, test_generation_approach):
    sampled_data_vis = sample_data(data, 500)
    train_data = data[data['generation_approach'].isin(train_generation_approach)]
    
    X_train, y_train = train_data['synthetic misinformation'], train_data['generated_by']
    X_sampled_data_vis, y_sampled_data_vis = sampled_data_vis['synthetic misinformation'], sampled_data_vis['generated_by']
    domains = sampled_data_vis['generation_approach']

    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_sampled_data_vis_encoded = label_encoder.fit_transform(y_sampled_data_vis)
    domain_encoder = LabelEncoder()
    domains_encoded = domain_encoder.fit_transform(domains)

    model_name = 'microsoft/deberta-base'
    tokenizer = DebertaTokenizer.from_pretrained(model_name)
    model = DebertaForFineTuning(model_name, num_labels=len(label_encoder.classes_))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    train_encodings = tokenizer(list(X_train), truncation=True, padding=True, return_tensors='pt')
    train_labels = torch.tensor(y_train_encoded).to(device)
    train_dataset = NewsDataset(train_encodings, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

    vis_encodings = tokenizer(list(X_sampled_data_vis), truncation=True, padding=True, return_tensors='pt')
    vis_encodings = {k: v.to(device) for k, v in vis_encodings.items()}
    vis_labels = torch.tensor(y_sampled_data_vis_encoded).to(device)

    optimizer = AdamW(model.parameters(), lr=5e-5)
    for epoch in tqdm(range(10), desc='Training Epochs'):
        model.train()
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            loss, _, _ = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels'])
            loss.backward()
            optimizer.step()

    model.eval()

    # Get embeddings for the vis data
    sentence_embeddings = get_deberta_embeddings(list(X_sampled_data_vis), model, tokenizer, device)

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
    plt.title('t-SNE Visualization of DeBERTa Embeddings')

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
    # plt.savefig('DeBERTa_tSNE_Visualization.pdf')
    plt.savefig('SCL_tSNE_Visualization.pdf')

    plt.show()   

##############################################################################################
# Check if a GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
torch.cuda.empty_cache()
torch.manual_seed(42)

##############################################################################################
print('In Domain Results')
print('--------------------------------------------------------------')

# Load datasets
path = './data/'
llms = ['gpt-3.5-turbo','llama2_70b', 'vicuna-v1.3_33b']
dataset = ['coaid', 'gossipcop', 'politifact']
generation_approach = ['open_ended_generation', 'paraphrase_generation', 'rewrite_generation']
human = False

# # #llms = ['gpt-3.5-turbo', 'llama2_7b', 'llama2_13b', 'llama2_70b', 'vicuna-v1.3_7b', 'vicuna-v1.3_13b', 'vicuna-v1.3_33b']

# # Load a pre-trained model
model = SentenceTransformer('all-MiniLM-L6-v2')

data = load_data(llms, dataset, generation_approach , human)
filtered_data = filter_data(data)

train_generation_approach = ['open_ended_generation', 'rewrite_generation', 'paraphrase_generation']
test_generation_approach = ['paraphrase_generation']


# fine_tune_BERT_vis(data, train_generation_approach, test_generation_approach)
fine_tune_DeBERTa_vis(data, train_generation_approach, test_generation_approach)

# # Generate combinations for every possible length and add them to the list
# for i in range(1, len(generation_approach) + 1):
#     combinations = itertools.combinations(generation_approach, i)
#     for combination in combinations:
#         data = load_data(llms, dataset, list(combination), human)
#         filtered_data = filter_data(data)
#         fine_tune_BERT_with_kfold(filtered_data, 5)
#         # fine_tune_DeBERTa_with_kfold(filtered_data, 5)
#         print('###############################')

##############################################################################################
# print('Out of Domain Results')
# print('--------------------------------------------------------------')

# # Load datasets
# path = './data/'
# llms = ['llama2_70b', 'vicuna-v1.3_33b']
# dataset = ['coaid', 'gossipcop', 'politifact']
# generation_approach = ['open_ended_generation', 'paraphrase_generation', 'rewrite_generation']
# human = False

# data = load_data(llms, dataset, generation_approach, human)
# filtered_data = filter_data(data)

# # Generate combinations of different sizes, excluding the full set
# for r in range(1, len(generation_approach)):
#     for subset in combinations(generation_approach, r):
#         train_generation_approach = list(subset)
#         test_generation_approach = [x for x in generation_approach if x not in subset]
#         print('Train Generation Approach: {}'.format(train_generation_approach))
#         print('--------------------------------------------------------------')
#         print('Test Generation Approach: {}'.format(test_generation_approach))
#         print('--------------------------------------------------------------')
#         fine_tune_BERT_domain_test(filtered_data, train_generation_approach, test_generation_approach)
#         # fine_tune_DeBERTa_domain_test(filtered_data, train_generation_approach, test_generation_approach)
#         print('###############################')


