import pandas as pd
from sentence_transformers import SentenceTransformer, util

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

# Load datasets
path = './data/'
llms = ['gpt-3.5-turbo', 'llama2_70b', 'vicuna-v1.3_33b']
dataset = ['coaid', 'gossipcop', 'politifact']
generation_approach = ['open_ended_generation', 'paraphrase_generation', 'rewrite_generation']
human = False

#llms = ['gpt-3.5-turbo', 'llama2_7b', 'llama2_13b', 'llama2_70b', 'vicuna-v1.3_7b', 'vicuna-v1.3_13b', 'vicuna-v1.3_33b']
#generation_approach = ['open_ended_generation', 'paraphrase_generation', 'rewrite_generation']

df = load_data(llms, dataset, generation_approach, human)

# Load a pre-trained model
model = SentenceTransformer('all-MiniLM-L6-v2')

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

# Apply the function to each row
df['similarity'] = df.apply(conditional_similarity, axis=1)

# Compute mean similarity
# mean_similarity = df['similarity'].mean()

# print(df)
# print(f'Mean Similarity: {mean_similarity}')

grouped_means = df.groupby(['generated_by', 'generation_approach'])['similarity'].mean().reset_index()
print(grouped_means)
print('--------------------------------------------------------------')

# Merge the mean similarity scores back into the original DataFrame
df_with_means = pd.merge(df, grouped_means, on=['generated_by', 'generation_approach'], suffixes=('', '_mean'))

# Filter rows where the similarity is greater than or equal to the group mean
filtered_df = df_with_means[df_with_means['similarity'] >= df_with_means['similarity_mean']]

# Now, filtered_df contains only the rows where the similarity score 
# is equal to or higher than the mean for its 'generated_by' and 'generation_approach' group.
print('--------------------------------------------------------------')
print(filtered_df['generated_by'].value_counts())
print('--------------------------------------------------------------')
print(filtered_df['generation_approach'].value_counts())
print('--------------------------------------------------------------')