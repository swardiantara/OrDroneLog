import pandas as pd
import torch
import os
import math
from torch.utils.data import DataLoader, Dataset
from sentence_transformers import SentenceTransformer, InputExample
from sentence_transformers.readers import InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator

from utils.losses import OrdinalContrastiveLoss
# Check if GPU is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

# Step 1: Load a pre-trained model
model_name = 'all-mpnet-base-v2'  # or 'hkunlp/instructor-xl'
model = SentenceTransformer(model_name).to(device)

# Step 2: Prepare the dataset
class DroneLogsDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        return InputExample(texts=[row['message']], label=row['cluster_id'])

# Load your dataset
df = pd.read_csv(os.path.join('dataset', 'filtered_train.csv'))  # Assume the CSV has 'message' and 'cluster_id' columns

# Create pairs for contrastive learning
def create_pairs(df: pd.DataFrame, distance: bool=False):
    label2id = {
        'normal': 1,
        'low': 2,
        'medium': 3,
        'high': 4
    }
    df['labelidx'] = df['label'].map(label2id)
    examples = []
    for label in df['labelidx'].unique():
        cluster_df = df[df['labelidx'] == label]
        other_df = df[df['labelidx'] != label]
        for i, row in cluster_df.iterrows():
            for j, other_row in cluster_df.iterrows():
                if i != j:
                    pair_label = 1
                    if distance:
                        pair_label = 0
                    examples.append(InputExample(texts=[row['message'], other_row['message']], label=int(pair_label)))
            for j, other_row in other_df.iterrows():
                pair_label = 0
                if distance:
                    pair_label = abs(row['labelidx'] - other_row['labelidx'])
                examples.append(InputExample(texts=[row['message'], other_row['message']], label=int(pair_label)))
    return examples

examples = create_pairs(df, distance=True)
# Step 3: Create DataLoader
train_dataloader = DataLoader(examples, shuffle=True, batch_size=64)
# print(train_dataloader)

# Step 4: Define the contrastive loss
train_loss = OrdinalContrastiveLoss(model=model)

# Optional: Define evaluator for validation
evaluator = EmbeddingSimilarityEvaluator.from_input_examples(examples, name='drone-log-eval')

# Step 5: Train the model
num_epochs = 3
warmup_steps = int(len(train_dataloader) * num_epochs * 0.1)
output_path = os.path.join('experiments', 'embeddings')
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    evaluator=evaluator,
    epochs=num_epochs,
    warmup_steps=warmup_steps,
    output_path=output_path
)

# Save the model
model.save(output_path, 'drone_log_semantic')
