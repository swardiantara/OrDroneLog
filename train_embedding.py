import pandas as pd
import torch
import os
import math
from torch.utils.data import DataLoader, Dataset
from sentence_transformers import SentenceTransformer, InputExample
from sentence_transformers.readers import InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from huggingface_hub import HfApi, hf_hub_download, repo_exists

from scipy.spatial.distance import cosine, euclidean

from utils.losses import OrdinalContrastiveLoss, ContrastiveLoss


def scale_l2_value(d, a=0.5, b=2, d_min=0, d_max=1.7320508075688772):
    """
    Scale a single L2 (Euclidean) distance value to a given range [a, b].

    Parameters:
    - d (float): The original L2 distance value.
    - d_min (float): The minimum L2 distance in the dataset.
    - d_max (float): The maximum L2 distance in the dataset.
    - a (float): The lower bound of the new range.
    - b (float): The upper bound of the new range.

    Returns:
    - float: The scaled L2 distance value.
    """
    if d_min == d_max:  # Prevent division by zero
        return a  # If all distances are the same, return the lower bound

    return a + ((d - d_min) * (b - a)) / (d_max - d_min)


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
df = pd.read_csv(os.path.join('dataset', 'merged-manual-unique.csv'))  # Assume the CSV has 'message' and 'cluster_id' columns

# Create pairs for contrastive learning
def create_pairs(df: pd.DataFrame, label_type = "nominal", distance_funct = euclidean, distance: bool=False):
    label2id = {
        'normal': 1,
        'low': 2,
        'medium': 3,
        'high': 4
    }
    label2vec = {
        'normal': [1,0,0,0],
        'low': [1,1,0,0],
        'medium': [1,1,1,0],
        'high': [1,1,1,1]
    }
    if label_type == 'nominal':
        df['labelidx'] = df['label'].map(label2id)
    elif label_type == 'vector':
        df['label_vector'] = df['label'].map(label2vec)
    examples = []
    for label in df['label'].unique():
        cluster_df = df[df['label'] == label]
        other_df = df[df['label'] != label]
        for i, row in cluster_df.iterrows():
            for j, other_row in cluster_df.iloc[i+1:].iterrows():
                if i != j:
                    pair_label = 1
                    if distance:
                        pair_label = 0
                    examples.append(InputExample(texts=[row['message'], other_row['message']], label=int(pair_label)))
            for j, other_row in other_df.iloc[i+1:].iterrows():
                pair_label = 0
                if distance:
                    if label_type == 'nominal':
                        pair_label = abs(row['labelidx'] - other_row['labelidx'])
                    elif label_type == 'vector':
                        pair_label = distance_funct(row['label_vector'], other_row['label_vector'])
                        pair_label = scale_l2_value(pair_label)
                examples.append(InputExample(texts=[row['message'], other_row['message']], label=int(pair_label)))
    return examples

examples = create_pairs(df, label_type='vector', distance_funct=euclidean, distance=True)
# Step 3: Create DataLoader
train_dataloader = DataLoader(examples, shuffle=True, batch_size=64)
# print(train_dataloader)

# Step 4: Define the contrastive loss
train_loss = OrdinalContrastiveLoss(model=model)

# Optional: Define evaluator for validation
evaluator = EmbeddingSimilarityEvaluator.from_input_examples(examples, name='severity-all')

# Step 5: Train the model
num_epochs = 3
warmup_steps = int(len(train_dataloader) * num_epochs * 0.1)
output_path = os.path.join('experiments', 'severity-all')
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    evaluator=evaluator,
    epochs=num_epochs,
    warmup_steps=warmup_steps,
    output_path=output_path
)

# Save the model
model.save(output_path, 'severity-all')

# push model to Huggingface
api = HfApi(token=os.getenv("HF_TOKEN"))
repo_name = f'drone-ordinal-all'
api.create_repo(repo_id=f'swardiantara/{repo_name}', exist_ok=True, repo_type="model")

api.upload_folder(
    folder_path=output_path,
    repo_id=f"swardiantara/{repo_name}",
    repo_type="model",
)
