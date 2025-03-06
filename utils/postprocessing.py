import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

class_names = ['normal', 'low', 'medium', 'high']

def forward_infer(label_vector):
    # def infer_class_sequential(label_vector, class_names):
    """
    Infers the predicted class based on the first zero encountered. Cheng et al., (2009)
    
    Args:
    - label_vector (list): A binary list (e.g., [1, 1, 0, 0])
    - class_names (list): List of class names in order (e.g., ['normal', 'low', 'medium', 'high'])
    
    Returns:
    - str: Predicted class name
    """
    for i in range(len(label_vector)):
        if label_vector[i] == 0:
            return class_names[i - 1] if i > 0 else 'normal'  # Handle case where first element is 0
    return class_names[-1]  # If no zero is found, return the last class


def backward_infer(label_vector):
    # def infer_class_severity(label_vector, class_names):
    """
    Infers the predicted class based on the last 1 encountered (severity-oriented decoding).
    
    Args:
    - label_vector (list): A binary list (e.g., [1, 1, 0, 0])
    - class_names (list): List of class names in order (e.g., ['normal', 'low', 'medium', 'high'])
    
    Returns:
    - str: Predicted class name
    """
    for i in range(len(label_vector) - 1, -1, -1):  # Iterate from last to first
        if label_vector[i] == 1:
            return class_names[i]
    return 'normal'  # If all are zero, return None (or 'normal' if that's the default)


def label2class(label):
    if label[0] == 1:
        return "high"
    else:
        if label[2] == 1:
            return "medium"
        elif label[1] == 1:
            return "low"
        else:
            return "normal"
        
        
def visualize_projection(dataset_loader, idx2label, model, device, output_dir):
    lower2capital = {
        'normal': 'Normal',
        'low': 'Low',
        'medium': 'Medium',
        'high': 'High'
    }

    all_labels_multiclass = []
    all_embeddings = []
    with torch.no_grad():
        for batch in dataset_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels_index = batch["labels_index"]
    
            embeddings, _, _ = model(input_ids, attention_mask)
            all_labels_multiclass.extend(labels_index.cpu().numpy())
            all_embeddings.append(embeddings)
    
    tsne = TSNE(n_components=2, random_state=42)
    all_embeddings = torch.cat(all_embeddings, dim=0)
    reduced_embeddings = tsne.fit_transform(all_embeddings.cpu().numpy())
    label_decoded = [idx2label.get(key) for key in all_labels_multiclass]
    # label_encoder_multi.inverse_transform(all_labels_multiclass)
    label_df = pd.DataFrame()
    label_df["label"] = list(label_decoded)
    label_df["label"] = label_df["label"].map(lower2capital)
    labels = label_df['label'].tolist()
    
    plt.figure(figsize=(5, 2.5))
    fig, ax = plt.subplots()

    unique_labels = ['Normal', 'Low', 'Medium', 'High']
    colors = ['#4CAF50', '#FFC107', '#FF5722', '#D32F2F']
    
    counter = 0
    for label in unique_labels:
        # Filter data points for each unique label
        x_filtered = [reduced_embeddings[i][0] for i in range(len(reduced_embeddings)) if labels[i] == label]
        y_filtered = [reduced_embeddings[i][1] for i in range(len(reduced_embeddings)) if labels[i] == label]
        ax.scatter(x_filtered, y_filtered, label=label, s=15, c=colors[counter])
        counter+=1

    # Add a legend with only unique labels
    ax.set_xticks([])
    ax.set_yticks([])
    # legend = ax.legend(loc='lower right')
    plt.legend([]).set_visible(False)
    # Display the plot
    plt.savefig(os.path.join(output_dir, "dataset_viz.pdf"), bbox_inches='tight')
    plt.close()
    
    
def visualize_prediction(test_loader, model, device, output_dir, prediction_df):
    lower2capital = {
        'normal': 'Normal',
        'low': 'Low',
        'medium': 'Medium',
        'high': 'High'
    }
    # Prepare the dataframe
    if 'verdict' not in prediction_df.columns:
        prediction_df['verdict'] = [label == pred for label, pred in zip(prediction_df['label'], prediction_df['pred'])]
    prediction_df["label"] = prediction_df["label"].map(lower2capital)
    prediction_df["pred"] = prediction_df["pred"].map(lower2capital)
    prediction_df_labels = prediction_df['pred'].tolist()
    verdict_list = prediction_df['verdict'].tolist()

    # Obtain the hidden state
    # test_labels_multiclass = []
    test_embeddings = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            # labels_index = batch["labels_index"]
    
            embeddings, _, _ = model(input_ids, attention_mask)
            # test_labels_multiclass.extend(labels_index)
            test_embeddings.append(embeddings)
    
    # Dimensionality reduction
    tsne = TSNE(n_components=2, random_state=42)        
    test_embeddings = torch.cat(test_embeddings, dim=0)
    reduced_test_embeddings = tsne.fit_transform(test_embeddings.cpu().numpy())
    
    # Plot the data points
    plt.figure(figsize=(5, 2.5))
    fig, ax = plt.subplots() 
    # String labels
    # classes = ['High', 'Normal', 'Low', 'Medium']
    classes = ['Normal', 'Low', 'Medium', 'High']
    markers = ['o', '^', 's', 'D']
    colors = ['#4CAF50', '#FFC107', '#FF5722', '#D32F2F'] 
    # markers = ['D', 'o', '^', 's']
    # colors = ['#D32F2F', '#4CAF50', '#FFC107', '#FF5722']   
    for label in classes:
        false_df = prediction_df[prediction_df['verdict'] == False]
        class_df = false_df[false_df['pred'] == label]
        if len(class_df) < 1:
            continue
        x_filtered = [reduced_test_embeddings[i][0] for i in range(len(reduced_test_embeddings)) if ((prediction_df_labels[i] == label) & (verdict_list[i] == False))]
        y_filtered = [reduced_test_embeddings[i][1] for i in range(len(reduced_test_embeddings)) if ((prediction_df_labels[i] == label) & (verdict_list[i] == False))]
        pred_colors = [colors[classes.index(pred_label)] for pred_label in class_df['label']]
        marker_idx = classes.index(label)
        ax.scatter(x_filtered, y_filtered, c=pred_colors, marker=markers[marker_idx], label=label)
                
    legend = ax.legend()
    handles = legend.legend_handles
    for i, handle in enumerate(handles):
        handle.set_color(colors[i])
        
    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig(os.path.join(output_dir, "viz_prediction_tcolor_pshape.pdf"), bbox_inches='tight')
    plt.close()
