import torch
from PIL import Image
import clip
import os
from tqdm import tqdm
from datasets import load_dataset

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Load Instruct 150k dataset from local JSON file
dataset = load_dataset('json', data_files='llava_instruct_150k.json', split='train')

# Process images and store embeddings
embeddings = {}
for item in tqdm(dataset):
    image_path = os.path.join('train2017', item['image'])  # Add 'train2017' to the path
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    
    with torch.no_grad():
        image_features = model.encode_image(image)
    
    embeddings[item['id']] = image_features.cpu().numpy()

# Save embeddings
torch.save(embeddings, 'clip_embeddings.pt')
