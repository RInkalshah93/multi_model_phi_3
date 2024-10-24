import torch
from torch.utils.data import Dataset
import random

class ImageConversationDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_name = item['id']
        conversations = item['conversations']

        # Randomly select an even-numbered index
        max_even_index = len(conversations) - 1
        selected_index = random.randrange(0, max_even_index + 1, 2)

        # Get the human question and GPT response
        human_msg = conversations[selected_index]['value']
        ai_msg = conversations[selected_index + 1]['value']

        input_ids = self.tokenizer.encode(human_msg, max_length=self.max_length, truncation=True, padding='max_length', return_tensors='pt')
        target_ids = self.tokenizer.encode(ai_msg, max_length=self.max_length, truncation=True, padding='max_length', return_tensors='pt')
        
        input_ids = input_ids.squeeze(0)
        target_ids = target_ids.squeeze(0)
        
        return image_name, input_ids, target_ids

