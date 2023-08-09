import torch
from torch.utils.data import Dataset, DataLoader


class ImageCaptionDataset(Dataset):
    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, target = self.dataset[idx]

        # take the first sentence of target to form the img_text pair
        # encoding contains 3 keys: input_ids, attentio_mask, pixel_values
        encoding = self.processor(images=image, text=target[0], padding="max_length", return_tensors="pt")

        # remove batch dimension: [1, 3, 224, 224] -> [3, 224, 224]
        encoding = {k:v.squeeze() for k,v in encoding.items()}

        return encoding
