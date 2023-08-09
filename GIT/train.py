import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CocoCaptions
from dataset import ImageCaptionDataset

from transformers import AutoProcessor
from transformers import AutoModelForCausalLM

from tqdm import tqdm


# set hyper parameters and device
epochs = 10
batch_size = 32 # 512 cause CUDA out of memory
lr = 2.5e-6

device = "cuda" if torch.cuda.is_available() else "cpu"


#############################################
# Deifine COCO dataloader for image caption #
#############################################

train_coco_dataset = CocoCaptions(root="../data/COCO2017/train2017/", annFile="../data/COCO2017/annotations/captions_train2017.json")
val_coco_dataset = CocoCaptions(root="../data/COCO2017/val2017/", annFile="../data/COCO2017/annotations/captions_val2017.json")
print("num of train images:", len(train_coco_dataset))
print("num of val images:", len(val_coco_dataset))
# image, target = val_coco_dataset[0]
# image is a PIL image
# target is a list of sentences
processor = AutoProcessor.from_pretrained("microsoft/git-base", cache_dir="/tmp2/ttchen/.cache/")


train_dataset = ImageCaptionDataset(train_coco_dataset, processor)
val_dataset = ImageCaptionDataset(val_coco_dataset, processor)
# item = dataset[0]
# for k, v in item.items():
#     print(k, v.shape)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)


#############################
# train and val for 1 epoch #
#############################

def train_epoch(dataloader, model):
    train_loss = 0

    for idx, batch in enumerate(tqdm(dataloader)):

        # input_ids, attention_mask, pixel_values
        outputs = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            pixel_values=batch["pixel_values"].to(device),
            labels=batch["input_ids"].to(device)
        )

        loss = outputs.loss
        train_loss += loss.item()
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

    return model, train_loss


def val(dataloader, model):
    val_loss = 0

    with torch.no_grad():
        for idx, batch in enumerate(tqdm(dataloader)):

            # input_ids, attention_mask, pixel_values
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                pixel_values=batch["pixel_values"].to(device),
                labels=batch["input_ids"].to(device)
            )

            loss = outputs.loss
            val_loss += loss.item()

    return val_loss


############################
# Fine-tune git-base model #
############################

model = AutoModelForCausalLM.from_pretrained("microsoft/git-base", cache_dir="/tmp2/ttchen/.cache/")
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

for epoch in range(epochs):
    print("="*40)
    print("Epoch:", epoch+1)

    print("Start training...")
    model, train_loss = train_epoch(train_dataloader, model)
    print("Train Loss:", train_loss)

    print("Start validation...")
    val_loss = val(val_dataloader, model)
    print("Val Loss:", val_loss)
print("Finish")

# save model
torch.save(model.state_dict(), './checkpoint/git-base-fine-tuned.ckpt')
