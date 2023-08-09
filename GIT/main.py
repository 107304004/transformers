import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CocoCaptions
from dataset import ImageCaptionDataset

from transformers import AutoProcessor
from transformers import AutoModelForCausalLM

from PIL import Image

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--run', default='exp', type=str)
parser.add_argument('--img', default=None, type=str)
args = parser.parse_args()


device = "cuda" if torch.cuda.is_available() else "cpu"


# use val coco dataset
val_coco_dataset = CocoCaptions(root="../data/COCO2017/val2017/", annFile="../data/COCO2017/annotations/captions_val2017.json")
print("num of val images:", len(val_coco_dataset))
processor = AutoProcessor.from_pretrained("microsoft/git-base", cache_dir="/tmp2/ttchen/.cache/")
val_dataset = ImageCaptionDataset(val_coco_dataset, processor)
val_dataloader = DataLoader(val_dataset, batch_size=3, shuffle=False)


# exp result
def run_exp(dataloader, model):
    with torch.no_grad:
        for idx, batch in enumerate(tqdm(dataloader)):
            original_caption = processor.batch_decode(batch["input_ids"][0], skip_special_tokens=True)[0]
            print(original_caption)

            generated_ids = model.generate(pixel_values=batch['pixel_values'].to(device), max_length=50)
            generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            print(generated_caption)


def inference(img_path, model):
    image = Image.open(img_path)
    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values=pixel_values, max_length=50)
    generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(generated_caption)


if __name__ == '__main__':
    model = AutoModelForCausalLM.from_pretrained("microsoft/git-base", cache_dir="/tmp2/ttchen/.cache/")
    model.to(device)
    model.load_state_dict(torch.load('./checkpoint/git-base-fine-tuned.ckpt'))

    if args.run == "exp":
        run_exp(val_dataloader, model)

    if args.run == "inference":
        inference(args.img_path, model)
