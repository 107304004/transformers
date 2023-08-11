import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CocoCaptions
from dataset import ImageCaptionDataset
from torchmetrics.text import BLEUScore

from transformers import AutoProcessor
from transformers import AutoModelForCausalLM
# import evaluate # hugging face evaluate

from tqdm import tqdm
from PIL import Image

import argparse

batch_size=20
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
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


# exp result
def run_exp(dataloader, model):
    # evaluation metrics
    bleu_metric = BLEUScore()

    with torch.no_grad():
        for idx, batch in enumerate(tqdm(dataloader)):
            # original_caption = processor.batch_decode(batch["input_ids"], skip_special_tokens=True)
            original_captions = []
            for i in range(idx*batch_size, (idx+1)*batch_size):
                original_captions.append(val_coco_dataset[i][1])
            # print("original:", original_captions) # a list of batch_size sentences

            generated_ids = model.generate(pixel_values=batch['pixel_values'].to(device), max_length=50)
            generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)
            # print("generated:", generated_caption) # a list of batch_size sentences

            # 1 pred vs multiple references
            bleu_metric.update(generated_caption, original_captions)

    print("BLEU score:", bleu_metric.compute())


def inference(img_path, model):
    image = Image.open(img_path)
    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values=pixel_values, max_length=50)
    generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(generated_caption)


if __name__ == '__main__':
    # 3 models
    model_base = AutoModelForCausalLM.from_pretrained("microsoft/git-base", cache_dir="/tmp2/ttchen/.cache/")

    model_finetuned = AutoModelForCausalLM.from_pretrained("microsoft/git-base", cache_dir="/tmp2/ttchen/.cache/")
    model_finetuned.load_state_dict(torch.load('./checkpoint/git-base-fine-tuned-5epochs.ckpt'))

    model_coco = AutoModelForCausalLM.from_pretrained("microsoft/git-base-coco", cache_dir="/tmp2/ttchen/.cache/")


    if args.run == "exp":
        print("model_base:")
        run_exp(val_dataloader, model_base.to(device))

        print("model_finetuned:")
        run_exp(val_dataloader, model_finetuned.to(device))

        print("model_coco:")
        run_exp(val_dataloader, model_coco.to(device))

    if args.run == "inference":
        inference(args.img, model_finetuned)
