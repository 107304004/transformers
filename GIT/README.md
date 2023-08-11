GIT small exp.<br>
Compare git-base, git-base(fine-tuned), git-base-coco on COCO2017 dataset
```bash
./download.sh
python train.py
python main.py --run exp
```

Inference on a single image use git-base(fine-tuned) model
```bash
python main.py --run inference --img [img_path]
```
