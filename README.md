# Quick start

# 1. Dataset

- [Rareplanes](https://www.cosmiqworks.org/rareplanes-public-user-guide/)
- Folder `synthetic/test`
- Download: `aws s3 cp --recursive s3://rareplanes-public/synthetic/test .`

# 2. Pretrained File

- [weights/weight.pth](https://drive.google.com/file/d/1xMWxeDJQvRCxavwPEQXrmf2hjZFehIYr/view)

# 3. Training

- `python train.py --epoch 5 --batch-size 1 --scale 0.5 --validation 10`

# 4. Predict

- `python predict.py --model checkpoints/weight.pth --input ./images/demo.png`
