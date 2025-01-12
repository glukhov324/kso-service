Link to the original Changeformer repo: https://github.com/wgcban/ChangeFormer?tab=readme-ov-file

Download folders checkpoints and classification_models by link https://drive.google.com/drive/folders/1H61bM1Q2QagrBMJiPrk0qrGBIBUZs3MZ?usp=sharing

put folders checkpoints and classification_models to src

# __Installation & start guide__

1. docker build -t kso:v1 .
2. docker run -p 8000:8000 --gpus=all kso:v1

after previous step you can test my project with test images from folder test_images by link 

http://127.0.0.1:8000/  - windows

http://0.0.0.0:8000/    - linux

you can check docs of my API by link:

http://127.0.0.1:8000/docs  - windows

http://0.0.0.0:8000/docs    - linux