Link to the original Changeformer repo: https://github.com/wgcban/ChangeFormer?tab=readme-ov-file

Download folders checkpoints and classification_models by link https://drive.google.com/drive/folders/1H61bM1Q2QagrBMJiPrk0qrGBIBUZs3MZ?usp=sharing

put folders checkpoints and classification_models to src

# __Fast start with docker__

1. ```sh
   docker build -t kso:v1 .
   ```
2. ```sh
   docker run -p 8000:8000 --gpus=all kso:v1
   ```

after previous step you can test my project with test images from folder test_images by link 

http://0.0.0.0:8000/

you can check docs of my API by link:

http://0.0.0.0:8000/docs