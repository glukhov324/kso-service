Link to the original Changeformer repo: https://github.com/wgcban/ChangeFormer?tab=readme-ov-file

Download folders checkpoints and classification_models by link https://drive.google.com/drive/folders/1H61bM1Q2QagrBMJiPrk0qrGBIBUZs3MZ?usp=sharing

put folders checkpoints and classification_models to src

Installation & start guide

Windows:

1. py -3.10 -m venv venv 
2. .\venv\Scripts\activate
3. python -m pip install pip --upgrade
4. pip install -r .\requirements.txt

Linux/macOS:

1. python3.10 -m venv venv
2. source venv/bin/activate
3. python -m pip install pip --upgrade
4. pip install -r requirements.txt

to start uvicorn server: python -m uvicorn app:app --reload