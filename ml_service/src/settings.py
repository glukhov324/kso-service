from pydantic_settings import BaseSettings, SettingsConfigDict
import torch



class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")

    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEV_MODE: bool = False
    APP_TITLE: str
    APP_DESCRIPTION: str
    APP_VERSION: str


    # Models
    CD_MODEL_WTS_PATH: str
    ONION_CLF_WTS_PATH: str
    POTATO_CLF_WTS_PATH: str
    CABBAGE_CLF_WTS_PATH: str
    POLLOCK_CLF_WTS_PATH: str
    BEET_CLF_WTS_PATH: str
    CLF_CONFIDENCE_THRESHOLD: float
    DEVICE: torch.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Mask post processing
    DILATE_ITER: int
    
    
settings = Settings()