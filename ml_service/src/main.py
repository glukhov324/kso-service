from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api import router
from src.settings import settings



app = FastAPI(title=settings.APP_TITLE,
              description=settings.APP_DESCRIPTION,
              version=settings.APP_VERSION)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router,
                   tags=["Predict"])