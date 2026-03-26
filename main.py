from fastapi import FastAPI
from controllers import api_router

app = FastAPI()

app.include_router(api_router)