from fastapi import FastAPI, File
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from openai import OpenAI
import os
import uvicorn
from middlewares.time import StartTimeMiddleware
from database.mongodb import MongoDBClient
from exceptions.loggers import MongoLogger
from routers import stt
from config.utils import config


@asynccontextmanager
async def app_lifespan(app: FastAPI):
    try:
        print("Lifespan start!")
        # app.state.mongo_client = await MongoDBClient.get_instance()
        # print("MongoDB done!")
        # await MongoLogger.initialize()
        # print("Logger done!")
        app.state.openai_client = OpenAI(api_key=config["API"]["OPENAI_API_KEY"])
        print("OpenAI client done")
        print("All Set!")

        yield
    except Exception as e:
        print(f"Error during lifespan: {e}")
    finally:
        print("Closing MongoDB connection...")
        if hasattr(app.state, "mongo_client"):
            await app.state.mongo_client.close()
        print("All Close!")


app = FastAPI(lifespan=app_lifespan)


# Middleware 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(StartTimeMiddleware)

# router 추가
app.include_router(stt.router, prefix="/api/v1/stt")


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
