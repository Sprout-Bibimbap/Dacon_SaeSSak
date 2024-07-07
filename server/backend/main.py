from fastapi import FastAPI, File
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn
from middlewares.time import StartTimeMiddleware
from database.mongodb import MongoDBClient
from exceptions.loggers import MongoLogger
from model.openai import OpenAIClient
from routers import stt, tts
from config import settings 


@asynccontextmanager
async def app_lifespan(app: FastAPI):
    try:
        print("Lifespan start!")
        app.state.mongo_client = await MongoDBClient.get_instance()                           # MongoDB client 
        app.state.db_user = app.state.mongo_client[settings.DB_USER]                          # 유저 관련 Database
        app.state.db_resource = app.state.mongo_client[settings.DB_RESOURCE]                  # 리소스 관련 Database
        
        app.state.user_conv = app.state.db_user[settings.COLLECTION_USER_CONVERSATION]        # 유저 대화 collection
        app.state.user_info = app.state.db_user[settings.COLLECTION_USER_INFORMATION]         # 유저 정보 collection
        app.state.user_report = app.state.db_user[settings.COLLECTION_USER_REPORT]            # 유저 리포트 collection
        
        app.state.resource_reference = app.state.db_resource[settings.COLLECTION_REFERENCES]  # 리소스 레퍼런스 collection
        app.state.resource_error_log = app.state.db_resource[settings.COLLECTION_ERROR_LOG]   # Info log collection 
        app.state.resource_info_log = app.state.db_resource[settings.COLLECTION_INFO_LOG]     # Error log collection
        print("MongoDB done!")
        
        # await MongoLogger.initialize()
        # print("Logger done!")
        
        app.state.openai_client = await OpenAIClient.get_instance()
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
app.include_router(stt.router, prefix="/api/v1/response")
app.include_router(tts.router, prefix="/api/v1/response")


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
