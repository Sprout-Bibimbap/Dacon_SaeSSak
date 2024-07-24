import asyncio
from fastapi import HTTPException, status
from pymongo.errors import ConnectionFailure
from motor.motor_asyncio import AsyncIOMotorClient

from config import settings


class MongoDBClient:
    _instance = None

    @classmethod
    async def get_instance(cls):
        """싱글톤 구조 구현"""
        if cls._instance is None:
            cls._instance = await cls._create_instance()
            # await cls.ensure_connection(cls._instance)
        return cls._instance

    @classmethod
    async def _create_instance(cls):
        """MongoDB client 생성 및 초기 연결 검증"""
        try:
            client = AsyncIOMotorClient(
                settings.MONGO_URL,
                serverSelectionTimeoutMS=5000,
                maxPoolSize=20,
                minPoolSize=5,
                retryWrites=True,
                connectTimeoutMS=10000,
            )
            return client

        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"An unexpected error occurred: {str(e)}",
            )

    @staticmethod
    async def ensure_connection(client):
        max_retries = 5
        retry_delay = 1
        for attempt in range(max_retries):
            try:
                # Ping the database to check the connection
                await client.admin.command("ping")
                print("Successfully connected to MongoDB")
                return
            except ConnectionFailure as e:
                if attempt < max_retries - 1:
                    print(
                        f"Connection attempt {attempt + 1} failed. Retrying in {retry_delay} seconds..."
                    )
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail=f"Could not connect to MongoDB after {max_retries} attempts: {str(e)}",
                    )
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"An unexpected error occurred during connection: {str(e)}",
                )
