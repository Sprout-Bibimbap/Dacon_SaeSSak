from motor.motor_asyncio import AsyncIOMotorClient
from config import settings 
from fastapi import HTTPException, status


class MongoDBClient:
    _instance = None

    @classmethod
    async def get_instance(cls):
        """싱글톤 구조 구현"""
        if cls._instance is None:
            cls._instance = await cls._create_instance()
        return cls._instance

    @classmethod
    async def _create_instance(cls):
        """MongoDB client 생성 및 초기 연결 검증"""
        try:
            client = AsyncIOMotorClient(
                settings.MONGO_URL,
                serverSelectionTimeoutMS=5,
                maxPoolSize=20,
                retryWrites=True,
            )

            return client

        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"An unexpected error occurred: {str(e)}",
            )


async def insert_data(
    client: AsyncIOMotorClient, db_name: str, collection_name: str, document: dict
):
    """MongoDB의 특정 db, collection에 데이터를 넣는 함수"""
    try:
        db = client[db_name]
        collection = db[collection_name]

        result = await collection.insert_one(document)
        return result.inserted_id
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {str(e)}",
        )
