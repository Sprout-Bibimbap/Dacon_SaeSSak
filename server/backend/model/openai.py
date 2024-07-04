from openai import OpenAI
from config.utils import config
from fastapi import HTTPException, status


class OpenAIClient:
    _instance = None

    @classmethod
    async def get_instance(cls):
        if cls._instance is None:
            cls._instance = await cls._create_instance()
        return cls._instance

    @classmethod
    async def _create_instance(cls):
        try:
            client = OpenAI(api_key=config["API"]["OPENAI_API_KEY"])

            return client
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"An unexpected error occurred: {str(e)}",
            )
