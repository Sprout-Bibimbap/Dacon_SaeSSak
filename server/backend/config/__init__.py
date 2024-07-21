from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()

class CommonSettings(BaseSettings):
    OPENAI_API_KEY: str = ""


class ServerSettings(BaseSettings):
    HOST: str = ""
    PORT: int = 8000


class DatabaseSettings(BaseSettings):
    MONGO_USERNAME: str = ""
    MONGO_PASSWORD: str = ""
    MONGO_HOST: str = ""
    MONGO_CLUSTER_NAME: str = ""
    MONGO_OPTIONS: str = ""
    
    DB_RESOURCE: str = "Resource"
    COLLECTION_REFERENCES: str = "References"
    COLLECTION_ERROR_LOG: str = "ErrorLog"
    COLLECTION_INFO_LOG: str = "InfoLog"

    DB_USER: str = "User"
    COLLECTION_USER_CONVERSATION: str = "UserConv"
    COLLECTION_USER_INFORMATION: str = "UserInfo"
    COLLECTION_USER_REPORT: str = "UserReport"

    @property
    def MONGO_URL(self) -> str:
        return f"mongodb+srv://{self.MONGO_USERNAME}:{self.MONGO_PASSWORD}@{self.MONGO_HOST}/{self.MONGO_CLUSTER_NAME}?{self.MONGO_OPTIONS}"

    class Config:
        env_file = ".env"

class LoginSettings(BaseSettings):
    jwt_secret_key: str
    
    class Config:
        env_file = ".env"

class Settings(CommonSettings, ServerSettings, DatabaseSettings, LoginSettings):
    pass


settings = Settings()