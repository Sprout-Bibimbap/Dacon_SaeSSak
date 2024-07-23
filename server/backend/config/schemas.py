from pydantic import BaseModel


class User(BaseModel):
    """회원가입 정보"""

    username: str
    name: str
    password: str
    age: int
    gender: str


class Token(BaseModel):
    access_token: str
    token_type: str
