from pydantic import BaseModel


class User(BaseModel):
    """회원가입 정보"""

    username: str
    password: str
    name: str
    age: int
    gender: str


class UserInDB(User):
    hashed_password: str


class Token(BaseModel):
    access_token: str
    token_type: str
