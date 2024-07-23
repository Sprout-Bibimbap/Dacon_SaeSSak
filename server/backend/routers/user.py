from fastapi import HTTPException, Depends, APIRouter, Request
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta
from typing import Optional
from config import settings
from config.schemas import *
from bson import ObjectId

router = APIRouter()

# 비밀번호 해싱을 위한 설정
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT 설정
SECRET_KEY = settings.jwt_secret_key
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# OAuth2 설정
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


# 토큰 생성 함수
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


# 사용자 인증 함수
async def authenticate_user(request: Request, username: str, password: str):
    db = request.app.state.user_info

    # 비밀번호 해싱 및 검증 함수
    def verify_password(plain_password, hashed_password):
        return pwd_context.verify(plain_password, hashed_password)

    user = await db.find_one({"username": username})
    if not user:
        return False
    if not verify_password(password, user["hashed_password"]):
        return False
    return user


@router.post("/signup")
async def signup(request: Request, user: User):
    """회원 가입"""
    db = request.app.state.user_info

    def get_password_hash(password):
        return pwd_context.hash(password)

    # 사용자 중복 확인
    db_user = await db.find_one({"username": user.username})
    if db_user:
        raise HTTPException(status_code=400, detail="Username already registered")

    # 비밀번호 해싱
    hashed_password = get_password_hash(user.password)

    # 사용자 데이터
    user_dict = user.dict()
    user_dict["hashed_password"] = hashed_password
    del user_dict["password"]

    # 사용자 추가
    result = await db.insert_one(user_dict)

    # 삽입 결과 확인
    if result.inserted_id:
        return {
            "message": "User created successfully",
            "user_id": str(result.inserted_id),
        }
    else:
        raise HTTPException(status_code=500, detail="Failed to create user")


@router.post("/login", response_model=Token)
async def login(request: Request, form_data: OAuth2PasswordRequestForm = Depends()):
    """로그인"""
    user = await authenticate_user(request, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user["username"]}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}


# 현재 사용자 정보 가져오기
async def get_current_user(request: Request, token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=401,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    db = request.app.state.user_info
    user = await db.find_one({"username": username})
    if user is None:
        raise credentials_exception
    return user


def convert_objectid(data):
    if isinstance(data, list):
        return [convert_objectid(item) for item in data]
    elif isinstance(data, dict):
        return {key: convert_objectid(value) for key, value in data.items()}
    elif isinstance(data, ObjectId):
        return str(data)
    else:
        return data


@router.get("/me")
async def read_users_me(
    request: Request, current_user: User = Depends(get_current_user)
):
    return convert_objectid(current_user)
