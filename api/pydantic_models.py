from pydantic import BaseModel, EmailStr

class UserCreate(BaseModel):
    name: str
    password: str
    email: EmailStr

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class QueryRequest(BaseModel):
    query: str
    top_k: int = 2

class AssignTopic(BaseModel):
    query: str

class UserInDB(BaseModel):
    user_id: str
    user_name: str
    email: str

class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    username: str | None = None

class AssignTopic(BaseModel):
    conversation_name: str
    conversation_description: str