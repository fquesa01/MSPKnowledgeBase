"""Application configuration."""
from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    # App
    app_name: str = "MSP Knowledge Base"
    debug: bool = False
    
    # Auth
    secret_key: str = "CHANGE_ME_IN_PRODUCTION"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 60 * 24 * 7  # 1 week
    
    # Database
    database_url: str = "sqlite+aiosqlite:///./data/msp_kb.db"
    
    @property
    def async_database_url(self) -> str:
        url = self.database_url
        if url.startswith("postgresql://"):
            url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
            if "sslmode=" in url:
                url = url.replace("sslmode=require", "ssl=require")
                url = url.replace("sslmode=disable", "ssl=disable")
                url = url.replace("sslmode=prefer", "ssl=prefer")
        return url
    
    # OpenAI
    openai_api_key: str = ""
    openai_model: str = "gpt-4o"
    
    # Storage
    upload_dir: str = "./data/uploads"
    index_dir: str = "./data/indexes"
    
    class Config:
        env_file = ".env"

@lru_cache()
def get_settings():
    return Settings()
