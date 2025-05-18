from pydantic_settings import BaseSettings, SettingsConfigDict

class Config(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=True)
    
    groq_api_key:str
    
ProjectConfig = Config()
    