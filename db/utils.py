from sqlalchemy import create_engine
from dotenv import load_dotenv
import os

load_dotenv()

_engine = None

def get_engine():
    global _engine
    if _engine is None:
        db_url = os.getenv("POSTGRES_URL")
        if not db_url:
            raise RuntimeError("POSTGRES_URL not set in .env")
        _engine = create_engine(db_url)
    
    return _engine

# Purpose to create an engine that connects to the Postgres
# Get data base URL from .env
# If the engine has not been created, then create the engine based on the URL that has been pulled. 
# return the engine