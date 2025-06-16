import os
import logging
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("git").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)
logger = logging.getLogger("gitreviewer")

DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gemini-2.5-flash-preview-05-20")

class CommitMessage(BaseModel):
    message: str
    details: list[str]