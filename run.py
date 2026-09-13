"""OnQ 서버 실행: python run.py"""
from app.main import app
import uvicorn
import config

if __name__ == "__main__":
    uvicorn.run(app, host=config.HOST, port=config.PORT, log_level="info")
