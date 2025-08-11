import os 
from dotenv import load_dotenv

from sqlalchemy import (
    create_engine, Column, Integer, String, Text, DateTime, func
)

from sqlalchemy.orm import declarative_base

load_dotenv()


Base = declarative_base()


class Feedback(Base):
    __tablename__ = 'feedback'

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    model_version = Column(String(50))
    prompt = Column(Text, nullable=False)
    response = Column(Text, nullable=False)
    feedback_type = Column(String(10), nullable=False)
    correction_text = Column(Text)
    status = Column(String(20), default='new')


    def __repr__(self):
        return f"<Feedback(id={self.id}, prompt='{self.prompt[:30]}...')>"
    
def get_engine():
    DATABASE_URL = os.getenv("DATABASE_URL")
    if DATABASE_URL is None:
        raise ValueError("DATABSE_URL environment variable is not set.")
        
    engine = create_engine(DATABASE_URL)
    return engine


def create_db_tables():
    try:
        engine = get_engine()
        print("Database engine successfully created")

        Base.metadata.create_all(engine)
        print("Table 'feedback' successfully created or already existed before.")
    except Exception as e:
        print(f"Issue in creating table: {e}")


if __name__ == "__main__":
    create_db_tables()