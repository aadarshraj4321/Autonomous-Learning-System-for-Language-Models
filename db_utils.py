from database_setup import Feedback, get_engine
from sqlalchemy.orm import sessionmaker


SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=get_engine())



def log_feedback_to_db(model_version, prompt, response, feedback_type, correction_text=None):
    db_session = SessionLocal()
    try:
        new_feedback = Feedback(
            model_version = model_version,
            prompt = prompt,
            response = response,
            feedback_type = feedback_type,
            correction_text = correction_text
        )

        db_session.add(new_feedback)

        db_session.commit()

        print(f"Feedback successfully logged: {feedback_type}")
    
    except Exception as e:
        print(f"Feedback loged Error: {e}")
        db_session.rollback()
    
    finally:
        db_session.close()



def get_all_feedback_from_db():
    db_session = SessionLocal()
    try:
        feedback_list = db_session.query(Feedback).all()
        return feedback_list
    finally:
        db_session.close()
    
