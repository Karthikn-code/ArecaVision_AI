import sqlite3
import os
from datetime import datetime
import pandas as pd
from config.config import DB_PATH
from utils.logger import get_logger

logger = get_logger("DatabaseManager")

def get_db_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    logger.info("Initializing database...")
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Create prediction_history table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS prediction_history (
            prediction_id TEXT PRIMARY KEY,
            image_path TEXT NOT NULL,
            predicted_class TEXT NOT NULL,
            confidence REAL NOT NULL,
            date TEXT NOT NULL,
            time TEXT NOT NULL,
            processing_time REAL NOT NULL,
            model_used TEXT NOT NULL,
            spots_detected INTEGER DEFAULT 0
        )
    ''')
    # Auto-migration if spots_detected column missing
    cursor.execute("PRAGMA table_info(prediction_history)")
    columns = [row[1] for row in cursor.fetchall()]
    if "spots_detected" not in columns:
        try:
            cursor.execute("ALTER TABLE prediction_history ADD COLUMN spots_detected INTEGER DEFAULT 0")
        except Exception:
            pass
    conn.commit()
    conn.close()
    logger.info("Database initialized successfully.")

def save_prediction(prediction_id, image_path, predicted_class, confidence, processing_time, model_used, spots_detected=0):
    logger.info(f"Saving prediction {prediction_id} to database...")
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")
    
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute('''
            INSERT INTO prediction_history 
            (prediction_id, image_path, predicted_class, confidence, date, time, processing_time, model_used, spots_detected)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (prediction_id, image_path, predicted_class, confidence, date_str, time_str, processing_time, model_used, spots_detected))
        conn.commit()
        logger.info(f"Prediction {prediction_id} saved successfully.")
    except Exception as e:
        logger.error(f"Error saving prediction to database: {e}")
        conn.rollback()
        raise e
    finally:
        conn.close()

def get_history_df():
    logger.info("Retrieving prediction history as DataFrame...")
    conn = get_db_connection()
    try:
        df = pd.read_sql_query("SELECT * FROM prediction_history ORDER BY date DESC, time DESC", conn)
        return df
    except Exception as e:
        logger.error(f"Error reading prediction history: {e}")
        return pd.DataFrame()
    finally:
        conn.close()

def delete_prediction(prediction_id):
    logger.info(f"Deleting prediction {prediction_id} from database...")
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("DELETE FROM prediction_history WHERE prediction_id = ?", (prediction_id,))
        conn.commit()
        logger.info(f"Prediction {prediction_id} deleted successfully.")
    except Exception as e:
        logger.error(f"Error deleting prediction {prediction_id}: {e}")
        conn.rollback()
        raise e
    finally:
        conn.close()

def clear_all_history():
    logger.info("Clearing all prediction history from database...")
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("DELETE FROM prediction_history")
        conn.commit()
        logger.info("All history cleared successfully.")
    except Exception as e:
        logger.error(f"Error clearing history: {e}")
        conn.rollback()
        raise e
    finally:
        conn.close()
