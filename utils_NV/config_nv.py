# config.py (New module for configuration)
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
print(BASE_DIR)
DATA_DIR = os.path.join(BASE_DIR, 'data')
print(DATA_DIR)
EMBEDDING_DIR = os.path.join(BASE_DIR, 'embeddings')
print(EMBEDDING_DIR)

# Set the database file path
DB_FILE_DIR = os.path.join(BASE_DIR, 'fraud_kb')
print(DB_FILE_DIR)

CHECK_FRAUD_PATTERNS_CSV = os.path.join(DATA_DIR, "check_fraud_patterns.csv")
print(CHECK_FRAUD_PATTERNS_CSV)

FAISS_INDEX_PATH = os.path.join(EMBEDDING_DIR, 'faiss_index')
print(FAISS_INDEX_PATH)

DB_FILE_PATH = os.path.join(DB_FILE_DIR, 'fraud_detection_db')
print(DB_FILE_PATH)
