import sqlite3
import sqlite_vec
from sentence_transformers import SentenceTransformer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the simple embedding model
_embedding_model = None

def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        logger.info("Loading sentence-transformers model...")
        _embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    return _embedding_model

DB_PATH = "chat_memory.sqlite"

def get_db_connection():
    # SQLite connection setup
    conn = sqlite3.connect(DB_PATH)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initializes the database and the required tables with a vector column."""
    with get_db_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS chat_memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                user_msg TEXT,
                assistant_msg TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                embedding FLOAT[384] -- sqlite-vec syntax for exactly 384 dimensions
            );
        """)
        conn.commit()

# Initialize DB on import
init_db()

def embed_text(text: str) -> bytes:
    """Helper function to embed text and return it as continuous bytes float array for sqlite-vec."""
    model = get_embedding_model()
    # model.encode returns a numpy array, sqlite-vec needs it serialized to bytes
    vector = model.encode(text) 
    return sqlite_vec.serialize_float32(vector)

def add_memory(session_id: str, user_msg: str, assistant_msg: str):
    """
    Stores an interaction with the vector embedding of the user's message.
    """
    if not session_id:
        return
        
    embedding_bytes = embed_text(user_msg)
    
    with get_db_connection() as conn:
        conn.execute("""
            INSERT INTO chat_memory (session_id, user_msg, assistant_msg, timestamp, embedding)
            VALUES (?, ?, ?, datetime('now'), ?)
        """, (session_id, user_msg, assistant_msg, embedding_bytes))
        conn.commit()
    logger.info(f"Added memory for session {session_id}")


def get_memories(session_id: str, query: str, top_k: int = 3) -> str:
    """
    Retrieves the Top-K closest historical interactions from the last 48 hours for a given session.
    """
    if not session_id or not query:
        return ""

    query_embedding_bytes = embed_text(query)

    with get_db_connection() as conn:
        # We find vector distance and only records within the last 48 hours for the matching session
        cursor = conn.execute("""
            SELECT 
                user_msg, 
                assistant_msg, 
                timestamp, 
                vec_distance_L2(embedding, ?) as distance
            FROM chat_memory
            WHERE session_id = ?
              AND timestamp >= datetime('now', '-48 hours')
            ORDER BY distance ASC
            LIMIT ?;
        """, (query_embedding_bytes, session_id, top_k))
        
        rows = cursor.fetchall()
        
    if not rows:
        return ""
        
    memories_str = "--- PREVIOUS RELEVANT MEMORIES (Last 48 Hours) ---\n"
    for row in rows:
        memories_str += f"[Time: {row['timestamp']}]\n"
        memories_str += f"User: {row['user_msg']}\n"
        memories_str += f"Assistant: {row['assistant_msg']}\n"
        memories_str += "---\n"
        
    return memories_str

def cleanup_old_memories():
    """
    Hard deletion mechanism. 
    Deletes any vector embeddings and associated metadata that are older than 48 hours.
    """
    with get_db_connection() as conn:
        cursor = conn.execute("""
            DELETE FROM chat_memory
            WHERE timestamp < datetime('now', '-48 hours')
        """)
        deleted_rows = cursor.rowcount
        conn.commit()
        
    logger.info(f"TTL Cleanup: Deleted {deleted_rows} old memories.")
    return deleted_rows

if __name__ == '__main__':
    # basic test to be ran internally to ensure initialization doesn't throw.
    print("Database Initialized")
