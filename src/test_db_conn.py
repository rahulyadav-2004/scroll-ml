import os
import psycopg2
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def test_connection():
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        print("❌ Error: DATABASE_URL not found in .env file.")
        return

    try:
        # Attempt to connect to the database
        conn = psycopg2.connect(db_url)
        cur = conn.cursor()
        
        # Simple query to test the connection
        cur.execute("SELECT version();")
        db_version = cur.fetchone()
        print(f"✅ Successfully connected to the database!")
        print(f"Database version: {db_version[0]}")
        
        # Test if our new tables exist
        cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' AND table_name IN ('scrolls', 'scroll_interactions', 'scroll_impressions');")
        tables = cur.fetchall()
        print("\nFound tables related to scrolls:")
        for table in tables:
            print(f"- {table[0]}")
            
        cur.close()
        conn.close()
    except Exception as e:
        print(f"❌ Error connecting to the database: {e}")

if __name__ == "__main__":
    test_connection()
