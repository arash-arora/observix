from sqlalchemy import create_engine, text

CLICKHOUSE_URL = "clickhouse+native://clickhouse:clickhouse@localhost:9000/default"

def reset_database():
    engine = create_engine(CLICKHOUSE_URL)
    with engine.connect() as conn:
        print("Dropping tables...")
        conn.execute(text("DROP TABLE IF EXISTS traces"))
        conn.execute(text("DROP TABLE IF EXISTS observations"))
        print("Tables dropped.")

if __name__ == "__main__":
    reset_database()
