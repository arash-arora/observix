import asyncio
import time

from sqlalchemy import create_engine, text

from obs_sdk import trace_decorator


def run_demo():
    print("🚀 Running Zero-Config Observability Demo...")
    # NOTE: init_observability is NOT called manually. 
    # It runs automatically on import if .env is present.
    
    print("running functions...")

    @trace_decorator(name="calculate_sum")
    def calculate_sum(a, b):
        time.sleep(0.1)
        return a + b

    @trace_decorator(name="async_multiply")
    async def multiply(a, b):
        await asyncio.sleep(0.1)
        return a * b

    @trace_decorator(name="risky_function")
    def risky_function():
        raise ValueError("Oops! Something went wrong.")

    # 1. Sync function
    res = calculate_sum(5, 10)
    print(f"   calculate_sum(5, 10) = {res}")

    # 2. Async function
    res_async = asyncio.run(multiply(3, 4))
    print(f"   async_multiply(3, 4) = {res_async}")

    # 3. Error handling
    try:
        risky_function()
    except ValueError:
        print("   risky_function() caught expected error.")

    print("✅ Functions executed. Waiting for background export...")
    # Allow some time for the background thread pool to flush observations
    # and OTel batch processor to flush spans.
    time.sleep(6) 
    
    print("🔍 querying database for traces and observations...")
    query_database()

def query_database():
    import os
    # Connect directly to ClickHouse to verify backend wrote data
    # Assuming standard ClickHouse port 9000 is open
    url = "clickhouse+native://clickhouse:clickhouse@localhost:9000/default"
    engine = create_engine(url)
    with engine.connect() as conn:
        print("\n--- TRACES (Last 5) ---")
        try:
            # Check for project_id and user_id which are new
            result = conn.execute(text(
                "SELECT name, duration_ms, trace_id, project_id, user_id FROM traces "
                "ORDER BY start_time DESC LIMIT 5"
            ))
            for row in result:
                print(
                    f"Trace: {row[0]}, Duration: {row[1]:.2f}ms, "
                    f"ID: {row[2]}, Project: {row[3]}, User: {row[4]}"
                )
        except Exception as e:
            print(f"Error querying traces: {e}")

        print("\n--- OBSERVATIONS (Last 5) ---")
        try:
            # Check for observations table
            result = conn.execute(text(
                "SELECT name, type, input_text, output_text, project_id FROM observations "
                "ORDER BY created_at DESC LIMIT 5"
            ))
            for row in result:
                print(f"Obs: {row[0]} ({row[1]})")
                print(f"     Input: {row[2]}")
                print(f"     Output: {row[3]}")
                print(f"     Project: {row[4]}")
        except Exception as e:
            print(f"Error querying observations: {e}")

if __name__ == "__main__":
    run_demo()
