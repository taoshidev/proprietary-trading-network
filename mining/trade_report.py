import duckdb
import pandas
from datetime import datetime 


def save_trades_to_csv(db_filename: str = 'trades.duckdb', table_name: str = 'trades') -> None:
        conn = duckdb.connect(db_filename)
        try:
            # Retrieve the last row of the table
            result = conn.execute(f"""
                SELECT *
                FROM {table_name}
            """).df()

            if not result.empty:
                        # Get the current datetime
                        current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
                        # Save the result to a CSV file with the current datetime in the filename
                        result.to_csv(f'duck_db_trades_log_{current_datetime}.csv', index=False)
                        print(f"Data saved to duck_db_trades_log_{current_datetime}.csv")
            else:
                        print("The table is empty.")
        finally:
            conn.close()
            
            
save_trades_to_csv()