from fastapi import FastAPI, HTTPException
import mysql.connector
from mysql.connector import Error
import pandas as pd
import os
import config

app = FastAPI()


def get_db_connection():
    try:
        connection = mysql.connector.connect(
            host=config.MYSQL_HOST,
            port=config.MYSQL_PORT,
            user=config.MYSQL_USER,
            password=config.MYSQL_PASSWORD,
            database=config.MYSQL_DATABASE,
        )
        if connection.is_connected():
            return connection
    except Error as e:
        print(f"Error: {e}")
        return None


@app.get("/export_csv/{organization_id}/{sheet_id}")
def export_csv(organization_id: int, sheet_id: int):
    connection = get_db_connection()
    if connection is None:
        raise HTTPException(status_code=500, detail="Database connection failed")

    try:
        cursor = connection.cursor(dictionary=True)

        # Fetch attribute names for CSV header
        cursor.execute("""
        SELECT a.id AS attribute_id, a.name AS attribute_name
        FROM attribute a
        JOIN sheet s ON a.sheet_id = s.id
        JOIN organization o ON s.organization_id = o.organization_id
        WHERE o.organization_id = %s
          AND s.id = %s;
        """, (organization_id, sheet_id))
        attributes = cursor.fetchall()

        if not attributes:
            raise HTTPException(status_code=404, detail="No attributes found for the given organization and sheet")

        attribute_ids = [attribute['attribute_id'] for attribute in attributes]
        headers = ['company_name'] + [attribute['attribute_name'] for attribute in attributes]

        # Create dynamic query part for attributes
        attribute_cases = ', '.join(
            [f"MAX(CASE WHEN attribute_id = {attr['attribute_id']} THEN value END) AS {attr['attribute_name']}" for attr
             in attributes]
        )

        # Fetch content values for the attributes
        query = f"""
        SELECT c.name AS company_name, {attribute_cases}
        FROM content ct
        JOIN deal d ON ct.deal_id = d.id
        JOIN company c ON d.company_id = c.id
        WHERE attribute_id IN ({', '.join(map(str, attribute_ids))})
        GROUP BY c.name
        """
        cursor.execute(query)
        content_values = cursor.fetchall()

        # Create DataFrame and save to CSV
        df = pd.DataFrame(content_values, columns=headers)
        if not os.path.exists('data'):
            os.makedirs('data')
        file_path = f"data/{organization_id}_{sheet_id}.csv"
        df.to_csv(file_path, index=False)

        return {"detail": f"CSV file saved at {file_path}"}

    except Error as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")

    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
