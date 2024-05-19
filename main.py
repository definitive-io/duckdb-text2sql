from fastapi import FastAPI, HTTPException, Query
from typing import Optional
import mysql.connector
from mysql.connector import Error
import pandas as pd
import os
import shutil
from openai import OpenAI
import config
from groq import Groq
import duckdb
import sqlparse
import json

from pydantic import BaseModel



app = FastAPI()

class QueryRequest(BaseModel):
    user_question: str


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


def generate_prompt_and_update_files(update_csv_path, save_csv_path, organization_id, sheet_id):
    client = OpenAI(api_key=os.getenv("GPT_SERVER_KEY"))

    SYSTEM_TEMPLATE = """Please read the columns of the dataframe and provide a description of their attributes. write it in the same form as the example.
    ###Example

    [DateFrame]
    {}

    [Attributes]
    회사명 (VARCHAR): The name of the company.
    카테고리 (VARCHAR): The industry category to which the company belongs (e.g., 소프트웨어, 컨설팅, 제조업).
    거래단계 (VARCHAR): The current stage of the transaction with the company (e.g., 잠재고객, 협상 중, 자격 확인됨).
    회사설명 (TEXT): A brief description of the company's business and specialization.
    도메인 (VARCHAR): The company's website domain.
    투자단계 (VARCHAR): The investment stage of the company (e.g., 시리즈 A, 시리즈 B, 시리즈 C).
    거래일 (DATE): The date of the transaction.
    거래금액 (INTEGER): The amount of money involved in the transaction."""

    USER_TEMPLATE = """[DateFrame]
    {}

    [Attributes]"""

    NEW_PROMPT_TEMPLATE = """You are Groq Advisor, and you are tasked with generating SQL queries for DuckDB based on user questions about data stored in two tables derived from CSV files:
    [Table] 
    {file_name}

    [Columns]
    {atttributes}


    Given a user's question about this data, write a valid DuckDB SQL query that accurately extracts or calculates the requested information from these tables and adheres to SQL best practices for DuckDB, optimizing for readability and performance where applicable.

    Here are some tips for writing DuckDB queries:
    * DuckDB syntax requires querying from the .csv file itself, i.e. crm.csv. For example: SELECT * FROM crm.csv as crm_table
    * All tables referenced MUST be aliased
    * DuckDB does not implicitly include a GROUP BY clause
    * CURRENT_DATE gets today's date
    * Aggregated fields like COUNT(*) must be appropriately named

    Question:
    --------
    {{user_question}}
    --------
    Reminder: Generate a DuckDB SQL to answer to the question:
    * respond as a valid JSON Document
    * [Best] If the question can be answered with the available tables: {{"sql": <sql here>}} 
    * If the question cannot be answered with the available tables: {{"error": <explanation here>}}
    * Ensure that the entire output is returned on only one single line
    * Keep your query as simple and straightforward as possible; do not use subqueries

    The user has provided this additional context: 'please write explanation it in korean'"""

    if os.path.exists(save_csv_path):
        # CSV Updating Process
        OLD = pd.read_csv(save_csv_path)
        OLD_MD = OLD[:10].to_markdown(index=False)
        old_attribute = list(OLD.columns)

        NEW = pd.read_csv(update_csv_path)
        NEW_MD = NEW.to_markdown(index=False)
        new_attribute = list(NEW.columns)
        FILE_NAME = update_csv_path.split("/")[-1]

        # Prompt Updating
        if old_attribute != new_attribute:
            system_prompt = SYSTEM_TEMPLATE.format(OLD_MD)
            user_prompt = USER_TEMPLATE.format(NEW_MD)

            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1
            )

            output = response.choices[0].message.content.replace("[Attributes]\n", "")
            new_prompt = NEW_PROMPT_TEMPLATE.format(file_name=FILE_NAME, atttributes=output)

            prompt_path = './prompts/{}_prompt.txt'.format(FILE_NAME.split(".")[0])
            with open(prompt_path, 'r', encoding='utf-8') as file:
                old_prompt = file.readlines()
                old_prompt = ''.join(old_prompt)
            with open(prompt_path, 'w', encoding='utf-8') as file:
                file.writelines(new_prompt)

        # CSV Updating
        shutil.move(update_csv_path, save_csv_path)

    else:
        # CSV ADD Process
        DEFAULT_DF = pd.read_csv('./data/default.csv')

        NEW = pd.read_csv(update_csv_path)
        NEW_MD = NEW.to_markdown(index=False)
        FILE_NAME = update_csv_path.split("/")[-1]

        system_prompt = SYSTEM_TEMPLATE.format(DEFAULT_DF)
        user_prompt = USER_TEMPLATE.format(NEW_MD)

        # Prompt ADD
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1
        )

        output = response.choices[0].message.content.replace("[Attributes]\n", "")
        new_prompt = NEW_PROMPT_TEMPLATE.format(file_name=FILE_NAME, atttributes=output)

        prompt_path = './prompts/{}_prompt.txt'.format(FILE_NAME.split(".")[0])
        with open(prompt_path, 'w', encoding='utf-8') as file:
            file.write(new_prompt)

        save_csv_path = update_csv_path.replace("data/tmp/", "data/")

        # CSV ADD
        shutil.move(update_csv_path, save_csv_path)


@app.get("/update-sheet/{organization_id}/{sheet_id}")
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
        if not os.path.exists('data/tmp'):
            os.makedirs('data/tmp')
        update_csv_path = f"data/tmp/{organization_id}_{sheet_id}.csv"
        save_csv_path = f"data/{organization_id}_{sheet_id}.csv"
        df.to_csv(update_csv_path, index=False)

        # Generate prompt and update files
        generate_prompt_and_update_files(update_csv_path, save_csv_path, organization_id, sheet_id)

        return {"detail": f"CSV file saved at {save_csv_path}"}

    except Error as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")

    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()


def chat_with_groq(client, prompt, model):
    completion = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}]
    )
    return completion.choices[0].message.content


def execute_duckdb_query(query):
    try:
        conn = duckdb.connect(database=":memory:", read_only=False)
        query_result = conn.execute(query).fetchdf().reset_index(drop=True)
    finally:
        conn.close()
    return query_result


def get_json_output(llm_response):
    llm_response_no_escape = (
        llm_response.replace("\\n", " ")
        .replace("\n", " ")
        .replace("\\", "")
        .replace("\\", "")
        .strip()
    )
    open_idx = llm_response_no_escape.find("{")
    close_idx = llm_response_no_escape.rindex("}") + 1
    cleaned_result = llm_response_no_escape[open_idx:close_idx]
    json_result = json.loads(cleaned_result)
    if "sql" in json_result:
        query = json_result["sql"]
        return True, sqlparse.format(query, reindent=True, keyword_case="upper")
    elif "error" in json_result:
        return False, json_result["error"]


def get_reflection(client, full_prompt, llm_response, model):
    reflection_prompt = f"""
    You were given the following prompt:

    {full_prompt}

    This was your response:

    {llm_response}

    There was an error with the response, either in the output format or the query itself.

    Ensure that the following rules are satisfied when correcting your response:
    1. SQL is valid DuckDB SQL, given the provided metadata and the DuckDB querying rules
    2. The query SPECIFICALLY references the correct tables: crm.csv, and those tables are properly aliased? (this is the most likely cause of failure)
    3. Response is in the correct format ({{"sql": <sql_here>}} or {{"error": <explanation here>}}) with no additional text?
    4. All fields are appropriately named
    5. There are no unnecessary sub-queries
    6. ALL TABLES are aliased (extremely important)

    Rewrite the response and respond ONLY with the valid output format with no additional commentary
    """
    return chat_with_groq(client, reflection_prompt, model)


def get_summarization(client, user_question, df, model, additional_context):
    prompt = f"""
    A user asked the following question pertaining to local database tables:

    {user_question}

    To answer the question, a dataframe was returned:

    Dataframe:
    {df.to_markdown(index=False)}

    In a few sentences, summarize the data in the table as it pertains to the original user question. Avoid qualifiers like "based on the data" and do not comment on the structure or metadata of the table itself
    """
    if additional_context:
        prompt += f"\nThe user has provided this additional context:\n{additional_context}"
    return chat_with_groq(client, prompt, model)


@app.post("/generate_query")
async def generate_query(request: QueryRequest):
    groq_api_key = os.getenv("GROQ_API_KEY")
    client = Groq(api_key=groq_api_key)
    model = "llama3-70b-8192"
    max_reflections = 5
    additional_context = ""

    with open("prompts/base_prompt.txt", "r") as file:
        base_prompt = file.read()

    full_prompt = base_prompt.format(user_question=request.user_question)
    llm_response = chat_with_groq(client, full_prompt, model)

    valid_response = False
    i = 0
    while not valid_response and i < max_reflections:
        try:
            is_sql, result = get_json_output(llm_response)
            if is_sql:
                results_df = execute_duckdb_query(result)
                valid_response = True
            else:
                valid_response = True
        except:
            llm_response = get_reflection(client, full_prompt, llm_response, model)
            i += 1

    if not valid_response:
        raise HTTPException(status_code=500, detail="Could not generate valid SQL for this question")

    try:
        if is_sql:
            summarization = get_summarization(client, request.user_question, results_df, model, additional_context)
            return {
                "query": result,
                "data": results_df.to_dict(orient="records"),
                "summarization": summarization.replace("$", "\\$")
            }
        else:
            return {"error": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error displaying the result: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
