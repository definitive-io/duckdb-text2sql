import streamlit as st
import os
from groqcloud import Groqcloud
import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import duckdb
import sqlparse


def get_conversational_history(base_prompt,user_question_history,chatbot_answer_history,memory_length):
    
    user_question_history = user_question_history[memory_length*-1:]
    chatbot_answer_history = chatbot_answer_history[memory_length*-1:]
    if len(chatbot_answer_history) > 0:
        conversational_history = '''
        
        As a recap, here is the current conversation:
            
        ''' + "\n".join(f"Human: {q}\nAI: {a}" for q, a in zip(user_question_history, chatbot_answer_history))

        full_prompt = base_prompt + conversational_history + '''
            Human: {user_question}
            AI:
        '''.format(user_question = user_question_history[-1])
    else:
        full_prompt = base_prompt.format(user_question = user_question_history[-1])
    
    return full_prompt


def chat_with_groq(client,prompt,model):
    
    completion = client.chat.completions.create(
    model=model,
    messages=[
      {
            "role": "user",
            "content": prompt
        }
        ]
    )
  
    return completion.choices[0].message.content


def execute_duckdb_query(query):

    original_cwd = os.getcwd()
    os.chdir('data')
    
    try:
        conn = duckdb.connect(database=':memory:', read_only=False)
        query_result = conn.execute(query).fetchdf().reset_index(drop=True)
    finally:
        os.chdir(original_cwd)


    return query_result



def get_json_output(llm_response):

    # remove bad characters and whitespace
    llm_response_no_escape = llm_response.replace('\\n', ' ').replace('\n', ' ').replace('\\', '').replace('\\', '').strip() 
    
    # Just in case - gets only content between brackets
    open_idx = llm_response_no_escape.find('{')
    close_idx = llm_response_no_escape.rindex('}') + 1
    cleaned_result = llm_response_no_escape[open_idx : close_idx]

    json_result = json.loads(cleaned_result)
    if 'sql' in json_result:
        query = json_result['sql']
        return True,sqlparse.format(query, reindent=True, keyword_case='upper')
    elif 'error' in json_result:
        return False,json_result['error']



def get_reflection(client,full_prompt,llm_response,model):
    reflection_prompt = '''
    You were giving the following prompt:

    {full_prompt}

    This was your response:

    {llm_response}

    There was an error with the response, either in the output format or the query itself.

    Ensure that the following rules are satisfied when correcting your response:
    1. SQL is valid DuckDB SQL, given the provided metadata and the DuckDB querying rules
    2. The query SPECIFICALLY references the correct tables: employees.csv and purchases.csv, and those tables are properly aliased? (this is the most likely cause of failure)
    3. Response is in the correct format ({{sql: <sql_here>}} or {{"error": <explanation here>}}) with no additional text?
    4. All fields are appropriately named
    5. There are no unnecessary sub-queries
    6. ALL TABLES are aliased (extremely important)

    Rewrite the response and respond ONLY with the valid output format with no additional commentary

    '''.format(full_prompt = full_prompt, llm_response=llm_response)

    return chat_with_groq(client,reflection_prompt,model)


def get_summarization(client,user_question,df,model,additional_context):
    prompt = '''
    A user asked the following question pertaining to local database tables:
    
    {user_question}
    
    To answer the question, a dataframe was returned:

    Dataframe:
    {df}

    In a few sentences, summarize the data in the table as it pertains to the original user question. Avoid qualifiers like "based on the data" and do not comment on the structure or metadata of the table itself
    '''.format(user_question = user_question, df = df)

    if additional_context != '':
        prompt += '''\n
        The user has provided this additional context:
        {additional_context}
        '''.format(additional_context=additional_context)

    return chat_with_groq(client,prompt,model)


def main():
    
    groq_api_key = os.environ['GROQ_API_KEY']
    client = Groqcloud(
        # This is the default and can be omitted
        api_key=groq_api_key,
        base_url="https://api.groqcloud.com"
    )

    spacer, col = st.columns([5, 1])  
    with col:  
        st.image('groqcloud_darkmode.png')

    st.title("DuckDB Query Generator")
    st.write("Ask questions about your DuckDB data, powered by Groq")

    st.sidebar.title('Customization')
    additional_context = st.sidebar.text_input('Enter additional summarization context for the LLM here (i.e. write it in spanish):')
    model = st.sidebar.selectbox(
        'Choose a model',
        ['mixtral-8x7b-32768', 'llama2-70b-4096']
    )
    max_num_reflections = st.sidebar.slider('Max reflections:', 0, 20, value=10)
    conversational_memory = st.sidebar.slider('Conversational memory length:', 1, 10)


    with open('prompts/base_prompt.txt', 'r') as file:
        base_prompt = file.read()
    user_question = st.text_input("Ask a question:")

    if 'user_question_history' not in st.session_state:
        st.session_state['user_question_history'] = []

    if 'chatbot_answer_history' not in st.session_state:
        st.session_state['chatbot_answer_history'] = []

    
    if user_question:
        st.session_state['user_question_history'].append(user_question)
        full_prompt = get_conversational_history(base_prompt,st.session_state['user_question_history'],st.session_state['chatbot_answer_history'],conversational_memory)
        
        llm_response = chat_with_groq(client,full_prompt,model)
        st.session_state['chatbot_answer_history'].append(llm_response)

        valid_response = False
        i=0
        while valid_response is False and i < max_num_reflections:
            try:
                is_sql,result = get_json_output(llm_response)
                if is_sql:
                    results_df = execute_duckdb_query(result)
                    valid_response = True
                else:
                    valid_response = True
            except:
                llm_response = get_reflection(client,full_prompt,llm_response,model)
                i+=1
                #st.write('BAD:',llm_response)

        try:
            if is_sql:
                st.markdown("```sql\n" + result + "\n```")
                st.markdown(results_df.to_html(index=False), unsafe_allow_html=True)

                summarization = get_summarization(client,user_question,results_df,model,additional_context)
                st.write(summarization.replace('$','\\$'))
            else:
                st.write(result)
        except:
            st.write("ERROR:", 'Could not generate valid SQL for this question')
            st.write(llm_response)
            

if __name__ == "__main__":
    main()

