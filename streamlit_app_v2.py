import streamlit as st
import random
import time

import os
import json
import duckdb
import pandas as pd
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
#from langchain_chroma import Chroma
from langchain.vectorstores import FAISS
from langchain_core.output_parsers import JsonOutputParser
from langchain.schema import Document
from langchain_core.example_selectors import SemanticSimilarityExampleSelector


# Ask for API token securely
OPENAI_API_KEY = st.text_input("Your API Token", type="password")

def load_metadata(metadata_path):
    
    f = open(metadata_path)
    metadata_json = json.load(f)
    f.close()
    
    result_lines = []
    for table_name, table_info in metadata_json.items():
        table_desc = table_info.get("table_description", "")
        columns = table_info.get("columns", [])
        for col in columns:
            col_name = col.get("name", "")
            col_desc = col.get("column_description", "")
            col_dtype = col.get("data_type", "")
            result_lines.append(Document(page_content=f"Table: '{table_name}', Table Description: '{table_desc}', Column: '{col_name}', Column Description: '{col_desc}', Column Data Type: '{col_dtype}'",\
                                         metadata={"table": table_name, "column": col_name}))
    return result_lines
    
# Only proceed if API token is provided
if OPENAI_API_KEY:
    embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    st.markdown(
        """
        <style>
        div[data-testid="stTabs"] button[aria-selected="true"] {
            color: #1de4d9 !important;
            border-bottom: 3px solid #249EF2 !important;
        }

        div[data-testid="stChatInput"] > div {
            border: 2px solid #249EF2 !important;
            border-radius: 25px !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    
    # Title and description
    st.title(" DB Chatbot")#🤖
    st.markdown("Welcome! Enter your question below to chat with the database")
    st.divider()
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")

    
    metadata_path = "metadata/metadata_v1.json"
    docs = load_metadata(metadata_path)

    # Create FAISS vectorstore in-memory from documents and embeddings
    table_metadata_vectordb = FAISS.from_documents(docs, embedding)
    
    ###################################################
    
    
    f = open(metadata_path)
    metadata_json = json.load(f)
    f.close()
    
    db_path = "database/database_v1"
    
    table_path_mapping = list()
    for table_name, table_info in metadata_json.items():
        table_path_mapping.append([table_name,'"'+db_path+'\\'+table_name.split('.')[1]+'.csv"'])
    
    ###################################################
    
    df = pd.read_csv("good_known_queries/sample_queries.csv")
    
    good_known_queries = df[['question', 'sql_query']].to_dict('records')
    
    good_known_queries_selector = SemanticSimilarityExampleSelector.from_examples(
        good_known_queries,
        embedding,
        vectorstore_cls=FAISS,
        k=5
    )
    
    ###################################################
    
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)

    # Function to generate a fun bot reply (simulated streaming generator)
    def response_generator(user_text):
        responses = [
            "Adffffffffffffffffffffffff",
            "Your token value is " + OPENAI_API_KEY,
            "Cfffffffffffffffffffff",
            f"You said: {user_text}. Tell me more!"
        ]
        response = random.choice(responses)

        # Simulate typing effect
        for word in response.split():
            yield word + ' '
            time.sleep(0.1)

    # Create separate full responses (simulate non-streaming for tabs)
    def generate_response_1(user_text):
        # In real usage, replace with actual API call or logic
        return f"Result: Concise answer for '{user_text}'."

    def generate_response_2(user_text):
        # In real usage, replace with actual API call or logic
        return f"Details: Here is a detailed explanation for '{user_text}'. Additional information can be provided here."

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for msg in st.session_state.messages:
        if msg['role'] == 'user':
            with st.chat_message("user", avatar="icons/user_2.png"):#👨🏻
                st.markdown(msg["content"])
        else:
            with st.chat_message("assistant", avatar="icons/bot_34.png"):#👽
                # Show bot responses in tabs for the last message with 'assistant' role only
                if isinstance(msg["content"], dict):
                    tab1, tab2 = st.tabs(["Result", "Details"])
                    with tab1:
                        st.markdown(msg["content"]["result"])
                    with tab2:
                        st.markdown(msg["content"]["details"])
                else:
                    st.markdown(msg["content"])

    # Chat input box
    if query := st.chat_input("Type your question here..."):

        # Add user message and show it
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user", avatar="icons/user_2.png"):#👨🏻
            st.markdown(query)

        #clarification_kill_switch = 7
        clarification_response = ''
        
        # while clarification_kill_switch >0:
        
        #     clarification_kill_switch -= 1
        #     #print('\nclarification_kill_switch',clarification_kill_switch)
            
        #     query = query
        
        # if (query.lower().strip() == 'exit') or (clarification_response.lower().strip() == 'exit'):
        #     break
            
        relevant_metadata = table_metadata_vectordb.similarity_search(query, k=10)
        relevant_metadata = [str(i+1)+'. '+str(relevant_metadata[i].page_content) + "\n\n" for i in range(len(relevant_metadata))]
        relevant_metadata = " ".join(str(x) for x in relevant_metadata)
    
        selected_good_known_queries = good_known_queries_selector.select_examples({"question": query})
        selected_good_known_queries = "\n\n".join(f"Question {idx + 1}:  {item['question']}\nSQL Query {idx + 1}:{item['sql_query']}"\
                             for idx, item in enumerate(selected_good_known_queries))
        #selected_good_known_queries=""
        
        fetch_sql_prompt = f"""
        You are an expert SQL developer tasked with generating SQL queries based on the provided database schema metadata and user query. There are also examples with question and sql query pairs to help you answer the user question. The examples may or may not be directly related to the user question asked.
        
        Input:
        - Database schema metadata: {relevant_metadata}
        - User question: {query}
        - Example question and sql_query pairs: {selected_good_known_queries}
        
        Instructions:
        1. If the user question is unclear or if the provided metadata is insufficient to confidently write a correct SQL query, respond ONLY with a clarification request formatted exactly as:
           CLARIFICATION_NEEDED: <your specific question to clarify the user's intent>
        
        2. Otherwise, produce ONLY a valid JSON object with these exact keys and requirements:
            - "SQL": "<valid SQL query string>",
            - "Tables_Columns": ["<table1_name>"."<column1_name>","<table1_name>"."<column2_name>","<table2_name>"."<column2_name>"...],
            - "Explanation": "<a very brief explanation (1-2 sentences) of the SQL query>"
        
        3. Do NOT include any text outside the JSON object when providing the SQL response.
        
        4. Ensure the JSON is syntactically valid and parsable.
        
        Your focus is on precision, clarity, and producing executable SQL or clear clarification requests only."""
    
        #print('\nfetch_sql_prompt:\n',fetch_sql_prompt)
        
        result = llm.invoke(fetch_sql_prompt)
        answer = result.content.strip()
    
        if answer.startswith("CLARIFICATION_NEEDED:"):
            clarification_question = answer.replace("CLARIFICATION_NEEDED:", "").strip()
            #print(f"Model needs more info: {clarification_question}")

            st.session_state.messages.append({"role": "assistant", "content": clarification_question})
            with st.chat_message("assistant", avatar="icons/bot_34.png"):#👨🏻
                st.markdown(clarification_question)

        else:
  
            try:
                
                
                #print('\nJSON Parser')
                parser = JsonOutputParser()
                parsed_output = parser.parse(answer)
                #print(parsed_output)
                
            except json.JSONDecodeError:
                # print("\nError: Model did not return valid JSON. Raw output:")
                # print(answer)
                sql_json = {"ERROR": "Error: Model did not return valid JSON. Raw output:\n"+answer}
        
        
            #print('\nJSON Dump')
            sql_json = json.dumps(parsed_output, indent=2)
        
            sql_json = json.loads(sql_json)
            
            #print('\nModel SQL JSON:\n\n',sql_json,'\n')
            
            sql_query_original = sql_json['SQL']
            sql_query = sql_query_original
            
            for i in table_path_mapping:
                sql_query = sql_query.replace(i[0],i[1])

            print("--------------",sql_query,"--------------")
            
            result_df = duckdb.query(sql_query).to_df()
            
            csv_string = result_df.to_csv(index=False)
            
            
            final_answer_prompt = f"""You are a data analyst. Use the following CSV data (with headers) to answer the user's question accurately.
            
            User question: 
            {query}
            
            CSV data:
            {csv_string}
            
            Please provide a clear, concise answer to the question strictly based on the data in the CSV table. If the CSV does not contain enough information to answer, say "Insufficient data to answer the question."
            """
            
            result = llm.invoke(final_answer_prompt)
            answer = result.content.strip()
    
            # Generate two responses
            #result_response = generate_response_1(prompt)
            #details_response = generate_response_2(prompt)
    
            # Store the responses as a dictionary
            bot_response = {"result": answer, "details": result_df}
    
            # Show responses inside tabs
            with st.chat_message("assistant", avatar="icons/bot_34.png"):#👽
                tab1, tab2 = st.tabs(["Result", "Details"])
                with tab1:
                    st.markdown(answer)
                with tab2:
                    #st.markdown(sql_query)
                    
                    st.dataframe(result_df, hide_index=True)
                    st.write("")
                    st.markdown(f"<div style='background-color: #23242a; padding: 10px; border-radius: 5px;'>{sql_query_original}</div>",
                                unsafe_allow_html=True) #color: #249EF2;
    
            # Save the bot response dict in the message history
            st.session_state.messages.append({"role": "assistant", "content": bot_response})

else:
    st.warning("Please enter your API token to continue.")





