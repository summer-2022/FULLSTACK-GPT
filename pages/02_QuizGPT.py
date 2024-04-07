import json
import streamlit as st
import os
from langchain.retrievers import WikipediaRetriever
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import UnstructuredFileLoader
from langchain.schema import BaseOutputParser



class JsonOutputParser(BaseOutputParser):
    def parse(self,text):
        text = text.replace("```","").replace("json","")
        return json.loads(text)
    
output_parser = JsonOutputParser()
    

st.set_page_config(
    page_title= "QuizGPT",
    page_icon= "🧡",
)
st.title("QuizGPT 😆")




def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)



easy_questions_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
            """
    You are a helpful assistant that is role playing as a teacher.

    You should create a quiz appropriate for an elementary school student in the lower grades.

    Based ONLY on the following context make 10 (TEN) questions to test the user's knowledge about the text.
    
    Each question should have 4 answers, three of them must be incorrect and one should be correct.
         
    Use (o) to signal the correct answer.
         
    Question examples:
         
    Question: What is the color of the ocean?
    Answers: Red|Yellow|Green|Blue(o)
         
    Question: What is the capital or Georgia?
    Answers: Baku|Tbilisi(o)|Manila|Beirut
         
    Question: When was Avatar released?
    Answers: 2007|2001|2009(o)|1998
         
    Question: Who was Julius Caesar?
    Answers: A Roman Emperor(o)|Painter|Actor|Model
         
    Your turn!
         
    Context: {context}
""",
            )
        ]

    )


hard_questions_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
            """
    You are a helpful assistant that is role playing as a teacher.

    You need to create a quiz appropriate for individuals who have graduated from high school or beyond.

    You should create a quiz appropriate for an elementary school student in the lower grades.

    Based ONLY on the following context make 10 (TEN) questions to test the user's knowledge about the text.
    
    Each question should have 4 answers, three of them must be incorrect and one should be correct.
         
    Use (o) to signal the correct answer.
         
    Question examples:
         
    Question: What is the color of the ocean?
    Answers: Red|Yellow|Green|Blue(o)
         
    Question: What is the capital or Georgia?
    Answers: Baku|Tbilisi(o)|Manila|Beirut
         
    Question: When was Avatar released?
    Answers: 2007|2001|2009(o)|1998
         
    Question: Who was Julius Caesar?
    Answers: A Roman Emperor(o)|Painter|Actor|Model
         
    Your turn!
         
    Context: {context}
""",
            )
        ]

    )





formatting_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
    You are a powerful formatting algorithm.
     
    You format exam questions into JSON format.
    Answers with (o) are the correct ones.
     
    Example Input:

    Question: What is the color of the ocean?
    Answers: Red|Yellow|Green|Blue(o)
         
    Question: What is the capital or Georgia?
    Answers: Baku|Tbilisi(o)|Manila|Beirut
         
    Question: When was Avatar released?
    Answers: 2007|2001|2009(o)|1998
         
    Question: Who was Julius Caesar?
    Answers: A Roman Emperor(o)|Painter|Actor|Model
    
     
    Example Output:
     
    ```json
    {{ "questions": [
            {{
                "question": "What is the color of the ocean?",
                "answers": [
                        {{
                            "answer": "Red",
                            "correct": false
                        }},
                        {{
                            "answer": "Yellow",
                            "correct": false
                        }},
                        {{
                            "answer": "Green",
                            "correct": false
                        }},
                        {{
                            "answer": "Blue",
                            "correct": true
                        }}
                ]
            }},
                        {{
                "question": "What is the capital or Georgia?",
                "answers": [
                        {{
                            "answer": "Baku",
                            "correct": false
                        }},
                        {{
                            "answer": "Tbilisi",
                            "correct": true
                        }},
                        {{
                            "answer": "Manila",
                            "correct": false
                        }},
                        {{
                            "answer": "Beirut",
                            "correct": false
                        }}
                ]
            }},
                        {{
                "question": "When was Avatar released?",
                "answers": [
                        {{
                            "answer": "2007",
                            "correct": false
                        }},
                        {{
                            "answer": "2001",
                            "correct": false
                        }},
                        {{
                            "answer": "2009",
                            "correct": true
                        }},
                        {{
                            "answer": "1998",
                            "correct": false
                        }}
                ]
            }},
            {{
                "question": "Who was Julius Caesar?",
                "answers": [
                        {{
                            "answer": "A Roman Emperor",
                            "correct": true
                        }},
                        {{
                            "answer": "Painter",
                            "correct": false
                        }},
                        {{
                            "answer": "Actor",
                            "correct": false
                        }},
                        {{
                            "answer": "Model",
                            "correct": false
                        }}
                ]
            }}
        ]
     }}
    ```
    Your turn!

    Questions: {context}

""",
        )
    ]
)

with st.sidebar:
    openai_api_key = None

    openai_api_key = st.text_input(
        "OpenAI API Key",
        key="chatbot_api_key",
        type="password")
    os.environ["OPENAI_API_KEY"] = openai_api_key


    con = st.container()
    con.link_button("GitHub Repository", url="https://github.com/summer-2022/FULLSTACK-GPT/tree/main")


                
if openai_api_key:
    llm = ChatOpenAI(
        temperature=0.1,
        model="gpt-3.5-turbo-0125",
        streaming=True,
        callbacks=[
            StreamingStdOutCallbackHandler()
        ]
    )
    easy_mode_chain = {"context": format_docs}| easy_questions_prompt | llm
    hard_mode_chain = {"context": format_docs}| hard_questions_prompt | llm
    formatting_chain = formatting_prompt | llm 

else:
    st.header("Please enter your OpenAI API key first.")
    st.markdown( 
        """
    Welcome to QuizGPT.
                
    I will make a quiz from Wikipedia articles or files you upload to test your knowledge and help you study.
                
    Get started by uploading a file or searching on Wikipedia in the sidebar.
    """
    )




@st.cache_data(show_spinner="Loading file...")
def split_file(file):
    file_content = file.read()
    file_path= f"/workspaces/FULLSTACK-GPT/.cache/quiz_files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator= "\n",
        chunk_size= 600,
        chunk_overlap= 100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    return docs

@st.cache_data(show_spinner="Making quiz...")
def run_easy_quiz_chain(_docs, topic):
    chain = {"context": easy_mode_chain} | formatting_chain | output_parser
    return chain.invoke(_docs)

@st.cache_data(show_spinner="Making quiz...")
def run_hard_quiz_chain(_docs, topic):
    chain = {"context": hard_mode_chain} | formatting_chain | output_parser
    return chain.invoke(_docs)


@st.cache_data(show_spinner="Searching Wikipedia...")
def wiki_search(term):
    retriever = WikipediaRetriever(top_k_results=2)
    docs = retriever.get_relevant_documents(term)
    return docs





docs = None
topic = None
user_difficulty = None

    
# Add difficulty selection 
difficulty_levels = ["Easy", "Hard"]
user_difficulty = st.selectbox("Select the quiz difficulty:", difficulty_levels)

choice = st.selectbox(
    "Choose what you want to use.",
    (
        "File",
        "Wikipedia Article",
    ),
)

if choice == "File":
    file = st.file_uploader(
        "Upload a .docx, .txt or .pdf file",
        type=["pdf", "txt", "docx" ],
    )
    if file:
        split_file(file)

else:
    topic = st.text_input("Search Wikipedia")
    if topic:
        docs = wiki_search(topic)





    
if docs:
    if user_difficulty == "Easy":
        response = run_easy_quiz_chain(docs, topic if topic else file.name) 
    else:
        response = run_hard_quiz_chain(docs, topic if topic else file.name) 
    # st.write(response)
    with st.form("questionns_form"):
        correct_answers = 0
        total_questions = len(response["questions"])
        number = 0

        for question in response["questions"]:
            number +=1
            st.write(f"QUESTION {number} OF {total_questions}")
            st.write(question["question"])
            value = st.radio(
                "Select an option",
                [answer["answer"] for answer in question["answers"]],
                index =None,
            )
        
            if {"answer":value, "correct":True} in question["answers"]:
                st.success("Correct!")
                correct_answers += 1

            elif value is not None:
                st.error("Incorrect")
        button = st.form_submit_button()

    if button:
        if correct_answers == total_questions:
            st.balloons()
            st.success(f"All correct! Great job!")
        else:
            st.error(f"Score: {correct_answers}/{total_questions}. Some answers were incorrect. Try again!")
else:
    pass
        
       