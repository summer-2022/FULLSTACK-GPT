import streamlit as st
import json
import time
from langchain.utilities import DuckDuckGoSearchAPIWrapper, WikipediaAPIWrapper
from langchain.tools import DuckDuckGoSearchResults
from langchain.document_loaders import AsyncHtmlLoader
from langchain.document_transformers import Html2TextTransformer
from openai import OpenAI


st.set_page_config(
    page_title= "ResaerchAssistant",
    page_icon= "🧡",
)
st.title("Resaerch Assistant")



with st.sidebar:
    st.link_button(
            "GitHub Repo",
            "https://github.com/summer-2022/FULLSTACK-GPT/tree/main/pages",
        )
    if "openai_api_key" not in st.session_state:
            st.session_state["openai_api_key"] = ""

    st.session_state.openai_api_key = st.text_input(
        "OpenAI API Key",
        key="openai_api_key",
        type="password")
        
    openai_api_key= st.session_state.openai_api_key



def search_wikipedia(inputs):
    wkp = WikipediaAPIWrapper()
    query = inputs["query"]
    return wkp.run(query)

def search_duckduckgo(inputs):
    ddg = DuckDuckGoSearchAPIWrapper()
    query = inputs["query"]
    return ddg.run(query)

def get_document_text(inputs):
    url = inputs["url"]
    loader = AsyncHtmlLoader(url)
    docs = loader.load()
    html2text = Html2TextTransformer()
    docs = html2text.transform_documents(docs)
    return docs



functions_map = {
    "wikipedia_search": search_wikipedia,
    "duckduckgo_search": search_duckduckgo,
    "get_document_text": get_document_text,
}

functions = [
    {
        "type": "function",
        "function": {
            "name": "search_wikipedia",
            "description": "Use this tool to find the websites for the given query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The query you will search for. Example query: Research about the XZ backdoor",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_duckduckgo",
            "description": "Use this tool to find the websites for the given query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Use this tool to enter the website link and extract its content. Example query: Research about the XZ backdoor",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_document_text",
            "description": "Use this tool to enter the website link and extract its content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The url you will load. Example url: https://en.wikipedia.org/wiki/Backdoor_(computing)",
                    }
                },
                "required": ["url"],
            },
        },
    },
]

client = OpenAI(openai_api_key)

@st.cache_data
def create_assistant():
    return client.beta.assistants.create(
        name="Research Assistant",
        instructions="""
        You are a helful research manager assistant.
        
        The you should try to search in Wikipedia or DuckDuckGo. 
        If you find the most relevant website in DuckDuckGo you should enter the just ONLY ONE website and extract its content.
        Afte that, it should finish by saving the research to a .txt file.
            
        """,
        model="gpt-3.5-turbo-0125",
        tools=functions,
    )

@st.cache_data
def create_thread(message_content):
    return client.beta.threads.create(
        messages=[
            {
                "role": "user",
                "content": message_content,
            }
        ]
    )

@st.cache_data
def create_run(thread_id, assistant_id):
    return client.beta.threads.runs.create(
        thread_id=thread_id,
        assistant_id=assistant_id,
    )

def get_run(run_id, thread_id):
    return client.beta.threads.runs.retrieve(
        run_id=run_id,
        thread_id=thread_id,
    )

def send_message(thread_id, content):
    return client.beta.threads.messages.create(
        thread_id=thread_id, role="user", content=content
    )

def get_messages(thread_id, assistant_only=True):
    messages = client.beta.threads.messages.list(thread_id=thread_id).data
    messages.reverse()
    result = ""
    for message in messages:
        if assistant_only and message.role != "assistant":
            continue
        result = result + f"\n\n{message.content[0].text.value}"
    return result

def get_tool_outputs(run_id, thread_id):
    run = get_run(run_id, thread_id)
    outputs = []
    for action in run.required_action.submit_tool_outputs.tool_calls:
        action_id = action.id
        function = action.function
        print(f"Calling function: {function.name} with arg {function.arguments}")
        outputs.append(
            {
                "output": functions_map[function.name](json.loads(function.arguments)),
                "tool_call_id": action_id,
            }
        )
    return outputs

def submit_tool_outputs(run_id, thread_id):
    outpus = get_tool_outputs(run_id, thread_id)
    return client.beta.threads.runs.submit_tool_outputs(
        run_id=run_id, thread_id=thread_id, tool_outputs=outpus
    )


def make_download_button(docs):
    file_path= f"/workspaces/FULLSTACK-GPT/files/{time.strftime('%H%M%S')}.txt"
    with open(file_path, "wb") as f:
        docs_bytes = docs.encode('utf-8')
        f.write(docs_bytes)
    
    file_bytes = open(file_path, "rb").read()

    st.download_button(label= 'Download text file',
                       data= file_bytes)
