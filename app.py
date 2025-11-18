import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_objectbox.vectorstores import ObjectBox
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain import hub
import pprint

# Load environment variables
load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")

# Streamlit UI setup
st.title("🏏2024 IPL Information and Q&A")
st.write("""
This application loads data from the Wikipedia page about the 2024 Indian Premier League (IPL) 
and allows you to ask questions about it. Ask your queries about IPL - GPT will answer!
""")

# Load data from Wikipedia
@st.cache_resource
def load_data():
    loader = WebBaseLoader("https://en.wikipedia.org/wiki/2024_Indian_Premier_League")
    return loader.load()

data = load_data()

# Split documents
text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(data)

# Create vector store
@st.cache_resource
def create_vector_store(_documents):
    return ObjectBox.from_documents(_documents, OpenAIEmbeddings(), embedding_dimensions=768)

vector = create_vector_store(documents)

# Initialize the language model
llm = ChatOpenAI(model="gpt-4o")  # Using GPT-4o
prompt = hub.pull("rlm/rag-prompt")

# Create QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vector.as_retriever(),
    chain_type_kwargs={"prompt": prompt}
)

with st.form(key='question_form', clear_on_submit=True):
    question = st.text_area(
        "Enter your question about the 2024 IPL:",
        key='question_input',
        placeholder="Which teams qualified for playoffs?",
    )
    submit_button = st.form_submit_button(label='Submit')

# Process the question and display the answer
if submit_button:
    if question:
        result = qa_chain({"query": question})
        pp = pprint.PrettyPrinter(indent=5)
        st.write("### Answer:")
        st.write(result["result"])
    else:
        st.write("Please enter a question.")