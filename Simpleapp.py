import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

embeddings=OpenAIEmbeddings() #initialize the OpenAIEmbeddings class with the model name
### load data->Docs--> divide into chunks-->embedding-->vector store-->query-->retrive results
load_dotenv()
os.environ["HUGGING_FACE_API_KEY"] = os.getenv("HUGGING_FACE_API_KEY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"


loader=WebBaseLoader("https://en.wikipedia.org/wiki/Bigg_Boss_(Hindi_TV_series)_season_20")
loaded_docs=loader.load()

splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100) #split the documents into chunks of 1000 characters with no overlap
splitted_docs=splitter.split_documents(loaded_docs)
vector_store_db=FAISS.from_documents(splitted_docs,embeddings) #initialize the FAISS class with the embedding function

output_parser=StrOutputParser() #initialize the StrOutputParser class


# promt=ChatPromptTemplate.from_messages([("system","You are a Expert AI Engineer assistant."),("user","{input}")]) #test the ChatPromptTemplate class with a query

