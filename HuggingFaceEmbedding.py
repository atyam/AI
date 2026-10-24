from langchain_huggingface import HuggingFaceEmbeddings
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
doc=TextLoader("speech.txt").load() #load the text file using the TextLoader class

splitter=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50) #split the documents into chunks of 1000 characters with no overlap
final_documents=splitter.split_documents(doc) #split the documents into chunks

load_dotenv()
os.environ["HUGGING_FACE_API_KEY"] = os.getenv("HUGGING_FACE_API_KEY")
embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
print("HuggingFace Embedding class initialized successfully!")
from langchain_community.vectorstores import Chroma    
#save db
db=Chroma.from_vector_query(vectors,embeddings,persist_directory="./chroma_db") #create a vector store from the documents and the embedding class

#retrieveDB from local
db2=Chroma(persist_directory="./chroma_db",embedding_function=embeddings) #retrieve the vector store from the local directory   
print("Vector store created successfully!")
retrieve_results=db.similarity_search("they can help people access knowledge") #test the vector store with a query
print("Retrieved results: ", retrieve_results)

#langchain_huggingface