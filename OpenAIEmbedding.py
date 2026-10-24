import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
#vector store db
from langchain_community.vectorstores import Chroma
txtLoader=TextLoader("speech.txt") #load the text file using the TextLoader class
docs=txtLoader.load()
slipter=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50) #split the documents into chunks of 1000 characters with no overlap    
final_documents=slipter.split_documents(docs) #split the documents into chunks
load_dotenv() #load all env variables
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
embedding=OpenAIEmbeddings(model="text-embedding-3-large",dimensions=1024) #initialize the OpenAIEmbeddings class with the model name


# result=embedding.embed_documents("hello bro!") #test the embedding class with a list of documents
# print(result)

#vector embeding and vector store
db=Chroma.from_documents(final_documents,embedding) #create a vector store from the documents and the embedding class
print("Vector store created successfully!")
retrieve_results=db.similarity_search("they can help people access knowledge") #test the vector store with a query
print("Retrieved results: ", retrieve_results)

