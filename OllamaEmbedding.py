from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
doc=TextLoader("speech.txt").load() #load the text file using the TextLoader class

splitter=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50) #split the documents into chunks of 1000 characters with no overlap
final_documents=splitter.split_documents(doc) #split the documents into chunks

from langchain_community.vectorstores import Chroma    
embeddings=(OllamaEmbeddings(model='nomic-embed-text')) #initialize the OllamaEmbeddings class with the model name and type
print("Embedding class initialized successfully!")
res=embeddings.embed_query("hello bro!") #test the embedding class with a list of documents
print("Embedding result: ", res)

db=Chroma.from_documents(final_documents,embeddings) #create a vector store from the documents and the embedding class
print("Vector store created successfully!")
retrieve_results=db.similarity_search("they can help people access knowledge") #test the vector store with a query
print("Retrieved results: ", retrieve_results)
