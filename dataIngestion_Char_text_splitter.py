from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
pdfLoader=PyPDFLoader("test1.pdf")
docs=pdfLoader.load()
# print(docs)

text_splitter1 = CharacterTextSplitter(separator="\n\n",chunk_size=20, chunk_overlap=15)
TextLoader1=TextLoader("speech.txt").load()
text_doc=text_splitter1.split_documents(TextLoader1)  

print(text_doc[0].page_content[:50])
# print(text_doc[1].page_content[:50])