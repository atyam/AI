from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
pdfLoader=PyPDFLoader("test1.pdf")
docs=pdfLoader.load()
# print(docs)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
final_doc=text_splitter.split_documents(docs)
# print(final_doc[0].page_content[:200])
# print(final_doc[1].page_content[:200])

text_splitter1 = RecursiveCharacterTextSplitter(chunk_size=20, chunk_overlap=15)
TextLoader1=TextLoader("speech.txt").load()
text_doc=text_splitter1.split_documents(TextLoader1)  

print(text_doc[0].page_content[:50])
print(text_doc[1].page_content[:50])