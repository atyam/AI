## data ingestion techniques
##text loader
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
##web based loader
from langchain_community.document_loaders import WebBaseLoader
##arxiv loader
from langchain_community.document_loaders import ArxivLoader
from langchain_community.document_loaders import WikipediaLoader
import pymupdf
import wikipedia

wikipedia.set_user_agent(
    "LangchainLearning/1.0 (contact:aditya.tyagi@gmail.com)"
)

TextLoader=TextLoader("speech.txt")
text=TextLoader.load()

# print(text)


pdfLoader=PyPDFLoader("test1.pdf")
docs=pdfLoader.load()
# print(docs)

##webbased loader
loader=WebBaseLoader(web_paths=("https://daccet.blogspot.com/",),)
docs1=loader.load()    
# print(docs1)

arxivloader = ArxivLoader(
    query="1706.03762",
    load_max_docs=1
).load()

# print(arxivloader[0].page_content[:500])
# print(arxivloader[0].metadata)# pdf_path = "speech.pdf"

docswiki=WikipediaLoader("Python (programming language)",load_max_docs=2).load() 
print(docswiki[0].page_content[:500])
# reader = PdfReader(pdf_path)
# for i, page in enumerate(reader.pages):
#     try:
#         text = page.extract_text()
#         print(f"Page {i + 1}: extracted {len(text or '')} characters")
#     except Exception as e:
#         print(f"Page {i + 1} failed: {e}")