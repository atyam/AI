import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
load_dotenv()
output_parser=StrOutputParser() #initialize the StrOutputParser class
# print("ChatOpenAI class initialized successfully!")
os.environ["HUGGING_FACE_API_KEY"] = os.getenv("HUGGING_FACE_API_KEY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
promt=ChatPromptTemplate.from_messages([("system","You are a Expert AI Engineer assistant."),("user","{input}")]) #test the ChatPromptTemplate class with a query

llm=ChatOpenAI(model="gpt-4o") #initialize the ChatOpenAI class with the model name
chain=promt| llm|output_parser
response=chain.invoke({"input":"can you tell me about Diwali"}) #test the ChatOpenAI class with a query
print("Response: ", response);
# # print(llm);
# answer=llm.invoke("what do you know about Narendra Modi?") #test the ChatOpenAI class with a query
# print("Answer: ", answer.content);
