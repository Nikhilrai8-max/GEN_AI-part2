#News summerizer usin tavily search and mistral api

from dotenv import load_dotenv
load_dotenv()
from langchain_mistralai import ChatMistralAI
from langchain_tavily import TavilySearch
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


model = ChatMistralAI(model="mistral-small-2506")
search = TavilySearch(max_results=3)

prompt = ChatPromptTemplate.from_template(
    """You are a news summerizer.
    summerize the following news in simple words and give the summary 
    in 3-4 lines.

    News: {news}
    """
 )

chain = prompt | model | StrOutputParser()

result = search.run("latest news about cricket")
summary = chain.invoke({"news": result})
print("News: ", result)
print("--------------********************-----------------")
print("Summary: ", summary)