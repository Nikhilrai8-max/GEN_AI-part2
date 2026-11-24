from dotenv import load_dotenv
load_dotenv()
from langchain_mistralai import ChatMistralAI
from langchain.tools import tool 
from langchain_core.messages import HumanMessage


@tool
def get_text_length(text: str) -> int:
    """Return the number of characters in the given text."""
    return len(text)

tools = [get_text_length]
tools_by_name = {registered_tool.name: registered_tool for registered_tool in tools}
llm = ChatMistralAI(model="mistral-small-2506")

llm_with_tool = llm.bind_tools(tools)

messages = []
prompt = input("You: ")
messages.append(HumanMessage(content=prompt))

while True:
    result = llm_with_tool.invoke(messages)
    messages.append(result)

    if not result.tool_calls:
        break

    for tool_call in result.tool_calls:
        tool = tools_by_name.get(tool_call["name"])
        if tool is None:
            raise ValueError(f"Model requested an unknown tool: {tool_call['name']}")
        messages.append(tool.invoke(tool_call))

print(result.content)