
from dotenv import load_dotenv
load_dotenv()

import os
import requests

from langchain_mistralai import ChatMistralAI
from tavily import TavilyClient
from langchain.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage


# ==========================================
# 1. API KEY VALIDATION
# ==========================================

required_keys = [
    "MISTRAL_API_KEY",
    "TAVILY_API_KEY",
    "OPENWEATHER_API_KEY"
]

for key in required_keys:
    if not os.getenv(key):
        raise ValueError(f"Missing API key: {key}")

tavily_client = TavilyClient(
    api_key=os.getenv("TAVILY_API_KEY")
)


# ==========================================
# 2. WEATHER TOOL
# ==========================================

@tool
def get_weather(city: str) -> str:
    """Get current weather of an Indian city."""

    api_key = os.getenv("OPENWEATHER_API_KEY")

    url = "https://api.openweathermap.org/data/2.5/weather"

    params = {
        "q": f"{city},IN",
        "appid": api_key,
        "units": "metric"
    }

    try:
        response = requests.get(
            url,
            params=params,
            timeout=10
        )

        data = response.json()

        if response.status_code != 200:
            return f"Weather API Error: {data.get('message', 'Unknown error')}"

        temp = data["main"]["temp"]
        desc = data["weather"][0]["description"]
        humidity = data["main"]["humidity"]

        return (
            f"Weather in {city}:\n"
            f"Temperature: {temp}°C\n"
            f"Condition: {desc}\n"
            f"Humidity: {humidity}%"
        )

    except requests.RequestException as e:
        return f"Weather request failed: {str(e)}"


# ==========================================
# 3. NEWS TOOL
# ==========================================

@tool
def get_news(city: str) -> str:
    """Get latest news about a city using Tavily."""

    try:
        response = tavily_client.search(
            query=f"latest news in {city}",
            search_depth="basic",
            max_results=3
        )

        results = response.get("results", [])

        if not results:
            return f"No news found for {city}."

        news_list = []

        for r in results:
            title = r.get("title", "No title")
            url = r.get("url", "")
            snippet = r.get("content", "")

            news_list.append(
                f"Title: {title}\n"
                f"URL: {url}\n"
                f"Summary: {snippet[:200]}"
            )

        return f"Latest news in {city}:\n\n" + "\n\n".join(news_list)

    except Exception as e:
        return f"News search failed: {str(e)}"


# ==========================================
# 4. LLM SETUP
# ==========================================

llm = ChatMistralAI(
    model="mistral-small-2506",
    temperature=0
)

tools = {
    "get_weather": get_weather,
    "get_news": get_news
}

llm_with_tools = llm.bind_tools(list(tools.values()))


# ==========================================
# 5. AGENT LOOP
# ==========================================

messages = []

print("\nCity Intelligence System")
print("Powered by Mistral AI + LangChain + Tavily")
print("Type 'exit' to quit.\n")


while True:

    user_input = input("User: ").strip()

    if user_input.lower() == "exit":
        print("Goodbye!")
        break

    if not user_input:
        continue

    messages.append(HumanMessage(content=user_input))

    # Agent reasoning loop
    while True:

        try:
            result = llm_with_tools.invoke(messages)

        except Exception as e:
            print(f"LLM Error: {e}")
            break

        # Store AI response
        messages.append(result)

        # Check whether LLM wants to call tools
        if result.tool_calls:

            for tool_call in result.tool_calls:

                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                tool_id = tool_call["id"]

                print(f"\nAI wants to execute: {tool_name}")
                print(f"Arguments: {tool_args}")

                # Validate tool name
                if tool_name not in tools:

                    tool_result = f"Unknown tool: {tool_name}"

                else:

                    # Human in the Loop
                    confirm = input(
                        f"Execute '{tool_name}'? (yes/no): "
                    ).strip().lower()

                    if confirm == "yes":

                        try:
                            tool_result = tools[tool_name].invoke(
                                tool_args
                            )

                            print(
                                f"\nTool Result:\n{tool_result}\n"
                            )

                        except Exception as e:
                            tool_result = f"Tool execution failed: {str(e)}"

                    else:

                        tool_result = "Tool execution skipped by user."

                        print("Tool execution skipped.")

                # Send tool result back to LLM
                messages.append(
                    ToolMessage(
                        content=str(tool_result),
                        tool_call_id=tool_id
                    )
                )

        else:

            # Final AI response
            print(f"\nAI: {result.content}\n")
            break
