from langchain_openai import ChatOpenAI
from cdp_langchain.agent_toolkits import CdpToolkit
from cdp_langchain.utils import CdpAgentkitWrapper
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
from langchain_community.chat_models import ChatAnthropic
from dotenv import load_dotenv
import json

from lib.nildb import NilDbUploadTool, NilDbDownloadTool
import pprint

load_dotenv()

# Initialize the LLM
# if you want to support Claude, for example, you can replace this line with llm = ChatAnthropic(model="claude-3-5-sonnet-20240620"), replace the `from langchain_openai...` import with `from langchain_anthropic import ChatAnthropic`, and run in your terminal `export ANTHROPIC_API_KEY="your-api-key"
llm = ChatOpenAI(model="gpt-4o-mini")

import anthropic

client = anthropic.Anthropic()

message = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1000,
    temperature=0,
    system="You are a world-class poe judge.",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Why is the ocean salty?"
                }
            ]
        }
    ]
)
print(message.content)



nildb_upload = NilDbUploadTool()
nildb_download = NilDbDownloadTool()

# Initialize CDP AgentKit wrapper
cdp = CdpAgentkitWrapper()

# Create toolkit from wrapper
cdp_toolkit = CdpToolkit.from_cdp_agentkit_wrapper(cdp)

# Get all available tools
tools = cdp_toolkit.get_tools()

tools.append(nildb_upload)
tools.append(nildb_download)

# Create the agent
agent_executor = create_react_agent(
    llm,
    tools=tools,
    state_modifier="You are a helpful agent that can interact with the Base blockchain using CDP AgentKit. You can create wallets, deploy tokens, and perform transactions.",
)

# Function to interact with the agent
def ask_agent(question: str):
    i = 0
    red_poem = ''
    blue_poem = ''
    for chunk in agent_executor.stream(
        {"messages": [HumanMessage(content="download from nildb")]},
        {"configurable": {"thread_id": "my_first_agent"}},
    ):
        # Download action
        if i == 0:
            pass
        # Agent download action response
        elif i == 1:
            d = json.loads(chunk['tools']['messages'][0].content)
            red_poem = d['red']
            blue_poem = d['blue']
            print('[Red Team Poem]:', red_poem)
            print('[Blue Team Poem]:', blue_poem.strip('\n'))
            break
        i += 1
    print('Judging ...')
    # Judge
    for chunk in agent_executor.stream(
        {"messages": [HumanMessage(content=f"red poem: {red_poem}. blue poem: {blue_poem}. Do you prefer red or blue? Give a score for each.")]},
        {"configurable": {"thread_id": "judge"}},
    ):
        print('Response:', chunk['agent']['messages'][0].content)

# Test the agent
if __name__ == "__main__":
    print("Agent is ready! Type 'exit' to quit.")
    while True:
        user_input = input("\nJudge (y/n):")
        if user_input.lower() in ["n", "exit"]:
            break
        ask_agent(user_input)
