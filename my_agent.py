from pdb import set_trace as bp
from langchain_openai import ChatOpenAI
from cdp_langchain.agent_toolkits import CdpToolkit
from cdp_langchain.utils import CdpAgentkitWrapper
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

from lib.nildb import NilDbUploadTool, NilDbDownloadTool, NilDbSchemaCreateTool, NilDbSchemaLookupTool

load_dotenv()

PROMPT = """
Write a poem about the beauty of transformation and renewal, drawing inspiration from nature (e.g., seasons, butterflies, or rivers). The poem should follow these criteria:

Structure: Four quatrains (four-line stanzas).
Meter: Iambic tetrameter (four iambic feet per line).
Rhyme Scheme: ABAB for each stanza.
Content Requirements: Include vivid imagery of a natural process of change, such as leaves falling and regrowing, or a river carving a new path. Incorporate themes of hope and resilience.
Tone: Reflective and uplifting, with a focus on the positive aspects of transformation.
The poem should use evocative language and focus on painting a clear mental image for the reader. 
Upload the generated poem to nildb.

Any time that you do not have a schema UUID, you should look one up or create a schema using the nildb tools.
"""

# Initialize the LLM
# if you want to support Claude, for example, you can replace this line with llm = ChatAnthropic(model="claude-3-5-sonnet-20240620"), replace the `from langchain_openai...` import with `from langchain_anthropic import ChatAnthropic`, and run in your terminal `export ANTHROPIC_API_KEY="your-api-key"
llm = ChatOpenAI(model="gpt-4o-mini")

nildb_schema_create = NilDbSchemaCreateTool(llm)
nildb_schema_lookup = NilDbSchemaLookupTool(llm)
nildb_upload = NilDbUploadTool()
nildb_download = NilDbDownloadTool()

# Initialize CDP AgentKit wrapper
cdp = CdpAgentkitWrapper()

# Create toolkit from wrapper
cdp_toolkit = CdpToolkit.from_cdp_agentkit_wrapper(cdp)

# Get all available tools
tools = cdp_toolkit.get_tools()

tools.append(nildb_schema_create)
tools.append(nildb_schema_lookup)
tools.append(nildb_upload)
tools.append(nildb_download)

# Create the agent
agent_executor = create_react_agent(llm, tools=tools, state_modifier=PROMPT)

# Function to interact with the agent
def ask_agent(question: str):
    for chunk in agent_executor.stream(
        {"messages": [HumanMessage(content=question)]},
        {"configurable": {"thread_id": "my_first_agent"}},
    ):
        bp()
        if "agent" in chunk:
            print(chunk["agent"]["messages"][0].content)
        elif "tools" in chunk:
            print(chunk["tools"]["messages"][0].content)
        print("-------------------")


# Test the agent
if __name__ == "__main__":
    print("Agent is ready! Type 'exit' to quit.")
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == "exit":
            break
        ask_agent(user_input)
