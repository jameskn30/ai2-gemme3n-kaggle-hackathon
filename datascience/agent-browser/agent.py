
from browser_use.llm import ChatOpenAI, ChatOllama
from browser_use import Agent, BrowserSession
from dotenv import load_dotenv
from ollama import Client
import sys
import os 
load_dotenv()
import asyncio

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OLLAMA_HOST = "http://localhost:11434"
LAMINAR_PROJECT_KEY= os.environ.get("LAMINAR_PROJECT_KEY")
print("LAMINAR_PROJECT_KEY ", LAMINAR_PROJECT_KEY)

# MODEL= "deepseek-r1:latest"  # Change this to your desired model
MODEL= "llama3.1:8b"  # Change this to your desired model
# MODEL= "gemma3n:e4b"  # Change this to your desired model
# MODEL= "gemma3:12b"  # Change this to your desired model
LOG_OUTPUT_DIR = f"logs/conversation/{MODEL}"

# SETUP Laminar
from lmnr import Laminar

Laminar.initialize(project_api_key=LAMINAR_PROJECT_KEY)


#Checking 
client = Client(host = OLLAMA_HOST)

try:
    for model in client.list() :
        print(model)

    print("Connected successfully!")
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)

browser_session = BrowserSession(
    window_size={'width': 800, 'height': 600},
)

initial_actions = [
    {'go_to_url': {'url': 'https://www.google.com', 'new_tab': True}},
]


# agent = Agent(
#     task="Check the weather in Miami Florida",
#     llm = ChatOpenAI(model="gpt-4.1", api_key=OPENAI_API_KEY),
#     initial_actions=initial_actions,
# )

# message_context = "Don't go to images, only search for weather information using default settings "
# task_prompt = "Search google for weather information in Miami Florida"
task_prompt = "Search google for weather information in California"

llm = ChatOllama(
    host=OLLAMA_HOST,
    model=MODEL)

agent = Agent(
    task=task_prompt,
    save_conversation_path=LOG_OUTPUT_DIR,
    llm=llm,
    #   planner_llm=planner_llm, use this to come up with high-level plan
    #   use_vision_for_planning=false,  # Set to True if you want to use vision for planning
    #   planner_interval=1,  # Set to True if you want to use vision for planning
    initial_actions=initial_actions,
    browser_session=browser_session,
)

# agent = Agent(
#     browser_session=browser_session,
#     task="Check the weather in Miami Florida",
#     llm = ChatOpenAI(model="gpt-4.1", api_key=OPENAI_API_KEY),
#     initial_actions=initial_actions,
#     save_conversation_path='logs/conversation/gpt',
# )


# Run the agent
async def main():
    try:
        result = await agent.run()
    except Exception as e:
        print(f"An error occurred: {e}")
        # Example of accessing history
        history = await agent.run()

        # # Access (some) useful information
        # history.urls()              # List of visited URLs
        # history.screenshots()       # List of screenshot paths
        # history.action_names()      # Names of executed actions
        # history.extracted_content() # Content extracted during execution
        # history.errors()           # Any errors that occurred
        # history.model_actions()     # All actions with their parameters

    # print(history)

asyncio.run(main())
# await main()