import os
from dotenv import load_dotenv
import instructor
import openai
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from pydantic import Field
from atomic_agents.lib.components.system_prompt_generator import SystemPromptGenerator
from atomic_agents.agents.base_agent import BaseAgent, BaseAgentConfig
from atomic_agents.lib.base.base_io_schema import BaseIOSchema

# Carica le variabili dal file .env
load_dotenv()

# API Key setup
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise ValueError(
        "API key is not set. Please create a .env file with OPENAI_API_KEY=your_key_here"
    )

# Define schemas with MANDATORY docstrings
class ChatInputSchema(BaseIOSchema):
    """
    Input schema for chat messages from the user.
    """
    chat_message: str = Field(..., description="The user's input message")

class ChatOutputSchema(BaseIOSchema):
    """
    Output schema for chat messages from the agent.
    """
    chat_message: str = Field(..., description="The agent's response message")

# Initialize console
console = Console()

# OpenAI client setup
client = instructor.from_openai(openai.OpenAI(api_key=API_KEY))
print(ChatOutputSchema.__doc__)

# Agent setup
agent = BaseAgent(
    config=BaseAgentConfig(
        client=client,
        model="gpt-4o-mini",
        input_schema=ChatInputSchema,
        output_schema=ChatOutputSchema,
        system_prompt_generator=SystemPromptGenerator(
            background=["You are a helpful AI assistant."],
            steps=["Listen carefully to the user", "Provide clear and helpful responses"],
            output_instructions=["Be concise and friendly"]
        )
    )
)

# Display system prompt
system_prompt = agent.system_prompt_generator.generate_prompt()
console.print(Panel(system_prompt, width=console.width, style="bold cyan"))

# Initial greeting
console.print(Text("Agent:", style="bold green"), end=" ")
console.print(Text("Hello! How can I assist you today?", style="bold green"))

# Chat loop
while True:
    user_input = console.input("[bold blue]You:[/bold blue] ")
    
    if user_input.lower() in ["/exit", "/quit"]:
        console.print("Exiting chat...")
        break
    
    try:
        input_msg = ChatInputSchema(chat_message=user_input)
        response = agent.run(input_msg)
        
        console.print(Text("Agent:", style="bold green"), end=" ")
        console.print(Text(response.chat_message, style="bold green"))
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
