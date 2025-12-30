from pydantic import Field
from openai import OpenAI
import instructor

# atomic_agents
try:
    from atomic_agents import (
        AtomicAgent,
        AgentConfig,
        BasicChatInputSchema,
        BaseIOSchema,
    )
    from atomic_agents.context import SystemPromptGenerator, ChatHistory
except ImportError as e:
    raise ImportError(
        "Pacchetto atomic-agents non trovato. "
        "Assicurati di averlo installato con 'poetry add atomic-agents'."
    ) from e

# -------------------------------------------------
# Custom output schema
# -------------------------------------------------
class CustomOutputSchema(BaseIOSchema):
    """
    Schema per output dell'agent con messaggio e follow-up
    """
    chat_message: str = Field(..., description="Il messaggio di risposta dell'agente")
    suggested_questions: list[str] = Field(
        ..., description="Domande di follow-up suggerite dall'agente"
    )

    model_config = {
        "title": "CustomOutputSchema",
        "description": "Output schema per chat agent con messaggio e follow-up"
    }

# -------------------------------------------------
# Prompt generator
# -------------------------------------------------
system_prompt_generator = SystemPromptGenerator(
    background=["Assistente esperto che usa documentazione locale"],
    steps=[
        "Leggi il contesto",
        "Rispondi accuratamente",
        "Suggerisci follow-up"
    ],
    output_instructions=["Non inventare informazioni"]
)

# -------------------------------------------------
# OpenAI client via instructor
# -------------------------------------------------
client = instructor.from_openai(OpenAI())

# -------------------------------------------------
# Agent instance
# -------------------------------------------------
agent = AtomicAgent[
    BasicChatInputSchema,
    CustomOutputSchema
](
    config=AgentConfig(
        client=client,
        model="gpt-5-mini",
        system_prompt_generator=system_prompt_generator,
        history=ChatHistory(),
    )
)
