from pydantic import Field
from openai import OpenAI
import instructor

from atomic_agents import (
    AtomicAgent,
    AgentConfig,
    BasicChatInputSchema,
    BaseIOSchema,
)
from atomic_agents.context import SystemPromptGenerator, ChatHistory

class CustomOutputSchema(BaseIOSchema):
    chat_message: str = Field(...)
    suggested_questions: list[str] = Field(...)

system_prompt_generator = SystemPromptGenerator(
    background=["Assistente esperto che usa documentazione locale"],
    steps=[
        "Leggi il contesto",
        "Rispondi accuratamente",
        "Suggerisci follow-up"
    ],
    output_instructions=["Non inventare informazioni"]
)

client = instructor.from_openai(OpenAI())

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
