from dataclasses import dataclass

@dataclass
class AIRequest:
    user_message: str
    conversation_history: str
    company_context: str
    system_prompt: str