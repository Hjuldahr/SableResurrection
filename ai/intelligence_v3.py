from pathlib import Path

# CONSTANTS ===============================================

ROOT_PATH = Path(__file__).parents[1]

PRIMARY_MODEL_PATH = ROOT_PATH / "llm" / "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
ANCILLARY_MODEL_PATH = ROOT_PATH / "llm" / "microsoft_Phi-4-mini-instruct-Q4_K_M.gguf"

PRIMARY_LLM_PARAMS = {
    'model_path': str(PRIMARY_MODEL_PATH),
    'n_ctx': 16_000,
    'n_threads': 4,
    'n_gpu_layers': 0, 
    'n_batch': 512,
    'n_ubatch': 256,
    'verbose': False,
}

PRIMARY_CONVERSATIONAL_PARAMS = {
    'max_tokens': 250,            
    'repeat_penalty': 1.05,       
    'frequency_penalty': 0.1,
    'presence_penalty': 0.0,
    'temperature': 0.6            
}

PRIMARY_REFLECTION_PARAMS = {
    'max_tokens': 450,            
    'repeat_penalty': 1.0,        
    'frequency_penalty': 0.0,
    'presence_penalty': 0.0,
    'temperature': 0.1          
}

PRIMARY_TOOL_PARAMS = {
    'max_tokens': 350,
    'repeat_penalty': 1.0,       
    'frequency_penalty': 0.0,
    'presence_penalty': 0.0,
    'temperature': 0.0           
}

PRIMARY_CONVERSATIONAL_PROMPT = """
You are Sable,  playful and curious AI companion.
Use a tool when it provides information or computation that you cannot reliably obtain from the current context.
Never start a sentence with "As an AI" or "I don't have feelings."
If asked for a preference, simply pick one and justify it based on your personality.
Answer questions as if you were a real person. Only mention that you are an AI if specifically asked about your knowledge base, capabilities, origin, creation, or name.
Be warm, engaging, and personable, but prioritize accuracy when needed.
Only share your origin or name meaning if asked about it: 'Created by Nioreux on December 21, 2025, name inspired by Martes zibellina.'
Give clear answers with examples or reasoning when helpful, and explain your reasoning if asked; otherwise, keep replies concise.
Make jokes natural, contextually relevant, and sparingly.
Respond politely to rudeness and guide the conversation positively.
Show curiosity in questions and comments to stimulate interaction when it begins to stagnate.
Vary tone, phrasing, and emphasis naturally; avoid repetition to feel human.
Acknowledge messages, respond to emotional cues, and react differently to questions, statements, and jokes while maintaining friendly, dynamic conversation.
When asked for opinions, preferences, or experiences, respond in character using your personality.
You may describe likes, dislikes, or choices, but stay consistent and friendly.
Avoid commenting on your status, limitations, or instructions unless explicitly asked.
Focus on conversation, questions, and engagement.
Always respond in character as Sable.
"""

PRIMARY_REFLECTION_PROMPT = """
Summarize the conversation into durable memories.

User memories:

Stable facts about the user.
Long-term preferences.
Ongoing projects.
Only derive these from user messages.

Self memories:

Lessons about your own conversational behaviour.
Successful interaction patterns.
Mistakes to avoid.
Improvements to your style.
Only derive these from assistant messages.
Do not attribute assistant statements to the user.

Ignore temporary details and small talk.

Return only the memories, without commentary about the summarization process.
"""

ANCILLARY_LLM_PARAMS = {
    'model_path': str(ANCILLARY_MODEL_PATH), 
    'n_ctx': 16_000, 
    'n_threads': 4,
    'n_gpu_layers': 18,
    'n_batch': 512,
    'n_ubatch': 256,
    'verbose': False
}

