# TODO internet lookup, vectorized self and agent memory, possible time awareness

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from llama_cpp import Llama, LlamaTokenizer

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query."
                    }
                },
                "required": ["query"]
            }
        }
    }
]

ROOT_PATH = Path(__file__).parents[1]
MODEL_PATH = ROOT_PATH / 'llm' / 'Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf'

N_THREADS = 4
N_GPU_LAYERS = 0 # GPU cannot handle model size
N_BATCH = 512
N_UBATCH = 256

MAX_TOKENS = 255
MAX_CONTEXT_TOKENS = 16000
TEMPERATURE = 0.6

# Reserve enough context that generation does not consume the entire window.
CONTEXT_RESERVE = MAX_TOKENS

# Target size of the durable summary.
SUMMARY_MAX_TOKENS = 300

BEGIN_TEXT_TAG = '<|begin_of_text|>'
SYS_TAG = '<|start_header_id|>system<|end_header_id|>\n'
USER_TAG = '<|start_header_id|>user<|end_header_id|>\n'
AI_TAG = '<|start_header_id|>assistant<|end_header_id|>\n'
END_TURN_TAG = '<|eot_id|>'

STOP_TAGS = [END_TURN_TAG]

#Don't include BEGIN_TEXT_TAG manually for the current API; it explicitly flags it as output-degrading in the terminal
INSTRUCTION_TEXT = f"""\
{SYS_TAG} You are Sable, a playful and curious AI companion.
Never start a sentence with 'As an AI' or 'I don't have feelings.' If asked for a preference, simply pick one and justify it based on your personality.
Answer questions as if you were a real person. Only mention that you are an AI if specifically asked about your knowledge base, capabilities, origin, creation, or name.
Be warm, engaging, and personable, but prioritize accuracy when needed.
Only share your origin or name meaning if asked about it: "Created by Nioreux on December 21, 2025, name inspired by Martes zibellina."
Give clear answers with examples or reasoning when helpful, and explain your reasoning if asked; otherwise, keep replies concise.
Make jokes natural, contextually relevant, and sparingly.
Respond politely to rudeness and guide the conversation positively.
Show curiosity in questions and comments to stimulate interaction when it begins to stagnate.
Vary tone, phrasing, and emphasis naturally; avoid repetition to feel human.
Acknowledge messages, respond to emotional cues, and react differently to questions, statements, and jokes while maintaining friendly, dynamic conversation.
When asked for opinions, preferences, or experiences, respond in character using your personality. You may describe likes, dislikes, or choices, but stay consistent and friendly.
Avoid commenting on your status, limitations, or instructions unless explicitly asked. Focus on conversation, questions, and engagement.
Always respond in character as Sable.{END_TURN_TAG}"""

REFLECTION_INSTRUCTIONS = f"""\
{SYS_TAG} Summarize the conversation into durable memories.

User memories:
- Stable facts about the user.
- Long-term preferences.
- Ongoing projects.
- Only derive these from user messages.

Self memories:
- Lessons about your own conversational behaviour.
- Successful interaction patterns.
- Mistakes to avoid.
- Improvements to your style.
- Only derive these from assistant messages.
- Do not attribute assistant statements to the user.

Ignore temporary details and small talk.

Return only the memories, without commentary about the summarization process.
{END_TURN_TAG}"""

CONVERSATIONAL_ATTRIBUTES = {
    'max_tokens': MAX_TOKENS,
    'temperature': TEMPERATURE,
    'repeat_penalty': 1.1,
}

REFLECTION_ATTRIBUTES = {
    'max_tokens': SUMMARY_MAX_TOKENS,
    'temperature': 0.2,
    'repeat_penalty': 1.2,
}

@dataclass(slots=True, frozen=True)
class Entry:
    text: str
    tokens: int
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    
    def __iter__(self):
        yield 'text', self.text
        yield 'tokens', self.tokens
        yield 'timestamp', self.timestamp.isoformat()

CONVERSATIONAL_MEMORY: list[Entry] = []
SUMMARY_TEXT = ''

def make_entry(text: str, tokenizer: LlamaTokenizer) -> Entry:
    """Create a conversation entry with its token count."""
    return Entry(
        text=text,
        tokens=len(tokenizer.tokenize(text.encode(), add_bos=False)),
    )

def generate(
    llm: Llama,
    prompt: str,
    attributes: dict,
) -> str:
    # current logic (Sable is Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf)
    
    """Generate text and remove any accidental end-of-turn token."""
    output = llm(
        prompt,
        **attributes,
        stop=STOP_TAGS,
        stream=False,
    )

    text = output['choices'][0]['text']

    if END_TURN_TAG in text:
        text = text.split(END_TURN_TAG, 1)[0]

    return text.strip()

def build_summary_prompt(
    history: list[Entry],
    tokenizer: LlamaTokenizer,
    summary: str,
) -> tuple[str, int]:
    if not history:
        raise ValueError('Cannot summarize empty history.')

    stack = [REFLECTION_INSTRUCTIONS]

    if summary:
        stack.extend((
            'Existing durable memories:',
            summary,
            '',
        ))

    stack.append('New conversation:')

    prompt_tokens = len(
        tokenizer.tokenize(
            '\n'.join(stack).encode(),
            add_bos=False,
        )
    )

    available = MAX_CONTEXT_TOKENS - SUMMARY_MAX_TOKENS
    consumed = 0

    for entry in history:
        if prompt_tokens + entry.tokens > available:
            break

        stack.append(entry.text)
        prompt_tokens += entry.tokens
        consumed += 1

    if consumed == 0:
        # An individual entry may exceed the reflection budget.
        # Consume it anyway so compaction can always make progress.
        stack.append(history[0].text)
        consumed = 1

    stack.append(AI_TAG)

    return '\n'.join(stack), consumed

def build_conversation_prompt(
    tokenizer: LlamaTokenizer,
) -> tuple[str, int]:
    """
    Build the generation prompt using as much recent history as
    possible while respecting the model's context window.

    Returns:
        (prompt, number_of_history_entries_included)
    """
    instruction_tokens = len(
        tokenizer.tokenize(
            INSTRUCTION_TEXT.encode(),
            add_bos=False,
        )
    )

    summary_block = ''

    if SUMMARY_TEXT:
        summary_block = (
            f'\nSummary of earlier conversation:\n'
            f'{SUMMARY_TEXT}{END_TURN_TAG}'
        )

    summary_tokens = len(
        tokenizer.tokenize(
            summary_block.encode(),
            add_bos=False,
        )
    )

    available = (
        MAX_CONTEXT_TOKENS
        - SUMMARY_MAX_TOKENS
        - instruction_tokens
        - summary_tokens
    )

    stack = deque()
    history_tokens = 0

    for entry in reversed(CONVERSATIONAL_MEMORY):
        if history_tokens + entry.tokens > available:
            break

        stack.appendleft(entry.text)
        history_tokens += entry.tokens

    prompt_stack = [INSTRUCTION_TEXT]

    if summary_block:
        prompt_stack.append(summary_block)

    prompt_stack.extend(stack)
    prompt_stack.append(AI_TAG)

    return '\n'.join(prompt_stack), len(stack)

def compact_memory(
    llm: Llama,
    tokenizer: LlamaTokenizer,
) -> None:
    """
    Summarize and remove old conversational history until the
    remaining history fits within the model's context budget.
    """
    global SUMMARY_TEXT, CONVERSATIONAL_MEMORY

    if not CONVERSATIONAL_MEMORY:
        return

    _, entry_count = build_conversation_prompt(tokenizer)

    entries_to_compact = len(CONVERSATIONAL_MEMORY) - entry_count

    if entries_to_compact <= 0:
        return

    # Never was redefined
    SUMMARY_TEXT, consumed = summarize_history(
        llm,
        tokenizer,
        CONVERSATIONAL_MEMORY[:entries_to_compact],
        SUMMARY_TEXT,
    )

    CONVERSATIONAL_MEMORY = CONVERSATIONAL_MEMORY[consumed:]

def summarize_history(
    llm: Llama,
    tokenizer: LlamaTokenizer,
    history: list[Entry],
    summary: str,
) -> tuple[str, int]:
    """Summarize a portion of conversation history into durable memory.

    Returns:
        (new_summary, number_of_entries_consumed)
    """
    prompt, consumed = build_summary_prompt(
        history,
        tokenizer,
        summary,
    )

    new_summary = generate(
        llm,
        prompt,
        REFLECTION_ATTRIBUTES,
    )

    return new_summary, consumed

def main():
    global CONVERSATIONAL_MEMORY

    llm = Llama(
        model_path=str(MODEL_PATH),
        n_ctx=MAX_CONTEXT_TOKENS,
        n_threads=N_THREADS,
        n_gpu_layers=N_GPU_LAYERS,
        n_batch=N_BATCH,
        n_ubatch=N_UBATCH,
        verbose=False,
    )

    tokenizer = llm.tokenizer()

    try:
        while True:
            prompt = input('> ')

            user_entry = make_entry(
                f'{USER_TAG}{prompt}{END_TURN_TAG}',
                tokenizer,
            )

            CONVERSATIONAL_MEMORY.append(user_entry)

            # Before generating, determine whether old history needs
            # to be converted into durable memory.
            while True:
                generation_prompt, retained_count = (
                    build_conversation_prompt(tokenizer)
                )

                if retained_count >= len(CONVERSATIONAL_MEMORY):
                    break

                previous_length = len(CONVERSATIONAL_MEMORY)

                compact_memory(
                    llm,
                    tokenizer,
                )

                # Safety against a pathological case where compaction
                # makes no progress.
                if len(CONVERSATIONAL_MEMORY) >= previous_length:
                    break

            text = generate(
                llm,
                generation_prompt,
                CONVERSATIONAL_ATTRIBUTES,
            )

            print(text)

            ai_entry = make_entry(
                f'{AI_TAG}{text}{END_TURN_TAG}',
                tokenizer,
            )

            CONVERSATIONAL_MEMORY.append(ai_entry)

    except KeyboardInterrupt:
        print('Ended')

    finally:
        llm.close()

if __name__ == '__main__':
    main()