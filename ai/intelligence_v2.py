from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from llama_cpp import Llama, LlamaTokenizer
import sympy

from db.memory_store import MemoryStore
from net.search import QuerySummarizer

MATH_TOOL_META = {
    "type": "function",
    "function": {
        "name": "arithmetic",
        "description": (
            "Evaluate a mathematical expression and return the exact result. "
            "Use this for arithmetic, numerical calculations, and unit-free "
            "mathematical expressions instead of calculating mentally."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "A mathematical expression to evaluate.",
                },
            },
            "required": ["expression"],
        },
    },
}

SEARCH_TOOL_META = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web for information relevant to the user's question.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query.",
                },
            },
            "required": ["query"],
        },
    },
}

READ_FILE_TOOL_META = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": (
            "Read text from a file that has been made available to you."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "file_id": {
                    "type": "string",
                    "description": "The identifier of the available file.",
                },
            },
            "required": ["file_id"],
        },
    },
}

RECALL_MEMORY_META = {
    "type": "function",
    "function": {
        "name": "recall_memory",
        "description": (
            "Search long-term memories for information relevant to the "
            "current conversation."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What information to recall.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of memories to retrieve.",
                },
            },
            "required": ["query"],
        },
    },
}

embedder = None # TODO
MEMORY_STORE = MemoryStore(embedder)

def recall_memory(query: str, limit: int = 5) -> str:
    results = MEMORY_STORE.search(query, limit=limit)

    return "\n\n".join(
        f'{result.memory.timestamp} {result.memory.text}'
        for result in results
    )

def read_file(
    file_id: str,
    *,
    tokenizer: LlamaTokenizer,
    max_tokens: int = 2000,
) -> str:
    """Read a text file, truncating at max_tokens."""

    path = Path(file_id)

    if not path.exists():
        return f"No file exists at {file_id}"

    if not path.is_file():
        return f"{file_id} is a directory, not a file"

    lines: list[str] = []
    token_count = 0

    with path.open(encoding="utf-8") as file:
        for line in file:
            line_tokens = tokenizer.tokenize(
                line.encode("utf-8"),
                add_bos=False,
            )
            line_token_count = len(line_tokens)

            remaining = max_tokens - token_count

            if line_token_count <= remaining:
                lines.append(line)
                token_count += line_token_count
                continue

            if remaining > 0:
                tokens = line_tokens[:remaining]

                lines.append(
                    tokenizer.detokenize(tokens).decode(
                        "utf-8",
                        errors="ignore",
                    )
                )

            break

    return "".join(lines)

def web_search(query: str) -> str:
    # TODO dynamically build on first web_search and tear_down after last tool call (cache-like behaviour)
    
    sub_llm = Llama(
        model_path="../llm/microsoft_Phi-4-mini-instruct-Q4_K_M.gguf", 
        n_ctx=16000, 
        n_threads=4,
        n_gpu_layers=18, # 15 to 20
        n_batch=512,
        n_ubatch=256,
        verbose=False
    )
    
    result = QuerySummarizer(sub_llm)(query)
    
    sub_llm.close()
    
    return result

TOOL_HANDLERS = {
    "web_search": web_search,
    "recall_memory": recall_memory,
    "arithmetic": arithmetic,
    "read_file": read_file
}

TOOLS = [
    MATH_TOOL_META,
    SEARCH_TOOL_META,
    READ_FILE_TOOL_META,
    RECALL_MEMORY_META,
]

ROOT_PATH = Path(__file__).parents[1]
MODEL_PATH = ROOT_PATH / "llm" / "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"

N_THREADS = 4
N_GPU_LAYERS = 0 # GPU cannot handle model size
N_BATCH = 512
N_UBATCH = 256

MAX_TOKENS = 255
MAX_CONTEXT_TOKENS = 16000
TEMPERATURE = 0.6

SUMMARY_MAX_TOKENS = 300

SYSTEM_PROMPT = """
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

REFLECTION_PROMPT = """
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

CONVERSATIONAL_ATTRIBUTES = {
    "max_tokens": MAX_TOKENS,
    "temperature": 0.6,
    "repeat_penalty": 1.055,
}

REFLECTION_ATTRIBUTES = {
    "max_tokens": SUMMARY_MAX_TOKENS,
    "temperature": 0.2,
    "repeat_penalty": 1.2,
}

@dataclass(slots=True, frozen=True)
class Entry:
    role: str
    text: str
    tokens: int
    timestamp: datetime = field(
    default_factory=lambda: datetime.now(timezone.utc)
)

CONVERSATIONAL_MEMORY: list[Entry] = []
SUMMARY_TEXT = ""

def make_entry(
    role: str,
    text: str,
    llm: Llama,
) -> Entry:
    """Create a conversation entry with its actual token count."""

    tokens = len(
        llm.tokenize(
            text.encode("utf-8"),
            add_bos=False,
        )
    )

    return Entry(
        role=role,
        text=text,
        tokens=tokens,
    )

def entry_to_message(entry: Entry) -> dict[str, str]:
    """Convert an internal history entry into a chat-completion message."""

    return {
        "role": entry.role,
        "content": entry.text,
    }

def generate(
    llm: Llama,
    messages: list[dict[str, Any]],
    attributes: dict[str, Any],
    *,
    tools: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Generate a chat completion."""

    kwargs = {
        **attributes,
        "messages": messages,
        "stream": False,
    }

    if tools is not None:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"

    return llm.create_chat_completion(**kwargs)

def build_summary_prompt(
    history: list[Entry],
    llm: Llama,
    summary: str,
    ) -> tuple[list[dict[str, str]], int]:
    """Build the reflection conversation."""

    if not history:
        raise ValueError("Cannot summarize empty history.")

    messages: list[dict[str, str]] = [
        {
            "role": "system",
            "content": REFLECTION_PROMPT,
        },
    ]

    if summary:
        messages.append(
            {
                "role": "system",
                "content": (
                    "Existing durable memories:\n"
                    f"{summary}"
                ),
            }
        )

    messages.append(
        {
            "role": "user",
            "content": "New conversation:",
        }
    )

    prompt_tokens = sum(
        len(
            llm.tokenize(
                message["content"].encode("utf-8"),
                add_bos=False,
            )
        )
        for message in messages
    )

    available = MAX_CONTEXT_TOKENS - SUMMARY_MAX_TOKENS
    consumed = 0

    for entry in history:
        if prompt_tokens + entry.tokens > available:
            break

        messages.append(entry_to_message(entry))
        prompt_tokens += entry.tokens
        consumed += 1

    if consumed == 0:
        # Ensure compaction always makes progress.
        messages.append(entry_to_message(history[0]))
        consumed = 1

    messages.append(
        {
            "role": "assistant",
            "content": "",
        }
    )

    return messages, consumed

def build_conversation_messages(
    llm: Llama,
) -> tuple[list[dict[str, Any]], int]:
    """
    Build the generation messages using as much recent history as
    possible while respecting the model's context window.

    Returns:
        (messages, number_of_history_entries_included)
    """

    system_tokens = len(
        llm.tokenize(
            SYSTEM_PROMPT.encode("utf-8"),
            add_bos=False,
        )
    )

    summary_tokens = 0

    if SUMMARY_TEXT:
        summary_tokens = len(
            llm.tokenize(
                SUMMARY_TEXT.encode("utf-8"),
                add_bos=False,
            )
        )

    available = (
        MAX_CONTEXT_TOKENS
        - MAX_TOKENS
        - SUMMARY_MAX_TOKENS
        - system_tokens
        - summary_tokens
    )

    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
    ]

    if SUMMARY_TEXT:
        messages.append(
            {
                "role": "system",
                "content": (
                    "Summary of earlier conversation:\n"
                    f"{SUMMARY_TEXT}"
                ),
            }
        )

    history: deque[Entry] = deque()
    history_tokens = 0

    for entry in reversed(CONVERSATIONAL_MEMORY):
        if history_tokens + entry.tokens > available:
            break

        history.appendleft(entry)
        history_tokens += entry.tokens

    messages.extend(
        entry_to_message(entry)
        for entry in history
    )

    return messages, len(history)

def compact_memory(
    llm: Llama,
) -> None:
    """Summarize old conversational history into durable memory."""

    global SUMMARY_TEXT, CONVERSATIONAL_MEMORY

    if not CONVERSATIONAL_MEMORY:
        return

    _, retained_count = build_conversation_messages(llm)

    entries_to_compact = (
        len(CONVERSATIONAL_MEMORY) - retained_count
    )

    if entries_to_compact <= 0:
        return

    SUMMARY_TEXT, consumed = summarize_history(
        llm,
        CONVERSATIONAL_MEMORY[:entries_to_compact],
        SUMMARY_TEXT,
    )

    CONVERSATIONAL_MEMORY = CONVERSATIONAL_MEMORY[consumed:]

def summarize_history(
    llm: Llama,
    history: list[Entry],
    summary: str,
) -> tuple[str, int]:
    """Convert old conversation history into durable memory."""

    messages, consumed = build_summary_prompt(
        history,
        llm,
        summary,
    )

    response = generate(
        llm,
        messages,
        REFLECTION_ATTRIBUTES,
    )

    text = response["choices"][0]["message"]["content"]

    return text.strip(), consumed

def handle_tool_call(
    messages: list[dict[str, Any]],
    tool_call: dict[str, Any],
    tokenizer: LlamaTokenizer,
) -> None:
    """Execute one model-requested tool and append its result."""

    function = tool_call["function"]

    name = function["name"]
    arguments = json.loads(function["arguments"])

    handler = TOOL_HANDLERS.get(name)

    if handler is None:
        result = f"Unknown tool: {name}"
    else:
        try:
            if name == "read_file":
                result = handler(
                    **arguments,
                    tokenizer=tokenizer,
                )
            else:
                result = handler(**arguments)

        except Exception as exc:
            result = f"Tool execution failed: {exc}"

    messages.append(
        {
            "role": "tool",
            "tool_call_id": tool_call["id"],
            "content": str(result),
        }
    )

def chat(
    llm: Llama,
    tokenizer: LlamaTokenizer,
    prompt: str,
) -> str:
    """
    Process one user message.

    The model may either answer normally or request one or more tools.
    Tool results are fed back into the model until it produces a final answer.
    """

    global CONVERSATIONAL_MEMORY

    user_entry = make_entry(
        "user",
        prompt,
        llm,
    )

    CONVERSATIONAL_MEMORY.append(user_entry)

    while True:
        generation_messages, retained_count = (
            build_conversation_messages(llm)
        )

        if retained_count >= len(CONVERSATIONAL_MEMORY):
            break

        previous_length = len(CONVERSATIONAL_MEMORY)

        compact_memory(llm)

        if len(CONVERSATIONAL_MEMORY) >= previous_length:
            break

    while True:
        response = generate(
            llm,
            generation_messages,
            CONVERSATIONAL_ATTRIBUTES,
            tools=TOOLS,
        )

        message = response["choices"][0]["message"]
        tool_calls = message.get("tool_calls")

        if not tool_calls:
            break

        # Preserve the assistant's tool request in the conversation
        # sent back to the model.
        generation_messages.append(message)

        for tool_call in tool_calls:
            handle_tool_call(
                generation_messages,
                tool_call,
                tokenizer,
            )

    text = (message.get("content") or "").strip()

    ai_entry = make_entry(
        "assistant",
        text,
        llm,
    )

    CONVERSATIONAL_MEMORY.append(ai_entry)

    return text

def main() -> None:
    llm = Llama(
        model_path=str(MODEL_PATH),
        n_ctx=MAX_CONTEXT_TOKENS,
        n_threads=N_THREADS,
        n_gpu_layers=N_GPU_LAYERS,  # GPU cannot handle model size
        n_batch=N_BATCH,
        n_ubatch=N_UBATCH,
        verbose=False,
    )

    tokenizer = llm.tokenizer()

    try:
        while True:
            prompt = input("> ")
            print(chat(llm, tokenizer, prompt))

    except KeyboardInterrupt:
        print("Ended")

    finally:
        llm.close()

if __name__ == "__main__":
    main()