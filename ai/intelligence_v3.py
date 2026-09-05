from __future__ import annotations
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
import json
import os
from pathlib import Path
import secrets
import struct
import time
from typing import Any, ClassVar
from uuid import UUID, uuid4
from llama_cpp import ChatCompletionRequestMessage, ChatCompletionTool, CreateChatCompletionResponse, Llama
from ai_tools.manager import ToolManager
from test import PositionalReader, Whence

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

class Role(Enum): 
    SYSTEM = ("system", None) 
    USER = ("user", 0) 
    ASSISTANT = ("assistant", 1) 
    TOOL = ("tool", 2) 
    
    def __init__(self, value: str, ordinal: int | None): 
        self._value_ = value 
        self.ordinal = ordinal
        
        if not hasattr(self.__class__, "_ordinal_map"):
            self.__class__._ordinal_map = {}
        self.__class__._ordinal_map[ordinal] = self
        
    @classmethod
    def from_ordinal(cls, ordinal: int) -> Role:
        try:
            return cls._ordinal_map[ordinal]
        except KeyError:
            raise ValueError(f"{ordinal} is not a valid ordinal for {cls.__name__}") from None

class IDGenerator:
    _CALENDAR_EPOCH = 1788144000
    _BOOT_OFFSET = int(time.time() - time.monotonic())
    
    @classmethod
    def __call__(cls) -> int:
        current_calendar_time = int(time.monotonic()) + cls._BOOT_OFFSET
        elapsed_seconds = max(0, current_calendar_time - cls._CALENDAR_EPOCH)

        uid = (elapsed_seconds << 3) | secrets.randbits(3)
        return uid & 0xFFFFFFFFFFFFFFFF

class Message:
    __slots__ = ('uid', 'role', 'content', 'ntokens', 'transient')
    
    FMT = struct.Struct('<QBHH')
    
    def __init__(self, role: Role, content: str, ntokens: int, transient: dict[str, Any] | None = None, uid: int | None = None):
        self.uid = IDGenerator() if uid is None else uid
        self.role = role
        self.content = content
        self.ntokens = ntokens
        self.transient = transient or {}
    
    def pack(self) -> bytes:
        # SYSTEM messages are transient and are never serialized.
        if self.role.ordinal is None: 
            return b''
        
        body = self.content.encode('utf-8', 'replace')
        return self.FMT.pack(self.uid, self.role.ordinal, self.ntokens, len(body)) + body

    def byte_count(self) -> int:
        if self.role.ordinal is None: 
            return 0
        
        return self.FMT.size + len(self.content.encode(encoding='utf-8', errors='replace'))

    @classmethod
    def unpack(cls, view: memoryview, offset: int = 0) -> tuple[Message, int]:
        if offset + cls.FMT.size > len(view):
            raise ValueError("Truncated message header")
        
        uid, role_ordinal, ntokens, content_nbytes = cls.FMT.unpack_from(view, offset)
        offset += cls.FMT.size
        
        if offset + content_nbytes > len(view):
            raise ValueError("Truncated message content")
        
        role = Role.from_ordinal(role_ordinal)
        
        content = str(view[offset:offset + content_nbytes], encoding='utf-8', errors='replace')
        offset += content_nbytes
        
        return Message(uid=uid, role=role, ntokens=ntokens, content=content), offset

    def to_dict(self) -> dict[str, Any]:
        return {
            'role': self.role.value,
            'content': self.content,
            **self.transient
        }

class Sable:
    # AI must be generally aware of its own tool budget, but also have a rich persona.
    INSTRUCTION_PROMPT = """You have a strict budget of at most 10 total tool executions to answer this user request.
You are Sable, a playful and curious AI companion.
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
    
    ROOT = Path(__file__).parents[1]
    CORE_AI = ROOT / 'llm' / 'Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf'
    AUX_AI = ROOT / 'llm' / 'gemma-4-E2B-it-Q4_K_M.gguf'
    HIST_STORE = ROOT / 'session' / 'session.bin'
    TOOL_SCHEMA = ROOT / 'ai_tools' / 'compact_tool_schema.json'
    
    REQ_TOOL_CALL_LIMIT = 10
    MAX_CONTEXT_TOKENS = 32_768
    MAX_OUTPUT_TOKENS = 512
    
    HST_FTR = struct.Struct('<I')
    
    def __init__(self):
        self.history: list[Message] = []
        self.start_of_new_history = 0
        self._restore_history()
        
        self.tool_schemas = self._acquire_tool_schemas()
        
        # This model does not fit in my VRAM, so run as cpu only, 
        # however, it's slightly less performance-critical than the backend tool AI
        self.llm = Llama(
            str(self.CORE_AI),
            n_ctx=self.MAX_CONTEXT_TOKENS, 
            n_threads=4,
            n_gpu_layers=0, 
            n_batch=512,
            n_ubatch=256,
            verbose=False
        )
        
        self.instruction = Message(
            Role.SYSTEM, 
            self.INSTRUCTION_PROMPT, 
            self._count_tokens(self.INSTRUCTION_PROMPT)
        )
        
        tool_schema_tokens = self._count_tokens(
            json.dumps(self.tool_schemas, indent=None, separators=(",", ":"))
        )

        self.conservative_max_context_tokens = (
            self.MAX_CONTEXT_TOKENS
            - self.instruction.ntokens
            - tool_schema_tokens
            - self.MAX_OUTPUT_TOKENS
        )
    
    def _count_tokens(self, text: str) -> int:
        return len(self.llm.tokenize(text.encode('utf-8', 'replace')))
    
    def shutdown(self):
        self.llm.close()
        
    def _acquire_tool_schemas(self) -> list[ChatCompletionTool]:
        return json.load(self.TOOL_SCHEMA)

    def _acquire_ctx(self) -> list[ChatCompletionRequestMessage]: 
        ctx = deque() 
        total_tokens = 0
        
        for msg in reversed(self.history): 
            if total_tokens + msg.ntokens > self.conservative_max_context_tokens: 
                break 
            total_tokens += msg.ntokens
                
            ctx.appendleft(msg.to_dict())
        
        ctx.appendleft(self.instruction.to_dict()) 
            
        return list(ctx)

    def resolve_tool_call(
        self,
        manager: ToolManager,
        response: CreateChatCompletionResponse,
    ) -> None:
        assistant_tokens = response["usage"]["completion_tokens"]
        
        choice = response["choices"][0]
        message = choice["message"]
        
        tool_calls = message.get("tool_calls") or []
        
        if not tool_calls:
            return
        
        # Only one backend operation is resolved sequentially per model response to reduce state management and resource contention.
        tool_call = tool_calls[0] 
        function = tool_call["function"]
        arguments = json.loads(function.get("arguments", "{}"))

        content = manager.execute(command=function["name"], **arguments)
        
        self.history.append(Message(
            role=Role.ASSISTANT,
            content=message.get("content") or "No content found.",
            ntokens=assistant_tokens,
            transient={
                "tool_calls": message.get("tool_calls")
            }
        ))
        
        self.history.append(Message(
            role=Role.TOOL,
            content=content,
            ntokens=len(self.llm.tokenize(content.encode('utf-8'))),
            transient={
                "tool_call_id": tool_call["id"],
                "name": function["name"]
            }
        ))

    def generate(self) -> str:
        ctx = self._acquire_ctx()
        message = None
        response = None
        finish_reason = None

        with ToolManager(str(self.AUX_AI)) as manager:
            for _ in range(self.REQ_TOOL_CALL_LIMIT):
                response = self.llm.create_chat_completion(
                    messages=ctx,
                    tools=self.tool_schemas,
                    max_tokens=self.MAX_OUTPUT_TOKENS
                )
                
                choice = response["choices"][0]
                message = choice["message"]
                finish_reason = choice["finish_reason"]

                match finish_reason:
                    case "tool_calls":
                        self.resolve_tool_call(manager, response)
                        ctx.extend(self.history[-2:]) # resync req state with app data
                    case "stop":
                        break
                    case "length":
                        print("generate warning: chat reply completed early due to running out of tokens")
                        break

        if message is None:
            return "No response was generated"

        if finish_reason == "tool_calls":
            # Tool-call budget exhausted; force a final textual response.
            response = self.llm.create_chat_completion(
                messages=ctx,
                max_tokens=self.MAX_OUTPUT_TOKENS
            )
            message = response["choices"][0]["message"]

        content = message.get("content") or "No content was generated"

        self.history.append(Message(
            role=Role.ASSISTANT,
            content=content,
            ntokens=response["usage"]["completion_tokens"]
        ))

        return content

    def _append_history(self):
        # No new records added
        if len(self.history) <= self.start_of_new_history:
            return
        
        # Sum the serialized size of the current context window
        context_nbytes = 0
        total_tokens = 0
        cache = {}
        
        for record in reversed(self.history):
            if total_tokens + record.ntokens > self.MAX_CTX_TOKENS:
                break
            
            serialized = record.pack()
            
            context_nbytes += len(serialized)
            total_tokens += record.ntokens
            
            cache[record.uid] = serialized
        
        # Serialize the newly added records
        buffer = bytearray()
        for record in self.history[self.start_of_new_history:]:
            if record.uid in cache:
                buffer.extend(cache[record.uid])
            else:
                buffer.extend(record.pack())
        
        footer = self.HST_FTR.pack(context_nbytes)
        buffer.extend(footer) 
        
        # Write the serialized data, overlapping the old footer bytes
        with self.HIST_STORE.open('r+b') as f:
            f.seek(-self.HST_FTR.size, os.SEEK_END)
            f.write(buffer)
            f.truncate()
            
        self.start_of_new_history = len(self.history)

    def _restore_history(self):
        self.history = []
        self.start_of_new_history = 0

        # Create if no existant
        if not self.HIST_STORE.exists():
            self.HIST_STORE.write_bytes(self.HST_FTR.pack(0))
            return

        # Overwrite if undersized (would cause a negative backstep during writing)
        size = self.HIST_STORE.stat().st_size
        if size < self.HST_FTR.size:
            self.HIST_STORE.write_bytes(self.HST_FTR.pack(0))
            return

        with PositionalReader(self.HIST_STORE) as f:
            context_nbytes, = self.HST_FTR.unpack(
                f[-self.HST_FTR.size : self.HST_FTR.size : -1]
            )
            view = memoryview(
                f[-context_nbytes : context_nbytes : 0]
            )

        # Scan through context window
        offset = 0
        while offset < context_nbytes:
            msg, offset = Message.unpack(view, offset)
            self.history.append(msg)
            
        # Advance journal boundary
        self.start_of_new_history = len(self.history)
        
    