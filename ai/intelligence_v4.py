from __future__ import annotations
from collections import deque
from datetime import datetime, timezone
from enum import Enum
import json
from os import SEEK_END
from pathlib import Path
import struct
from typing import Any
import uuid
from llama_cpp import ChatCompletionRequestMessage, ChatCompletionTool, CreateChatCompletionResponse, Llama, llama_chat_format

from ai_tools.manager import ToolManager

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
        
    def __int__(self) -> int:
        return self.ordinal
    
    def __str__(self) -> str:
        return self._value_
        
    @classmethod
    def from_ordinal(cls, ordinal: int) -> Role | None:
        return cls._ordinal_map.get(ordinal)

class Message:
    __slots__ = ('uid', 'dt', 'role', 'content', 'content_bytes', 'ntokens', 'chat_meta', 'ui_meta')
    
    NAMESPACE = uuid.UUID('SABLEII')
    FMT = struct.Struct('<H7B2H')
    
    def __init__(
        self, 
        dt: datetime | None = None,
        role: Role = Role.SYSTEM, 
        content: str | None = None, 
        content_bytes: bytes | None = None,
        ntokens: int = 0, 
        *, 
        chat_meta: dict[str, Any] | None = None,
        ui_meta: dict[str, Any] | None = None
    ):
        self.dt = dt if dt is not None else datetime.now(timezone.utc)
        self.uid = uuid.uuid5(self.NAMESPACE, self.dt.strftime("%Y%m%d%H%M%S%.3f"))
        
        self.role = role
        
        self.content = content if content is not None else content_bytes.decode('utf-8', 'replace')
        self.content_bytes = content_bytes if content_bytes is not None else content.encode('utf-8', 'replace')
        
        self.ntokens = ntokens
        
        self.chat_meta = {} if chat_meta is None else chat_meta
        self.ui_meta = {} if ui_meta is None else ui_meta
    
    def pack(self) -> bytes:
        dt = self.dt
        return self.FMT.pack(
            dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second, dt.microsecond // 1000,
            int(self.role), self.ntokens, len(self.content_bytes)
        ) + self.content_bytes

    @property
    def byte_count(self) -> int:
        return self.FMT.size + len(self.content_bytes)

    @classmethod
    def unpack(cls, view: memoryview, offset: int = 0) -> tuple[Message, int]:
        if offset + cls.FMT.size > len(view):
            raise ValueError("Truncated message header")
        
        year, month, day, hour, minute, second, microsecond, role_ordinal, ntokens, content_nbytes = cls.FMT.unpack_from(view, offset)
        offset += cls.FMT.size
        
        if offset + content_nbytes > len(view):
            raise ValueError("Truncated message content")
        
        dt = datetime(year, month, day, hour, minute, second, microsecond, timezone.utc)
        
        role = Role.from_ordinal(role_ordinal)
        
        content_bytes = view[offset:offset + content_nbytes].tobytes()
        offset += content_nbytes
        
        return Message(dt, role, None, content_bytes, ntokens), offset

    def to_msg_obj(self) -> dict[str, Any]:
        return {
            **self.chat_meta,
            'role': self.role.value,
            'content': self.content,
        }

class Sable:
    # AI must be generally aware of its own tool budget, but also have a rich persona.
    INSTRUCTION_PROMPT = """You have a strict resource budget equivalent to 3 to 9 tool executions per user request, depending on the individual tools' resource costs.
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
    CACHE = ROOT / 'conf' / 'cache_cache.json'
    HIST_STORE = ROOT / 'session' / 'session.bin'
    TOOL_SCHEMA = ROOT / 'ai_tools' / 'tool_schema.json'
    CHAT_FORMAT = 'llama-3'
    
    REQ_TOOL_CALL_LIMIT = 9
    MAX_CONTEXT_TOKENS = 32_768
    MAX_OUTPUT_TOKENS = 512
    
    DEDUPLICATE_WINDOW = 60.0
    
    HST_FTR = struct.Struct('<I')
    
    def __init__(self):
        self.history: list[Message] = []
        self.start_of_new_history = 0
        self.restore_history()
        
        self.tool_schemas = self.acquire_tool_schemas()
        
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
        self.chat_handler = self.acquire_chat_formatter()
        
        self.template_baseline = 0
        self.instruction_overhead = 0
        self.conservative_max_context_tokens = 0
        
        self.instruction = Message(
            Role.SYSTEM, 
            self.INSTRUCTION_PROMPT, 
        )
        
        self.cache()
    
    def cache(self):
        if self.CACHE.exists():
            with self.CACHE.open(encoding='ascii') as f:
                boot_data = json.load(f)
                self.template_baseline = boot_data['tmpl_bl']
                self.instruction_overhead = boot_data['instr_ovh']
        else:
            self.template_baseline = self.count_tokens([], tools=self.tool_schemas)
            self.instruction_overhead = self.count_tokens([self.instruction.to_msg_obj()], tools=self.tool_schemas)
            
            self.CACHE.parent.mkdir(exist_ok=True, parents=True)
            with self.CACHE.open('w', encoding='ascii') as f:
                json.dump({
                    'tmpl_bl': self.template_baseline, 
                    'instr_ovh': self.instruction_overhead
                }, f)
                
        self.conservative_max_context_tokens = self.MAX_CONTEXT_TOKENS - (self.MAX_OUTPUT_TOKENS + self.instruction_overhead)
    
    def close(self):
        self.append_history()
        self.llm.close()
        
        self.llm = None
        self.sentence_model = None
    
    def count_tokens(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> int:
        formatted = self.chat_handler(llama=self.llm, messages=messages, tools=tools)
        return formatted["usage"]["prompt_tokens"]
    
    def token_count_message(self, message: Message) -> None:
        msg_dict = message.to_msg_obj()
        message_count = self.count_tokens([msg_dict], tools=self.tool_schemas)
        
        message.ntokens = message_count - self.template_baseline
    
    def acquire_chat_formatter(self) -> llama_chat_format.LlamaChatCompletionHandler:
        # Because llm.chat_handler is None, and llm.chat_format = chat_template.default which is not in the global registry, 
        # it is instead hardcoded as a class level constant due to it being unlikely to change
        return llama_chat_format.get_chat_completion_handler(self.CHAT_FORMAT)
    
    def acquire_tool_schemas(self) -> list[ChatCompletionTool]:
        with self.TOOL_SCHEMA.open("r", encoding="utf-8") as f:
            return json.load(f)

    def acquire_ctx(self) -> list[ChatCompletionRequestMessage]: 
        ctx = deque() 
        total_tokens = 0
        
        for msg in reversed(self.history): 
            if total_tokens + msg.ntokens > self.conservative_max_context_tokens: 
                break 
            total_tokens += msg.ntokens
                
            ctx.appendleft(msg.to_msg_obj())
        
        ctx.appendleft(self.instruction.to_msg_obj()) 
            
        return list(ctx)

    def resolve_tool_call(
        self,
        manager: ToolManager,
        response: CreateChatCompletionResponse,
        tool_call: dict[str, Any],
    ) -> None:
        assistant_tokens = response["usage"]["completion_tokens"]
        message = response["choices"][0]["message"]
        assistant_dt = datetime.fromtimestamp(response["created"], timezone.utc)

        function = tool_call["function"]
        name = function["name"]
        arguments = json.loads(function.get("arguments", "{}"))

        content = manager.execute(command=name, **arguments)
        tool_dt = datetime.now(timezone.utc)

        self.history.append(Message(
            role=Role.ASSISTANT,
            content=message.get("content") or "No content found.",
            ntokens=assistant_tokens,
            dt=assistant_dt,
            chat_meta={"tool_calls": message["tool_calls"]},
        ))

        self.history.append(Message(
            role=Role.TOOL,
            content=content,
            ntokens=len(self.llm.tokenize(content.encode("utf-8"))),
            dt=tool_dt,
            chat_meta={
                "tool_call_id": tool_call["id"],
                "name": name,
            },
        ))

    def generate(self) -> Message:
        message = None
        response = None
        finish_reason = None

        with ToolManager(str(self.AUX_AI)) as manager:
            while manager.report.tool_usage < self.REQ_TOOL_CALL_LIMIT:
                ctx = self.acquire_ctx()

                response = self.llm.create_chat_completion(
                    messages=ctx,
                    tools=self.tool_schemas,
                    max_tokens=self.MAX_OUTPUT_TOKENS,
                )

                choice = response["choices"][0]
                message = choice["message"]
                finish_reason = choice["finish_reason"]

                match finish_reason:
                    case "tool_calls":
                        tool_call = message["tool_calls"][0]
                        self.resolve_tool_call(manager, response, tool_call)

                    case "stop":
                        break

                    case "length":
                        print("generate warning: chat reply completed early due to running out of tokens")
                        break

            report = manager.report

        if finish_reason == "tool_calls":
            response = self.llm.create_chat_completion(
                messages=self.acquire_ctx(),
                max_tokens=self.MAX_OUTPUT_TOKENS,
            )
            message = response["choices"][0]["message"]

        content = message.get("content") or "No content was generated"

        assistant_msg = Message(
            role=Role.ASSISTANT,
            content=content,
            ui_meta={'tool-report': report}
        )
        self.token_count_message(assistant_msg)
        self.history.append(assistant_msg)

        return assistant_msg

    def append_history(self):
        # No new records added
        if len(self.history) <= self.start_of_new_history:
            return
        
        # Sum the serialized size of the current context window
        context_nbytes = 0
        total_tokens = 0
        cache = {}
        
        for record in reversed(self.history):
            if total_tokens + record.ntokens > self.conservative_max_context_tokens:
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
        with self.HIST_STORE.open('wb') as f:
            f.seek(-self.HST_FTR.size, SEEK_END)
            f.write(buffer)
            
        self.start_of_new_history = len(self.history)

    def restore_history(self):
        self.history = []
        self.start_of_new_history = 0

        # Create if no existant
        if not self.HIST_STORE.exists():
            self.HIST_STORE.write_bytes(self.HST_FTR.pack(0))
            return

        # Overwrite if undersized (would cause a negative backstep during writing)
        size = self.HIST_STORE.stat().st_size
        if size <= self.HST_FTR.size:
            self.HIST_STORE.write_bytes(self.HST_FTR.pack(0))
            return
        
        with self.HIST_STORE.open('rb') as f:
            f.seek(-self.HST_FTR.size, SEEK_END)
            context_nbytes, = self.HST_FTR.unpack(f.read(self.HST_FTR.size))
            
            f.seek(-(self.HST_FTR.size + context_nbytes), SEEK_END)
            view = memoryview(f.read(context_nbytes))

        # Scan through context window
        offset = 0
        while offset < context_nbytes:
            msg, offset = Message.unpack(view, offset)
            self.history.append(msg)
            
        # Advance journal boundary
        self.start_of_new_history = len(self.history)
        
    def find_last_message_by_role(self, role: Role) -> Message | None:
        return next((msg for msg in reversed(self.history) if msg.role == role), None)
        
    def submit(self, prompt: str, file_attachments: list[str | Path] | None = None) -> None:
        prompt = prompt.strip()
    
        ui_meta = {}
        
        if file_attachments:
            ui_meta['file_attachments'] = file_attachments
        
        msg = Message(
            role=Role.USER,
            content=prompt,
            ui_meta=ui_meta,
        )
        self.token_count_message(msg)
        
        self.history.append(msg)