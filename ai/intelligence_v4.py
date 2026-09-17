from __future__ import annotations
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any
from llama_cpp import ChatCompletionRequestMessage, ChatCompletionTool, CreateChatCompletionResponse, Llama, llama_chat_format

from ai.session_store import SessionStore
from ai_tools.manager import ToolManager
from ai.message import Message
from ai.role import Role

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
    
    TOOL_SCHEMA = ROOT / 'ai_tools' / 'tool_schema.json'
    CHAT_FORMAT = 'llama-3'
    
    REQ_TOOL_CALL_LIMIT = 9
    MAX_CONTEXT_TOKENS = 32_768
    MAX_OUTPUT_TOKENS = 512
    
    def __init__(self):
        self.history: list[Message] = []
        self.start_of_new_history = 0
        
        self.session_store = SessionStore()
        self.session_store.load_session()
        
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
            datetime.fromtimestamp(0.0, timezone.utc),
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
        ctx = [self.instruction.to_msg_obj()]
        ctx.extend(msg.to_msg_obj() for msg in self.session_store.fetch_context(self.conservative_max_context_tokens))
        return ctx

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