from pathlib import Path
from llama_cpp import Llama

class FileHandler:
    def __init__(
        self,
        llm: Llama,
        *,
        text_file_batch_limit: int = 10,
        max_source_tokens: int = 10_000,
        max_output_tokens: int = 350,
    ):
        if text_file_batch_limit < 1:
            raise ValueError("text_file_batch_limit must be at least 1")
        if max_source_tokens < 1:
            raise ValueError("max_source_tokens must be at least 1")
        if max_output_tokens < 1:
            raise ValueError("max_output_tokens must be at least 1")

        self.llm = llm
        self.tokenizer = llm.tokenizer()

        self.text_file_batch_limit = text_file_batch_limit
        self.max_source_tokens = max_source_tokens
        self.max_output_tokens = max_output_tokens

    def _validate_utf_file(self, file: Path) -> bool:
        """Performs a lightweight check for readable, non-empty UTF-8 text."""
        try:
            with file.open(mode="r", encoding="utf-8") as f:
                # Sample the start of the file to catch invalid encodings
                chunk = f.read(1024)
                
                # Returns True if file has content & lacks null bytes (binary marker)
                return bool(chunk) and "\x00" not in chunk
                
        except (OSError, UnicodeDecodeError):
            return False

    def _process_file(self, file: Path, max_file_tokens: int) -> str:
        try:
            with file.open(encoding="utf-8") as f:
                lines: list[str] = []
                token_count = 0

                for line in f:
                    line_tokens = self.tokenizer.tokenize(
                        line.encode("utf-8"),
                        add_bos=False,
                    )

                    remaining = max_file_tokens - token_count

                    if remaining <= 0:
                        break

                    if len(line_tokens) <= remaining:
                        lines.append(line)
                        token_count += len(line_tokens)
                        continue

                    # The line would exceed the remaining budget.
                    truncated = self.tokenizer.detokenize(
                        line_tokens[:remaining]
                    ).decode("utf-8", errors="ignore")

                    lines.append(truncated)
                    break

                return "".join(lines)

        except FileNotFoundError:
            return "ERROR: File does not exist."
        except IsADirectoryError:
            return "ERROR: Path is not a file."
        except PermissionError:
            return "ERROR: The file is not readable."
        except UnicodeDecodeError:
            return "ERROR: The file is not a readable text file."
        except OSError:
            return "ERROR: The file could not be read."

    def read_files(self, *files: Path) -> str:
        valid_files = []
        for file in files:
            if file in valid_files:
                continue
                
            if self._validate_utf_file(file):
                valid_files.append(file)
                
                if len(valid_files) >= self.text_file_batch_limit:
                    break

        if not valid_files:
            return "ERROR: No readable files were provided."                

        max_file_tokens = max(
            1, self.max_source_tokens // len(valid_files),
        )

        source_sections: list[str] = []

        for i, file in enumerate(valid_files):
            text = self._process_file(file, max_file_tokens)

            if text.startswith("ERROR: "):
                print(f"{file.name}: {text}")
                continue

            source_sections.append(
                f"File Number {i}\n"
                f"File Name: {file.name}\n"
                f"File Content\n"
                f"{text}"
            )

        if not source_sections:
            return "ERROR: None of the supplied files could be read."

        source_text = "\n\n".join(source_sections)

        prompt = f"""
Summarize the text below.

{source_text}

Write one concise paragraph that explains the content of the text files.

Rules:
- Use only facts and figures from the supplied information.
- Do not use outside knowledge.
- Preserve important qualifications and uncertainty.
- If the information disagrees, briefly acknowledge the disagreement.
- Do not mention the search process.
- Do not repeat these instructions.
- Do not use headings, bullets, labels, or meta-commentary.
- Do not use quotation marks.
- Do not use the words source or sources.
- End with a complete sentence.
- Output only the paragraph.
"""

        response = self.llm(
            prompt,
            max_tokens=self.max_output_tokens,
            temperature=0.3,
            stop=["</s>"],
        )

        return response["choices"][0]["text"].strip()