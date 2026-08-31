from datetime import datetime, timezone
import math
import mimetypes
from pathlib import Path
from llama_cpp import Llama

WORKSPACE_ROOT = Path(__file__).parents[1].resolve() / 'file-workspace'
GENERATED_ROOT = (WORKSPACE_ROOT / "generated").resolve()

def is_sanctioned_file(path: Path) -> bool:
    return not path.is_symlink() and path.is_file() and path.resolve().is_relative_to(WORKSPACE_ROOT)

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
            if not is_sanctioned_file(file):
                return False
            
            if file.stat().st_size == 0:
                return False

            with file.open(mode="r", encoding="utf-8") as f:
                return "\x00" not in f.read(1024)

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

Write one informational paragraph that explains the content of the text files.

Rules:
- Do not repeat these instructions, or output text that describes these constraints.
- Use only facts explicitly supported by the supplied information.
- Do not use outside knowledge.
- Preserve important qualifications and uncertainty.
- If the information disagrees, briefly acknowledge the disagreement.
- Do not use headings, bullets, labels, or meta-commentary.
- Do not use quotation marks.
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
    
def _inside_generated(child: Path) -> bool:
    return child.resolve().is_relative_to(GENERATED_ROOT)
    
def write_file(
    filename: str,
    file_content: str,
    *,
    append: bool = True,
) -> str:
    path = GENERATED_ROOT / filename

    if not _inside_generated(path):
        return "WARNING: Filepath is outside the permitted generated workspace."

    try:
        path.parent.mkdir(exist_ok=True, parents=True)

        mode = "a" if append else "w"

        with path.open(mode, encoding="utf-8", newline="\n") as f:
            f.write(file_content)

        action = "appended" if append else "wrote"
        return f"INFO: Successfully {action} to file {filename}"

    except OSError as exc:
        print(f"Failed to write file {filename}: {exc}")
        return (
            f"ERROR: Failed to {'append' if append else 'write'} "
            f"to file {filename}: {exc}"
        )
        
def format_size(size_in_bytes: int) -> str:
    if size_in_bytes < 0:
        raise ValueError("File size cannot be negative.")
    if size_in_bytes == 0:
        return "0 B"

    units = ("B", "KB", "MB", "GB", "TB", "PB", "EB")

    i = min(
        (size_in_bytes.bit_length() - 1) // 10,
        len(units) - 1,
    )

    value = size_in_bytes / (1024 ** i)

    return f"{value:.1f} {units[i]}" if i else f"{size_in_bytes} B"
        
def browse_file_candidates(glob: str = "*") -> str:
    data = []
    
    if not WORKSPACE_ROOT.exists():
        return "WARNING: File workspace has not been initialized yet."
    
    for file in WORKSPACE_ROOT.rglob(glob, case_sensitive=False):
        if not is_sanctioned_file(file):
            continue
        
        try:
            stats = file.stat()
            ctime = datetime.fromtimestamp(stats.st_birthtime, timezone.utc)
            mtime = datetime.fromtimestamp(stats.st_mtime, timezone.utc)
            atime = datetime.fromtimestamp(stats.st_atime, timezone.utc)
            size = format_size(stats.st_size)
        except OSError:
            continue

        try:
            with file.open("r", encoding="utf-8") as f:
                preview = f.readline(256).strip()
        except (OSError, UnicodeDecodeError):
            continue

        mime_type, _ = mimetypes.guess_type(file.name)
        mime = mime_type or "unknown"

        loc = str(file.relative_to(WORKSPACE_ROOT))

        data.append(
            f"- location: {loc}, " 
            f"mime: {mime}, "
            f"size: {size}, "
            f"created-at: {ctime.isoformat()}, " 
            f"modified-at: {mtime.isoformat()}, " 
            f"last-accessed-at: {atime.isoformat()}, "
            f"preview: {preview}"
        )
        
    if not data:
        return f"INFO: No files were found for the glob pattern: {glob}."
        
    return '\n'.join(data)

print(browse_file_candidates())