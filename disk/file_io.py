from __future__ import annotations

from datetime import datetime, timezone
import mimetypes
from pathlib import Path

from llama_cpp import Llama


ROOT = Path(__file__).parents[1].resolve()
WORKSPACE_ROOT = ROOT / "file-workspace"
GENERATED_ROOT = WORKSPACE_ROOT / "generated"


def is_sanctioned_file(path: Path) -> bool:
    """Return whether path is a regular, non-symlink file inside the workspace."""
    return (
        not path.is_symlink()
        and path.is_file()
        and path.resolve().is_relative_to(WORKSPACE_ROOT)
    )


def is_generated_path(path: Path) -> bool:
    """Return whether the resolved path lies inside the generated workspace."""
    return (
        not path.is_symlink()
        and path.resolve().is_relative_to(GENERATED_ROOT)
    )


def format_size(size_in_bytes: int) -> str:
    """Convert a byte count to a human-readable size."""
    if size_in_bytes < 0:
        raise ValueError("File size cannot be negative.")

    if size_in_bytes == 0:
        return "0 B"

    units = ("B", "KB", "MB", "GB", "TB", "PB", "EB")

    exponent = min(
        (size_in_bytes.bit_length() - 1) // 10,
        len(units) - 1,
    )

    value = size_in_bytes / (1024 ** exponent)

    if exponent == 0:
        return f"{size_in_bytes} B"

    return f"{value:.1f} {units[exponent]}"


def utc_timestamp(timestamp: float) -> str:
    """Convert a filesystem timestamp to an ISO-8601 UTC timestamp."""
    return datetime.fromtimestamp(
        timestamp,
        timezone.utc,
    ).isoformat()


def file_stats(file: Path) -> tuple[str, str, str, str]:
    """Return creation, modification, access times and human-readable size."""
    stats = file.stat()

    return (
        utc_timestamp(stats.st_birthtime),
        utc_timestamp(stats.st_mtime),
        utc_timestamp(stats.st_atime),
        format_size(stats.st_size),
    )


def browse_file_candidates(pattern: str = "*") -> str:
    """Return metadata and previews for readable files matching a workspace glob."""
    if not WORKSPACE_ROOT.exists():
        return "WARNING: File workspace has not been initialized."

    data: list[str] = []

    for file in WORKSPACE_ROOT.rglob(pattern, case_sensitive=False):
        if not is_sanctioned_file(file):
            continue

        try:
            created, modified, accessed, size = file_stats(file)

            with file.open("r", encoding="utf-8") as stream:
                preview = stream.readline(256).strip()

        except (OSError, UnicodeDecodeError):
            continue

        mime_type, _ = mimetypes.guess_type(file.name)
        mime = mime_type or "unknown"
        location = file.relative_to(WORKSPACE_ROOT)

        data.append(
            f"- location: {location}, "
            f"mime: {mime}, "
            f"size: {size}, "
            f"created-at: {created}, "
            f"modified-at: {modified}, "
            f"last-accessed-at: {accessed}, "
            f"preview: {preview}"
        )

    if not data:
        return f"INFO: No files were found for the glob pattern: {pattern}."

    return "\n".join(data)


def write_file(
    filename: str,
    file_content: str,
    *,
    append: bool = True,
) -> str:
    """Write model-generated content inside the generated workspace."""
    path = GENERATED_ROOT / filename

    if not is_generated_path(path):
        return "WARNING: Filepath is outside the permitted generated workspace."

    # An empty append is a no-op, while an empty overwrite intentionally
    # clears the file.
    if append and not file_content.strip():
        return "INFO: No content to append."

    try:
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        mode = "a" if append else "w"

        with path.open(
            mode,
            encoding="utf-8",
            newline="\n",
        ) as stream:
            stream.write(file_content)

    except OSError as exc:
        action = "append to" if append else "write to"

        print(f"Failed to {action} file {filename}: {exc}")

        return f"ERROR: Failed to {action} file {filename}: {exc}"

    action = "appended" if append else "wrote"

    return f"INFO: Successfully {action} to file {filename}"


class FileHandler:
    def __init__(
        self,
        llm: Llama,
        *,
        text_file_batch_limit: int = 10,
        max_source_tokens: int = 10_000,
        max_output_tokens: int = 350,
    ) -> None:
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

    @staticmethod
    def _resolve_workspace_path(file: str | Path) -> Path:
        """Resolve a file argument relative to the workspace when necessary."""
        path = Path(file)

        if not path.is_absolute():
            path = WORKSPACE_ROOT / path

        return path

    @staticmethod
    def _validate_utf_file(file: Path) -> bool:
        """Perform a lightweight check for readable, non-empty UTF-8 text."""
        try:
            if not is_sanctioned_file(file):
                return False

            if file.stat().st_size == 0:
                return False

            with file.open("r", encoding="utf-8") as stream:
                return "\x00" not in stream.read(1024)

        except (OSError, UnicodeDecodeError):
            return False

    def _process_file(
        self,
        file: Path,
        max_file_tokens: int,
    ) -> str:
        try:
            with file.open("r", encoding="utf-8") as stream:
                lines: list[str] = []
                token_count = 0

                for line in stream:
                    remaining = max_file_tokens - token_count

                    if remaining <= 0:
                        break

                    line_tokens = self.tokenizer.tokenize(
                        line.encode("utf-8"),
                        add_bos=False,
                    )

                    if len(line_tokens) <= remaining:
                        lines.append(line)
                        token_count += len(line_tokens)
                        continue

                    truncated = self.tokenizer.detokenize(
                        line_tokens[:remaining],
                    ).decode(
                        "utf-8",
                        errors="replace",
                    )

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

    def _collect_valid_files(self, files: tuple[str, ...]) -> list[Path]:
        """Resolve and validate file arguments up to the configured batch limit."""
        valid_paths: list[Path] = []
        seen: set[Path] = set()

        for file in files:
            path = self._resolve_workspace_path(file)

            # Resolve only for duplicate detection. Validation itself still
            # receives the original path so that symlink policy is preserved.
            try:
                identity = path.resolve()
            except OSError:
                continue

            if identity in seen:
                continue

            if not self._validate_utf_file(path):
                continue

            seen.add(identity)
            valid_paths.append(path)

            if len(valid_paths) >= self.text_file_batch_limit:
                break

        return valid_paths

    def _build_source_sections(
        self,
        files: list[Path],
        max_file_tokens: int,
    ) -> list[str]:
        """Read files and construct model-facing source sections."""
        sections: list[str] = []

        for path in files:
            text = self._process_file(
                path,
                max_file_tokens,
            )

            if text.startswith("ERROR: "):
                print(f"{path.name}: {text}")
                continue

            sections.append(
                f"File Name: {path.relative_to(WORKSPACE_ROOT)}\n"
                f"File Content\n"
                f"{text}"
            )

        return sections

    def _summarize(self, source_text: str) -> str:
        """Summarize a collection of source text using the configured LLM."""
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

    def read_files(self, *files: str) -> str:
        """Read and collectively summarize a batch of workspace text files."""
        valid_paths = self._collect_valid_files(files)

        if not valid_paths:
            return "ERROR: No readable files were provided."

        max_file_tokens = max(
            1,
            self.max_source_tokens // len(valid_paths),
        )

        source_sections = self._build_source_sections(
            valid_paths,
            max_file_tokens,
        )

        if not source_sections:
            return "ERROR: None of the supplied files could be read."

        source_text = "\n\n".join(source_sections)

        return self._summarize(source_text)