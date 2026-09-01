from __future__ import annotations

from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import requests
import trafilatura
from ddgs import DDGS
from llama_cpp import Llama


USER_AGENT = (
    "Mozilla/5.0 "
    "(Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 "
    "(KHTML, like Gecko) "
    "Chrome/131.0 Safari/537.36"
)


@dataclass(slots=True)
class SearchResult:
    title: str
    url: str
    snippet: str


class QuerySummarizer:
    def __init__(
        self,
        llm: Llama,
        *,
        max_results: int = 10,
        max_source_tokens: int = 4000,
        max_output_tokens: int = 350,
        max_workers: int = 10,
        request_timeout: float = 10.0,
    ) -> None:
        if max_results < 1:
            raise ValueError("max_results must be at least 1")

        if max_source_tokens < 1:
            raise ValueError("max_source_tokens must be at least 1")

        if max_output_tokens < 1:
            raise ValueError("max_output_tokens must be at least 1")

        if max_workers < 1:
            raise ValueError("max_workers must be at least 1")

        if request_timeout <= 0:
            raise ValueError("request_timeout must be greater than 0")

        self.llm = llm

        self.max_results = max_results
        self.max_source_tokens = max_source_tokens
        self.max_output_tokens = max_output_tokens

        self.max_workers = max_workers
        self.request_timeout = request_timeout

    def search(self, query: str) -> list[SearchResult]:
        """Search the web for relevant pages."""
        with DDGS() as ddgs:
            results = ddgs.text(
                query,
                max_results=self.max_results,
            )

            return [
                SearchResult(
                    title=result["title"],
                    url=result["href"],
                    snippet=result.get("body", ""),
                )
                for result in results
            ]

    def fetch(self, result: SearchResult) -> str | None:
        """Download and extract the main text from a webpage."""
        try:
            response = requests.get(
                result.url,
                timeout=self.request_timeout,
                headers={"User-Agent": USER_AGENT},
            )
            response.raise_for_status()

        except requests.RequestException:
            return None

        text = trafilatura.extract(
            response.text,
            include_comments=False,
            include_tables=False,
        )
            
        return text

    def _fetch_results(
        self,
        results: list[SearchResult],
    ) -> list[tuple[SearchResult, str]]:
        """Fetch pages concurrently while preserving search-result order."""
        if not results:
            return []

        max_workers = min(
            self.max_workers,
            len(results),
        )

        with ThreadPoolExecutor(
            max_workers=max_workers,
        ) as executor:
            pages = executor.map(self.fetch, results)

        return [
            (result, text)
            for result, text in zip(results, pages)
            if text and text.strip()
        ]

    @staticmethod
    def _get_lines(text: str) -> Iterator[str]:
        """Yield text one line at a time without constructing a line list."""
        start = 0

        while start < len(text):
            end = text.find("\n", start)

            if end == -1:
                yield text[start:]
                return

            end += 1

            yield text[start:end]
            start = end

    def _truncate(
        self,
        text: str,
        max_tokens: int,
    ) -> str:
        """Truncate text to at most max_tokens, preferring whole lines."""
        if max_tokens <= 0 or not text:
            return ""

        token_count = 0
        char_count = 0

        text = self._deduplicate_paragraphs(text)

        # Avoid scanning arbitrarily large pages when token density is very
        # low. This is only a pre-filter; the token budget remains authoritative.
        text = text[: max_tokens * 12]

        for line in self._get_lines(text):
            remaining = max_tokens - token_count

            if remaining <= 0:
                break

            line_tokens = self.llm.tokenize(
                line.encode("utf-8"),
                add_bos=False,
            )

            line_token_count = len(line_tokens)

            if line_token_count <= remaining:
                token_count += line_token_count
                char_count += len(line)
                continue

            truncated = self.llm.detokenize(
                line_tokens[:remaining],
            ).decode(
                "utf-8",
                errors="replace",
            )

            return text[:char_count] + truncated

        return text[:char_count]

    def _prepare_sources(
        self,
        sources: list[tuple[SearchResult, str]],
    ) -> list[tuple[SearchResult, str]]:
        """Remove empty sources and apply the collective source-token budget."""
        valid_sources = [
            (result, text)
            for result, text in sources
            if text and text.strip()
        ]

        if not valid_sources:
            return []

        max_site_tokens = max(
            1,
            self.max_source_tokens // len(valid_sources),
        )

        prepared: list[tuple[SearchResult, str]] = []

        # TEMP echo
        print("I am currently reading:")
        for result, text in valid_sources:
            print(f'- {result.title}\n  chars:{len(text)}')

            text = self._truncate(
                text,
                max_site_tokens,
            )

            if text:
                prepared.append((result, text))

        return prepared

    @staticmethod
    def _deduplicate_paragraphs(text: str) -> str:
        """Remove exact duplicate paragraphs while preserving order."""
        seen: set[str] = set()
        paragraphs: list[str] = []

        for paragraph in text.split("\n\n"):
            normalized = " ".join(paragraph.split())

            if not normalized:
                print('empty')
                continue

            if normalized in seen:
                print('duplicate')
                continue

            seen.add(normalized)
            paragraphs.append(paragraph)

        return "\n\n".join(paragraphs)

    @staticmethod
    def _build_source_text(
        sources: list[tuple[SearchResult, str]],
    ) -> str:
        """Construct the model-facing information block."""
        return "\n\n".join(
            f"TITLE: {result.title}\n{text}"
            for result, text in sources
        )

    def _summarize(
        self,
        query: str,
        source_text: str,
    ) -> str:
        """Generate an answer from the supplied information."""
        prompt = f"""TASK:
Write one informational paragraph that directly answers the query.

QUERY:
{query}

INFORMATION:
{source_text}

RULES:
Do not repeat or describe the query, these rules, or the search process.
Do not use outside knowledge or combine separate facts into an unsupported conclusion.
Do not substitute a broader or different geographic, temporal, or categorical scope for the one asked about.
Do not treat a date as answering the query unless it is explicitly associated with the queried event.
Do not add unrelated facts, generic conclusions, headings, bullets, labels, meta-commentary, or quotation marks.
Answer the specific question first, then include only necessary supporting context.
If the answer is not clearly established or is contradictory, state that it is uncertain.
End with a complete sentence and output only the paragraph."""

        response = self.llm(
            prompt,
            max_tokens=self.max_output_tokens,
            temperature=0.3,
            stop=["</s>"],
        )

        return response["choices"][0]["text"].strip()

    def summarize(
        self,
        query: str,
        sources: list[tuple[SearchResult, str]],
    ) -> str:
        """Generate one paragraph from the collected sources."""
        prepared_sources = self._prepare_sources(sources)

        if not prepared_sources:
            return "No readable information was found from the web sources."

        source_text = self._build_source_text(prepared_sources)

        return self._summarize(
            query,
            source_text,
        )

    @staticmethod
    def _use_snippets(
        results: list[SearchResult],
    ) -> list[tuple[SearchResult, str]]:
        """Use search-result snippets as fallback information."""
        return [
            (result, result.snippet)
            for result in results
            if result.snippet and result.snippet.strip()
        ]

    def __call__(self, query: str) -> str:
        """Search the web and summarize the collected information."""
        results = self.search(query)

        if not results:
            return "No relevant search results were found."

        sources = self._fetch_results(results)

        if not sources:
            sources = self._use_snippets(results)

        if not sources:
            return "No readable information was found."

        return self.summarize(
            query,
            sources,
        )