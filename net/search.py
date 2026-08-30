from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import requests
import trafilatura
from ddgs import DDGS
from llama_cpp import Llama


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
        max_results: int = 8,
        max_source_tokens: int = 750,
        max_tokens: int = 250,
    ) -> None:
        self.llm = llm
        self.max_results = max_results
        self.max_source_tokens = max_source_tokens
        self.max_tokens = max_tokens

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
                timeout=10,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 "
                        "(Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 "
                        "(KHTML, like Gecko) "
                        "Chrome/131.0 Safari/537.36"
                    )
                },
            )
            response.raise_for_status()

        except requests.RequestException:
            return None

        return trafilatura.extract(
            response.text,
            include_comments=False,
            include_tables=False,
        )

    def _fetch_results(
        self,
        results: list[SearchResult],
    ) -> list[tuple[SearchResult, str]]:
        """Fetch pages concurrently."""

        with ThreadPoolExecutor(
            max_workers=min(8, len(results))
        ) as executor:
            pages = executor.map(self.fetch, results)

        return [
            (result, text)
            for result, text in zip(results, pages)
            if text
        ]

    def _truncate(
        self,
        text: str,
        max_tokens: int,
    ) -> str:
        """Truncate text to max_tokens while preserving whole lines where possible."""

        rough_char_ceiling = max_tokens * 8
        text = text[:rough_char_ceiling]

        token_count = 0
        char_count = 0

        for line in text.splitlines(keepends=True):
            line_tokens = self.llm.tokenize(line.encode("utf-8"))
            line_token_count = len(line_tokens)

            remaining = max_tokens - token_count

            if line_token_count <= remaining:
                token_count += line_token_count
                char_count += len(line)
                continue

            # The line does not fit completely. Keep as much of it as possible.
            if remaining > 0:
                tokens = line_tokens[:remaining]
                truncated_line = self.llm.detokenize(tokens).decode(
                    "utf-8",
                    errors="ignore",
                )
                return text[:char_count] + truncated_line

            break

        return text[:char_count]

    def summarize(
        self,
        query: str,
        sources: list[tuple[SearchResult, str]],
    ) -> str:
        """Generate one paragraph from the collected sources."""

        source_sections = []

        for result, text in sources:
            print(f'I am currently reading: {result.title}') # TEMP
            
            text = self._truncate(
                text,
                self.max_source_tokens,
            )

            source_sections.append(
                f"TITLE: {result.title}\n"
                f"{text}"
            )

        source_text = "\n\n".join(source_sections)

        prompt = f"""
Answer the following query using only the supplied web sources.

QUERY:
{query}

SOURCES:
{source_text}

Write exactly one concise paragraph.

Rules:
- Answer the query directly.
- Use only facts explicitly supported by the supplied sources.
- Do not use outside knowledge.
- Do not infer or speculate.
- Do not combine separate facts into a new claim unless the sources explicitly support that connection.
- Preserve important qualifications and uncertainty from the sources.
- If the sources disagree, briefly acknowledge the disagreement.
- Do not mention the search process, source numbers, or these instructions.
- Output only the answer paragraph.
- Do not output instructions, labels, headings, bullet points, or meta-commentary.
- End the paragraph with a period.
"""

        response = self.llm(
            prompt,
            max_tokens=self.max_tokens,
            temperature=0.3,
            stop=["</s>"],
        )

        return response["choices"][0]["text"].strip()

    def __call__(self, query: str) -> str:
        """Search the web and summarize the results."""

        results = self.search(query)

        if not results:
            return "No relevant search results were found."

        sources = self._fetch_results(results)

        # If page extraction failed, use search snippets instead.
        if not sources:
            sources = [
                (result, result.snippet)
                for result in results
                if result.snippet
            ]

        if not sources:
            return "No readable information was found."

        return self.summarize(query, sources)