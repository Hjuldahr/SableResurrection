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
        max_results: int = 5,
        max_source_tokens: int = 500,
        max_tokens: int = 300,
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
        """
        Truncate text according to the model's actual tokenizer.

        This is preferable to estimating tokens from character count.
        """

        tokens = self.llm.tokenize(
            text.encode("utf-8")
        )

        if len(tokens) <= max_tokens:
            return text

        tokens = tokens[:max_tokens]

        return self.llm.detokenize(tokens).decode(
            "utf-8",
            errors="ignore",
        )

    def summarize(
        self,
        query: str,
        sources: list[tuple[SearchResult, str]],
    ) -> str:
        """Generate one paragraph from the collected sources."""

        source_sections = []

        for result, text in sources:
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
Answer the following query using the supplied web sources.

QUERY:
{query}

SOURCES:
{source_text}

Write exactly one concise paragraph.

Rules:
- Answer the query directly.
- Use only information supported by the sources.
- Do not invent facts.
- Do not mention the search process.
- If the sources disagree, briefly acknowledge it.
- Output only the paragraph.
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