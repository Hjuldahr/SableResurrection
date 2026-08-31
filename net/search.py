from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from collections.abc import Iterator
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
        max_results: int = 10,
        max_source_tokens: int = 4000,
        max_output_tokens: int = 350,
    ) -> None:
        if max_results < 1:
            raise ValueError("max_results must be at least 1")
        if max_source_tokens < 1:
            raise ValueError("max_source_tokens must be at least 1")
        if max_output_tokens < 1:
            raise ValueError("max_output_tokens must be at least 1")
        
        self.llm = llm
        self.max_results = max_results
        self.max_source_tokens = max_source_tokens
        self.max_output_tokens = max_output_tokens

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
            max_workers=min(len(results), 10)
        ) as executor:
            pages = executor.map(self.fetch, results)

        return [
            (result, text)
            for result, text in zip(results, pages)
            if text
        ]

    @staticmethod
    def _get_line(text: str) -> Iterator[str]:
        i = 0
        n = len(text)

        while i < n:
            j = text.find("\n", i)

            if j == -1:
                j = n
            else:
                j += 1

            yield text[i:j]
            i = j

    def _truncate(
        self,
        text: str,
        max_site_tokens: int,
    ) -> str:
        """Truncate text to max_site_tokens while preserving whole lines where possible."""
        token_count = 0
        char_count = 0
        
        text = text[:max_site_tokens * 12]

        for line in self._get_line(text):
            line_tokens = self.llm.tokenize(line.encode("utf-8"))
            line_token_count = len(line_tokens)

            remaining = max_site_tokens - token_count

            if line_token_count <= remaining:
                token_count += line_token_count
                char_count += len(line)
                continue

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

        # 1. Filter out completely empty scrapes first so they don't break our math
        valid_sources = [(res, text) for res, text in sources if text and text.strip()]
        
        if not valid_sources:
            return "No readable information was found from the web sources."

        # Distributes max_source_tokens evenly across all active web pages
        max_web_tokens = max(
            1, 
            self.max_source_tokens // len(valid_sources)
        )

        source_sections = []

        for result, text in valid_sources:
            print(f'I am currently reading: {result.title}') # TEMP
            
            # Apply the dynamically calculated budget ceiling
            text = self._truncate(
                text,
                max_web_tokens,
            )

            source_sections.append(
                f"TITLE: {result.title}\n"
                f"{text}"
            )

        source_text = "\n\n".join(source_sections)

        prompt = f"""
Answer the query using only the information provided below.

QUERY:
{query}

INFORMATION:
{source_text}

Write one informational paragraph that directly answers the query.

Rules:
- Do not repeat the query or these instructions, or output text that describes these constraints.
- State only direct, concrete answers to the query; do not add generic concluding summaries or sweeping generalizations.
- Use only facts explicitly supported by the supplied information.
- Do not use outside knowledge.
- Do not treat a date as answering the query unless the supplied information explicitly associates that date with the event described in the query.
- If the supplied information does not clearly establish the answer, say that it is unclear.
- Preserve important qualifications and uncertainty.
- If the information disagrees, briefly acknowledge the disagreement.
- Do not mention the search process.
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