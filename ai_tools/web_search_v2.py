from typing import Any, Iterator
from ddgs.ddgs import DDGS
from llama_cpp import Llama
import trafilatura
from trafilatura.xml import xmltotxt

from ai_tools.dto import DTO, ResourceFlags

def _format_text_result(result: dict[str, Any]) -> str:
    return (f'- title: {result["title"]}, url: {result["href"]}, preview: {result.get("body", "No text found")}')

def search_text(query: str, *, max_results: int = 10) -> tuple[str, list[DTO]]:
    """Search the web for relevant pages."""
    with DDGS() as ddgs:
        results = ddgs.text(query, region="us-en", safesearch="off", timelimit="y", max_results=max_results, backend="auto")

    return (
        ', '.join(_format_text_result(res) for res in results), 
        [DTO(res["title"], ResourceFlags.BROWSED | ResourceFlags.WEB, res["href"], "text") for res in results]
    )

def _format_news_result(result: dict[str, Any]) -> str:
    return (
        f'- title: {result["title"]}, url: {result["url"]}, date: {result["date"]}, source: {result["source"]}, preview: {result.get("body", "No text found")}'
    )

def search_news(query: str, *, max_results: int = 10) -> tuple[str, list[DTO]]:
    """Search the web for relevant news articles."""
    with DDGS() as ddgs:
        results = ddgs.news(query, region="ca-en", safesearch="off", timelimit="m", max_results=max_results, backend="auto")

    return (
        ', '.join(_format_news_result(res) for res in results), 
        [DTO(res["title"], ResourceFlags.BROWSED | ResourceFlags.WEB, res["url"], "news") for res in results]
    )

def _format_books_result(result: dict[str, Any]) -> str:
    return (
        f'- title: {result["title"]}, url: {result["url"]}, author: {result["author"]}, publisher: {result["publisher"]}, info: {result["info"]}'
    )

def search_books(query: str, *, max_results: int = 10) -> tuple[str, list[DTO]]:
    """Search the web for relevant uploaded literature."""
    with DDGS() as ddgs:
        results = ddgs.books(query, max_results=max_results, backend="auto")

    return (
        ', '.join(_format_books_result(res) for res in results), 
        [DTO(res["title"], ResourceFlags.BROWSED | ResourceFlags.WEB, res["url"], "books") for res in results]
    )

class PageSummarizer:
    INSTRUCTIONS = """Write one informational paragraph that directly answers the query using the information provided.
RULES:
Do not repeat or describe the query or these rules. 
Do not use outside knowledge or combine separate facts into an unsupported conclusion. 
Do not substitute a broader or different geographic, temporal, or categorical scope for the one asked about. 
Do not treat a date as answering the query unless it is explicitly associated with the queried event. 
Do not add unrelated facts, generic conclusions, headings, bullets, labels, meta-commentary, or quotation marks. 
If an answer is not clearly established or is contradictory, state that it is uncertain. 
Answer the query question first, then include auxiliary context as needed to address uncertainties within the source. 
End with a complete sentence and output only the paragraph."""
    
    def __init__(
        self,
        llm: Llama,
        *,
        max_source_tokens: int = 16_000,
        max_output_tokens: int = 350,
        request_timeout: float = 10.0,
    ) -> None:
        if max_source_tokens < 1:
            raise ValueError("max_source_tokens must be at least 1")

        if max_output_tokens < 1:
            raise ValueError("max_output_tokens must be at least 1")

        if request_timeout <= 0:
            raise ValueError("request_timeout must be greater than 0")

        self.llm = llm

        self.max_source_tokens = max_source_tokens
        self.max_output_tokens = max_output_tokens

        self.request_timeout = request_timeout

    @staticmethod
    def _iter_lines(text: str) -> Iterator[str]:
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

    @classmethod
    def _fetch(cls, url: str) -> tuple[str, str | None]:
        """Download and extract the page title and main text."""
        downloaded = trafilatura.fetch_url(url)

        if not downloaded:
            return "Webpage not found", None

        doc = trafilatura.bare_extraction(
            downloaded,
            with_metadata=True,
            include_links=False,
            include_tables=True,
            favor_precision=True,
            include_comments=False,
            include_formatting=True,
            deduplicate=True,
            date_extraction_params={
                "original_date": True,
                "outputformat": "%Y-%m-%d",
            }
        )

        if doc is None:
            return f"Document not found at {url}", None

        text = xmltotxt(doc.body, include_formatting=True)
        return doc.title or "Title not found", text

    def _summarize(self, query: str, source_text: str) -> str: 
        summary_prompt = f"""QUERY: {query} 
        
        INFORMATION: {source_text}""" 
        
        response = self.llm.create_chat_completion(
            messages=[ 
                { "role": "system", "content": self.INSTRUCTIONS}, 
                { "role": "user", "content": summary_prompt}, 
            ], max_tokens=self.max_output_tokens, temperature=0.3) 
        
        return response["choices"][0]["message"]["content"].strip()

    def _truncate(self, text: str) -> str:
        """Truncate text to at most max_tokens, preferring whole lines."""
        if not text:
            return ""

        token_count = 0
        char_count = 0

        # Avoid scanning arbitrarily large pages when token density is very
        # low. This is only a pre-filter; the token budget remains authoritative.
        text = text[: self.max_source_tokens * 12]

        for line in self._iter_lines(text):
            remaining = self.max_source_tokens - token_count

            if remaining <= 0:
                break

            line_tokens = self.llm.tokenize(line.encode("utf-8"), add_bos=False)

            line_token_count = len(line_tokens)

            if line_token_count <= remaining:
                token_count += line_token_count
                char_count += len(line)
                continue

            truncated = self.llm.detokenize(
                line_tokens[:remaining],
            ).decode("utf-8", errors="replace")

            return text[:char_count] + truncated

        return text[:char_count]

    def summarize_page(self, query: str, url: str) -> tuple[str, list[DTO]]:
        """Fetch and summarize a specified webpage in response to a query."""
        title, text = self._fetch(url)

        if not text:
            return (
                "No readable information was found.", 
                [DTO(title, ResourceFlags.PROCESSED | ResourceFlags.WEB, url)]
            )

        text = self._truncate(text)

        return (
            self._summarize(query, text), 
            [DTO(title, ResourceFlags.PROCESSED | ResourceFlags.WEB, url)]
        )