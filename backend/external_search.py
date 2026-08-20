"""
External search providers for fallback when vector DB has no relevant results.

Supports: Wikipedia, PubMed (NIH), and Groq LLM knowledge fallback.
No paid API keys required for Wikipedia and PubMed.
"""

import httpx
import re
import time
from typing import Optional


class WikipediaSearch:
    """Search Wikipedia for medical information."""

    BASE_URL = "https://en.wikipedia.org/api/rest_v1"
    SEARCH_URL = "https://en.wikipedia.org/w/api.php"

    async def search(self, query: str, max_chars: int = 2000) -> Optional[dict]:
        """
        Search Wikipedia and return article summary.

        Returns:
            dict with 'text', 'url', 'title' or None if not found
        """
        try:
            async with httpx.AsyncClient(timeout=8.0) as client:
                # Search for relevant article
                search_resp = await client.get(self.SEARCH_URL, params={
                    "action": "query",
                    "list": "search",
                    "srsearch": query,
                    "srlimit": 3,
                    "format": "json",
                    "srnamespace": 0
                })
                search_data = search_resp.json()
                results = search_data.get("query", {}).get("search", [])

                if not results:
                    return None

                # Get the top result's page summary
                title = results[0]["title"]
                summary_resp = await client.get(
                    f"{self.BASE_URL}/page/summary/{title.replace(' ', '_')}"
                )

                if summary_resp.status_code != 200:
                    return None

                data = summary_resp.json()
                extract = data.get("extract", "")

                if not extract or len(extract) < 50:
                    return None

                # Trim to max_chars
                if len(extract) > max_chars:
                    extract = extract[:max_chars] + "..."

                return {
                    "title": data.get("title", title),
                    "text": extract,
                    "url": data.get("content_urls", {}).get("desktop", {}).get("page", f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"),
                    "source": "wikipedia"
                }

        except Exception as e:
            print(f"Wikipedia search error: {e}")
            return None


class PubMedSearch:
    """Search PubMed for peer-reviewed medical literature."""

    ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    ESUMMARY_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"

    async def search(self, query: str, max_results: int = 3) -> Optional[dict]:
        """
        Search PubMed and return article abstracts.

        Returns:
            dict with 'text', 'url', 'title' or None if not found
        """
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                # Search for article IDs
                search_resp = await client.get(self.ESEARCH_URL, params={
                    "db": "pubmed",
                    "term": query,
                    "retmax": max_results,
                    "retmode": "json",
                    "sort": "relevance"
                })
                search_data = search_resp.json()
                ids = search_data.get("esearchresult", {}).get("idlist", [])

                if not ids:
                    return None

                # Fetch summaries for top result
                pmid = ids[0]
                summary_resp = await client.get(self.ESUMMARY_URL, params={
                    "db": "pubmed",
                    "id": pmid,
                    "retmode": "json"
                })
                summary_data = summary_resp.json()
                article = summary_data.get("result", {}).get(pmid, {})
                title = article.get("title", "")

                # Fetch abstract
                fetch_resp = await client.get(self.EFETCH_URL, params={
                    "db": "pubmed",
                    "id": pmid,
                    "rettype": "abstract",
                    "retmode": "text"
                })
                abstract_text = fetch_resp.text

                # Clean up abstract text
                abstract_text = re.sub(r'\n+', ' ', abstract_text).strip()
                if len(abstract_text) > 2000:
                    abstract_text = abstract_text[:2000] + "..."

                if not abstract_text or len(abstract_text) < 50:
                    return None

                return {
                    "title": title,
                    "text": abstract_text,
                    "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                    "source": "pubmed"
                }

        except Exception as e:
            print(f"PubMed search error: {e}")
            return None


class ExternalSearchService:
    """
    Orchestrates external search across multiple providers.
    Tries Wikipedia first, then PubMed.
    Caches results in-memory with a 24-hour TTL.
    Tracks metrics for monitoring.
    """

    CACHE_TTL = 86400  # 24 hours in seconds

    def __init__(self):
        self.wikipedia = WikipediaSearch()
        self.pubmed = PubMedSearch()
        # Cache: query -> (result, timestamp)
        self._cache: dict[str, tuple[Optional[dict], float]] = {}
        # Metrics
        self._metrics = {
            "total_searches": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "wikipedia_hits": 0,
            "pubmed_hits": 0,
            "no_results": 0,
        }

    def _cache_key(self, query: str) -> str:
        return query.strip().lower()

    def _get_cached(self, query: str) -> Optional[dict]:
        key = self._cache_key(query)
        entry = self._cache.get(key)
        if entry:
            result, ts = entry
            if time.time() - ts < self.CACHE_TTL:
                return result
            del self._cache[key]
        return None

    def _set_cache(self, query: str, result: Optional[dict]):
        self._cache[self._cache_key(query)] = (result, time.time())

    def get_metrics(self) -> dict:
        total = self._metrics["total_searches"]
        hit_rate = (
            round(self._metrics["cache_hits"] / total, 3) if total > 0 else 0.0
        )
        return {
            **self._metrics,
            "cache_size": len(self._cache),
            "cache_hit_rate": hit_rate,
        }

    async def search(self, query: str) -> Optional[dict]:
        """
        Search external sources in order: Wikipedia → PubMed.
        Results are cached for 24 hours.

        Returns:
            dict with 'text', 'url', 'title', 'source' or None if all fail
        """
        self._metrics["total_searches"] += 1

        # Check cache first
        cached = self._get_cached(query)
        if cached is not None:
            self._metrics["cache_hits"] += 1
            print(f"External search: cache hit for '{query[:50]}'")
            return cached

        self._metrics["cache_misses"] += 1

        # Try Wikipedia first (fastest, no rate limits)
        result = await self.wikipedia.search(query)
        if result:
            self._metrics["wikipedia_hits"] += 1
            print(f"External search: found result from Wikipedia for '{query[:50]}'")
            self._set_cache(query, result)
            return result

        # Try PubMed as fallback
        result = await self.pubmed.search(query)
        if result:
            self._metrics["pubmed_hits"] += 1
            print(f"External search: found result from PubMed for '{query[:50]}'")
            self._set_cache(query, result)
            return result

        self._metrics["no_results"] += 1
        print(f"External search: no results found for '{query[:50]}'")
        self._set_cache(query, None)
        return None
