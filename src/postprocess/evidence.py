# 근거 url

from typing import List, Dict, Any


def extract_urls(citations: List[Dict[str, Any]]) -> List[str]:

    urls = []

    for c in citations:
        url = c.get("source_url")
        if url and url not in urls:
            urls.append(url)

    return urls
