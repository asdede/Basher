import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time

# Polite crawling: identify bot so sites can block if needed; avoid hammering servers
USER_AGENT = "BasherDocBot/1.0 (+https://github.com; docs indexing)"
DEFAULT_DELAY_SEC = 1.5  # delay between requests (per domain)
DEFAULT_MAX_PAGES = 50   # cap pages per crawl to avoid runaway and load on target
DEFAULT_MAX_DEPTH = 1   # keep shallow to limit scope

def scrape_page(url, domain=None, session=None, timeout=15):
    """Scrape a page and return text + links. Uses a polite User-Agent."""
    session = session or requests.Session()
    session.headers.setdefault("User-Agent", USER_AGENT)
    try:
        r = session.get(url, timeout=timeout)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        for s in soup(["script", "style"]):
            s.extract()
        text = soup.get_text(separator=" ")
        text = " ".join(text.split())

        links = []
        for a in soup.find_all("a", href=True):
            href = urljoin(url, a["href"])
            if domain is None or urlparse(href).netloc.endswith(domain):
                links.append(href)

        return text, links
    except Exception as e:
        print(f"Failed {url}: {e}")
        return "", []


def crawl_urls(
    start_urls,
    max_depth=DEFAULT_MAX_DEPTH,
    max_pages=DEFAULT_MAX_PAGES,
    delay_sec=DEFAULT_DELAY_SEC,
):
    """
    Crawl start_urls (and same-domain links up to max_depth).
    Stops after max_pages pages and waits delay_sec between each request
    to avoid overloading target sites.
    """
    visited = set()
    to_visit = [(url, 0) for url in start_urls]
    all_texts = []
    all_urls = []
    session = requests.Session()
    session.headers.setdefault("User-Agent", USER_AGENT)

    while to_visit and len(visited) < max_pages:
        url, depth = to_visit.pop(0)
        if url in visited or depth > max_depth:
            continue
        visited.add(url)

        domain = urlparse(url).netloc
        text, links = scrape_page(url, domain=domain, session=session)
        if text:
            all_texts.append(text)
            all_urls.append(url)

        for link in links:
            if link not in visited and len(visited) < max_pages:
                to_visit.append((link, depth + 1))

        time.sleep(delay_sec)

    if to_visit and len(visited) >= max_pages:
        print(f"Capped at {max_pages} pages; {len(to_visit)} URLs not crawled.")
    return all_texts, all_urls
