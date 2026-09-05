import time
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

# Browser-like headers so doc sites (e.g. gnu.org) are less likely to return 403
USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)
DEFAULT_HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}
DEFAULT_DELAY_SEC = 1.5


def _session():
    """Session with browser-like headers to reduce 403 from strict sites."""
    s = requests.Session()
    for k, v in DEFAULT_HEADERS.items():
        s.headers.setdefault(k, v)
    return s


def _extract_main_text(soup, url=""):
    """
    Prefer main documentation content over site chrome (nav/footer).
    Site-specific selectors first, then generic <main>/<article>, else body.
    """
    host = urlparse(url).netloc.lower() if url else ""

    # Drop non-content early
    for s in soup(["script", "style", "noscript"]):
        s.extract()

    root = None
    if "git-scm.com" in host:
        root = soup.select_one("#main") or soup.select_one("#content")
    elif "man7.org" in host:
        # Man pages are a sequence of <pre> blocks after the title
        pres = soup.find_all("pre")
        if pres:
            return " ".join(" ".join(p.get_text(separator=" ").split()) for p in pres)
        for sel in (".nav-bar", ".page-top", "header", "nav", "footer"):
            for el in soup.select(sel):
                el.extract()
    elif "linuxcommand.org" in host:
        root = soup.find("pre") or soup.find("div", id="content")
    elif "docs.docker.com" in host:
        root = soup.find("article", class_=lambda c: c and "prose" in (c or ""))
        if not root:
            root = soup.find(attrs={"data-pagefind-body": True})

    if root is None:
        root = (
            soup.find("main")
            or soup.find("article")
            or soup.find(attrs={"role": "main"})
            or soup.body
            or soup
        )
        # Strip common chrome when falling back to body
        if root is soup.body or root is soup:
            for sel in ("nav", "header", "footer", ".nav-bar", ".page-top", "#header", "#footer"):
                for el in (root.select(sel) if hasattr(root, "select") else []):
                    el.extract()

    text = root.get_text(separator=" ") if root else soup.get_text(separator=" ")
    return " ".join(text.split())


def scrape_page(url, session=None, timeout=15, retries=3):
    """Fetch one URL and return its main text (no link following). Retries on 403/5xx."""
    session = session or _session()
    parsed = urlparse(url)
    origin = f"{parsed.scheme}://{parsed.netloc}/"
    last_err = None
    for attempt in range(max(1, retries)):
        try:
            r = session.get(url, timeout=timeout, headers={"Referer": origin})
            if r.status_code in (403, 429, 500, 502, 503):
                last_err = f"{r.status_code} for url: {url}"
                time.sleep(1.2 * (attempt + 1))
                continue
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "html.parser")
            return _extract_main_text(soup, url=url)
        except Exception as e:
            last_err = e
            time.sleep(0.8 * (attempt + 1))
    print(f"Failed {url}: {last_err}")
    return ""


def scrape_urls_only(url_list, delay_sec=DEFAULT_DELAY_SEC):
    """
    Scrape exactly the given URLs: one page per URL, no following links.
    Returns (list of page texts, list of urls that were fetched).
    Waits delay_sec between requests to avoid overloading servers.
    """
    if not url_list:
        return [], []
    session = _session()
    texts = []
    fetched = []
    for url in url_list:
        text = scrape_page(url, session=session)
        if text:
            texts.append(text)
            fetched.append(url)
        time.sleep(delay_sec)
    return texts, fetched


def scrape_man_page_index(index_url, fetch_details=True, delay_sec=DEFAULT_DELAY_SEC):
    """
    Scrape a linuxcommand.org-style man page index: parse tables for
    (command, description, detail_url), then optionally fetch each detail page.
    Returns list of {"command": str, "description": str, "text": str}
    where "text" is the full detail page content (or empty if not fetched).
    """
    session = _session()
    base = urlparse(index_url)
    base_url = f"{base.scheme}://{base.netloc}/"

    try:
        r = session.get(index_url, timeout=15)
        r.raise_for_status()
    except Exception as e:
        print(f"Failed to fetch man index {index_url}: {e}")
        return []

    soup = BeautifulSoup(r.text, "html.parser")
    entries = []

    for table in soup.find_all("table"):
        tbody = table.find("tbody")
        rows = (tbody.find_all("tr") if tbody else table.find_all("tr"))
        for tr in rows:
            cells = tr.find_all("td")
            if len(cells) < 2:
                continue
            first = cells[0]
            second = cells[1]
            link = first.find("a", href=True)
            if not link:
                continue
            command = link.get_text(strip=True) or ""
            if not command:
                continue
            description = second.get_text(strip=True) or ""
            href = link["href"]
            detail_url = urljoin(index_url, href)
            entries.append({
                "command": command,
                "description": description,
                "detail_url": detail_url,
                "text": "",
            })

    # Fallback: if no table rows found (e.g. site uses different layout), collect links to detail pages
    if not entries:
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if "lc3_man_pages/" in href or "man_pages/" in href:
                command = a.get_text(strip=True) or ""
                if command and len(command) < 80:  # skip nav/text blobs
                    detail_url = urljoin(index_url, href)
                    entries.append({
                        "command": command,
                        "description": "",
                        "detail_url": detail_url,
                        "text": "",
                    })

    if not fetch_details or not entries:
        return entries

    for i, entry in enumerate(entries):
        text = scrape_page(entry["detail_url"], session=session)
        entry["text"] = text
        if (i + 1) % 10 == 0:
            print(f"  Fetched {i + 1}/{len(entries)} detail pages...")
        time.sleep(delay_sec)

    return entries


def _extract_docker_cli_main_text(soup):
    """
    Extract main documentation content from a Docker docs page.
    Ignores sidebar, header, footer, TOC. Returns (title, description, full_text).
    """
    # Main content is in <article class="prose ..."> or [data-pagefind-body]
    article = soup.find("article", class_=lambda c: c and "prose" in (c or ""))
    if not article:
        article = soup.find(attrs={"data-pagefind-body": True})
    if not article:
        return "", "", ""

    for s in article(["script", "style", "nav", "button"]):
        s.extract()

    title = ""
    h1 = article.find("h1")
    if h1:
        title = h1.get_text(strip=True) or ""

    description = ""
    # First table often has Description row: <th>Description</th><td>...</td>
    for table in article.find_all("table"):
        for tr in table.find_all("tr"):
            cells = tr.find_all(["th", "td"])
            if len(cells) >= 2:
                first_text = (cells[0].get_text(strip=True) or "").lower()
                if "description" in first_text:
                    description = cells[1].get_text(strip=True) or ""
                    break
        if description:
            break

    text = " ".join(article.get_text(separator=" ").split())
    return title, description, text


def scrape_docker_cli_index(index_url, fetch_details=True, delay_sec=DEFAULT_DELAY_SEC):
    """
    Scrape Docker CLI reference: parse sidebar nav for command page URLs (ignore TOC),
    then optionally fetch each detail page. Same structure as man_pages: command, description, text.
    Returns list of {"command": str, "description": str, "text": str, "source": str (url)}
    """
    session = _session()
    base = urlparse(index_url)
    base_url = f"{base.scheme}://{base.netloc}"

    try:
        r = session.get(index_url, timeout=15)
        r.raise_for_status()
    except Exception as e:
        print(f"Failed to fetch Docker CLI index {index_url}: {e}")
        return []

    soup = BeautifulSoup(r.text, "html.parser")
    entries = []
    seen_urls = set()

    # Find all sidebar-style links to CLI reference pages (same domain, /reference/cli/ path)
    for a in soup.find_all("a", href=True):
        href = (a["href"] or "").strip()
        if not href or href.startswith("#"):
            continue
        if "reference/cli/" not in href:
            continue
        full_url = urljoin(base_url, href)
        if full_url in seen_urls:
            continue
        # Prefer same host as index (e.g. docs.docker.com)
        if base.netloc not in full_url:
            continue
        command = (a.get("title") or a.get_text(strip=True) or "").strip()
        if not command:
            continue
        seen_urls.add(full_url)
        entries.append({
            "command": command,
            "description": "",
            "detail_url": full_url,
            "text": "",
            "source": full_url,
        })

    if not fetch_details or not entries:
        return entries

    for i, entry in enumerate(entries):
        try:
            r = session.get(entry["detail_url"], timeout=15)
            r.raise_for_status()
            page_soup = BeautifulSoup(r.text, "html.parser")
            for s in page_soup(["script", "style"]):
                s.extract()
            title, desc, text = _extract_docker_cli_main_text(page_soup)
            if title and not entry["command"].strip():
                entry["command"] = title
            if desc:
                entry["description"] = desc
            entry["text"] = text
        except Exception as e:
            print(f"  Failed {entry['detail_url']}: {e}")
        if (i + 1) % 15 == 0:
            print(f"  Fetched {i + 1}/{len(entries)} Docker CLI pages...")
        time.sleep(delay_sec)

    return entries
