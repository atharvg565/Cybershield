from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

# Feed files: one URL or domain per line. Lines starting with # are comments.
# Put downloaded OpenPhish or URLhaus dumps here, or use the sample file below.
FEED_FILES = [
    DATA_DIR / "openphish_feed.txt",
    DATA_DIR / "urlhaus_feed.txt",
    DATA_DIR / "custom_blocklist.txt",
]


def _normalise(entry: str) -> tuple[str, str]:
    """Return (full_url_lowercase, domain_lowercase) for an entry."""
    entry = entry.strip().lower()
    if not entry or entry.startswith("#"):
        return "", ""
    if "://" not in entry:
        entry = "http://" + entry
    host = urlparse(entry).hostname or ""
    host = host[4:] if host.startswith("www.") else host
    return entry.rstrip("/"), host


@lru_cache(maxsize=1)
def load_feeds() -> tuple[frozenset[str], frozenset[str]]:
    """Load all feed files once. Returns (known_urls, known_domains)."""
    urls, domains = set(), set()
    for path in FEED_FILES:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            full, host = _normalise(line)
            if full:
                urls.add(full)
            if host:
                domains.add(host)
    return frozenset(urls), frozenset(domains)


def check_feeds(urls: list[str]) -> set[str]:
    """
    Return the subset of urls that appear in a threat feed.
    A URL matches if its full form is listed, or its domain (or a parent domain) is listed.
    """
    known_urls, known_domains = load_feeds()
    hits = set()
    for url in urls:
        full, host = _normalise(url)
        if not full:
            continue
        if full in known_urls:
            hits.add(url)
            continue
        # Check the domain and each parent domain, e.g. a.b.evil.com -> b.evil.com -> evil.com
        parts = host.split(".")
        for i in range(len(parts) - 1):
            if ".".join(parts[i:]) in known_domains:
                hits.add(url)
                break
    return hits


if __name__ == "__main__":
    test_urls = [
        "http://sbi-secure.xyz/login",
        "https://www.flipkart.com/orders",
        "http://evil-bank.example.com/verify",
    ]
    print("Loaded feeds:", {k: len(v) for k, v in zip(["urls", "domains"], load_feeds())})
    print("Hits:", check_feeds(test_urls))