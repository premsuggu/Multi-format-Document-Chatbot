# web_extractor.py

import re
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse

def extract_urls(text):
    url_pattern = r'https?://\S+'
    return re.findall(url_pattern, text)

def is_valid_url(url: str) -> bool:
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except:
        return False

def extract_web_content(url: str) -> str:
    """Extract clean, readable text from a webpage."""
    if not is_valid_url(url):
        raise ValueError(f"Invalid URL: {url}")

    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
    except requests.RequestException as e:
        raise RuntimeError(f"Failed to fetch URL: {e}")

    soup = BeautifulSoup(response.text, "html.parser")

    # Remove scripts and styles
    for script_or_style in soup(["script", "style", "noscript"]):
        script_or_style.decompose()

    # Extract visible text
    visible_text = soup.get_text(separator="\n")
    clean_lines = [
        line.strip() for line in visible_text.splitlines() if line.strip()
    ]
    return "\n".join(clean_lines)
