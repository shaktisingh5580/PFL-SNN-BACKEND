import requests
import logging

logger = logging.getLogger(__name__)

def shorten_url(url: str) -> str:
    """Shortens a URL using TinyURL's anonymous API."""
    try:
        if not url or not url.startswith("http"):
            return url
            
        # TinyURL API is simple and doesn't require an API key for basic usage
        api_url = f"http://tinyurl.com/api-create.php?url={url}"
        response = requests.get(api_url, timeout=5)
        
        if response.status_code == 200:
            return response.text
        else:
            return url  # Fallback to original URL
    except Exception as e:
        logger.warning(f"URL Shortener failed: {e}")
        return url
