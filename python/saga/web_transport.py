"""HTTP transport restricted to web URLs, including redirects."""

from urllib.error import URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


def _require_web_url(url: str) -> None:
    parts = urlsplit(url)
    if parts.scheme not in {"http", "https"} or not parts.hostname:
        raise URLError("Expected an HTTP or HTTPS URL")


class _WebRedirectHandler(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _require_web_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


_opener = build_opener(_WebRedirectHandler())


def open_web_request(request: Request, *, timeout: float):
    _require_web_url(request.full_url)
    return _opener.open(request, timeout=timeout)
