import hashlib
import ipaddress
import socket
import sqlite3
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener


DEFAULT_PARSER_VERSION = 'trafilatura-2.2.0'


class WebVisitError(RuntimeError):
    pass


class URLAccessPolicy:
    def __init__(self, allow_private_network=False):
        self.allow_private_network = allow_private_network

    def validate(self, url):
        normalized_url = normalize_url(url)
        if self.allow_private_network:
            return normalized_url
        parsed = urlsplit(normalized_url)
        try:
            addresses = socket.getaddrinfo(
                parsed.hostname,
                parsed.port or (443 if parsed.scheme == 'https' else 80),
                type=socket.SOCK_STREAM,
            )
        except socket.gaierror as exc:
            raise WebVisitError(f'Cannot resolve web host: {parsed.hostname}') from exc
        for address in addresses:
            ip = ipaddress.ip_address(address[4][0])
            if not ip.is_global:
                raise WebVisitError(
                    f'Web Visit blocked non-public address for {parsed.hostname}: {ip}'
                )
        return normalized_url


class _PolicyRedirectHandler(HTTPRedirectHandler):
    def __init__(self, access_policy):
        self.access_policy = access_policy
        super().__init__()

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        self.access_policy.validate(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class _TitleParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.in_title = False
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag.lower() == 'title':
            self.in_title = True

    def handle_endtag(self, tag):
        if tag.lower() == 'title':
            self.in_title = False

    def handle_data(self, data):
        if self.in_title:
            self.parts.append(data)

    @property
    def title(self):
        return ' '.join(' '.join(self.parts).split()) or None


def normalize_url(url):
    try:
        parsed = urlsplit(url)
        port = parsed.port
    except ValueError as exc:
        raise WebVisitError(f'Invalid URL: {url}') from exc
    scheme = parsed.scheme.lower()
    if scheme not in {'http', 'https'} or not parsed.hostname:
        raise WebVisitError('Visit only supports absolute HTTP(S) URLs')
    if parsed.username or parsed.password:
        raise WebVisitError('URLs containing credentials are not supported')

    hostname = parsed.hostname.lower()
    if ':' in hostname:
        hostname = f'[{hostname}]'
    default_port = (scheme == 'http' and port == 80) or (scheme == 'https' and port == 443)
    netloc = hostname if port is None or default_port else f'{hostname}:{port}'
    path = parsed.path or '/'
    return urlunsplit((scheme, netloc, path, parsed.query, ''))


def fetch_html(
    url,
    timeout=20.0,
    max_bytes=5_000_000,
    allow_private_network=False,
):
    access_policy = URLAccessPolicy(allow_private_network=allow_private_network)
    url = access_policy.validate(url)
    opener = build_opener(ProxyHandler({}), _PolicyRedirectHandler(access_policy))
    request = Request(
        url,
        headers={
            'User-Agent': 'ReActLocalRAG/1.0 (+research web visit)',
            'Accept': 'text/html,application/xhtml+xml',
        },
    )
    try:
        with opener.open(request, timeout=timeout) as response:
            content_type = response.headers.get_content_type().lower()
            if content_type not in {'text/html', 'application/xhtml+xml'}:
                raise WebVisitError(f'Unsupported content type: {content_type}')
            body = response.read(max_bytes + 1)
            if len(body) > max_bytes:
                raise WebVisitError(f'Web page exceeds the {max_bytes}-byte limit')
            encoding = response.headers.get_content_charset() or 'utf-8'
            return {
                'final_url': access_policy.validate(response.geturl()),
                'status_code': response.status,
                'content_type': content_type,
                'html': body.decode(encoding, errors='replace'),
            }
    except WebVisitError:
        raise
    except HTTPError as exc:
        raise WebVisitError(f'Web request failed with HTTP {exc.code}: {url}') from exc
    except URLError as exc:
        raise WebVisitError(f'Cannot fetch {url}: {exc.reason}') from exc
    except TimeoutError as exc:
        raise WebVisitError(f'Web request timed out: {url}') from exc


def extract_main_content(html):
    try:
        import trafilatura
    except ImportError as exc:
        raise WebVisitError(
            "Live web Visit requires trafilatura; install the 'retrieval' or 'server' extra"
        ) from exc

    content = trafilatura.extract(
        html,
        include_comments=False,
        include_tables=True,
        favor_precision=True,
    )
    if not content or not content.strip():
        raise WebVisitError('No readable main content was extracted from the page')
    title_parser = _TitleParser()
    title_parser.feed(html)
    return {'title': title_parser.title, 'content': content.strip()}


class WebDocumentStore:
    '''Fetch, clean, and persistently cache static HTML documents.'''

    def __init__(
        self,
        cache_path,
        fetcher=None,
        extractor=None,
        parser_version=DEFAULT_PARSER_VERSION,
        allow_private_network=False,
    ):
        self.cache_path = Path(cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.fetcher = fetcher or (
            lambda url: fetch_html(
                url,
                allow_private_network=allow_private_network,
            )
        )
        self.extractor = extractor or extract_main_content
        self.parser_version = parser_version
        self._initialize_cache()

    def _connect(self):
        return sqlite3.connect(self.cache_path, timeout=30)

    def _initialize_cache(self):
        with self._connect() as connection:
            connection.execute('PRAGMA journal_mode=WAL')
            connection.execute('''
                CREATE TABLE IF NOT EXISTS web_documents (
                    normalized_url TEXT NOT NULL,
                    parser_version TEXT NOT NULL,
                    final_url TEXT NOT NULL,
                    title TEXT,
                    content TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    fetched_at TEXT NOT NULL,
                    status_code INTEGER NOT NULL,
                    content_type TEXT NOT NULL,
                    PRIMARY KEY (normalized_url, parser_version)
                )
            ''')

    def visit(self, url):
        normalized_url = normalize_url(url)
        cached = self._read_cached(normalized_url)
        if cached is not None:
            cached['cache_hit'] = True
            return cached

        fetched = self.fetcher(normalized_url)
        extracted = self.extractor(fetched['html'])
        content = extracted['content'].strip()
        if not content:
            raise WebVisitError('No readable main content was extracted from the page')
        document = {
            'doc_id': None,
            'url': fetched['final_url'],
            'normalized_url': normalized_url,
            'title': extracted.get('title'),
            'content': content,
            'content_hash': hashlib.sha256(content.encode('utf-8')).hexdigest(),
            'fetched_at': datetime.now(timezone.utc).isoformat(),
            'status_code': int(fetched['status_code']),
            'content_type': fetched['content_type'],
            'parser_version': self.parser_version,
            'source': 'live_web',
            'cache_hit': False,
        }
        self._write_cached(document)
        return document

    def _read_cached(self, normalized_url):
        with self._connect() as connection:
            row = connection.execute('''
                SELECT final_url, title, content, content_hash, fetched_at,
                       status_code, content_type
                FROM web_documents
                WHERE normalized_url = ? AND parser_version = ?
            ''', (normalized_url, self.parser_version)).fetchone()
        if row is None:
            return None
        return {
            'doc_id': None,
            'url': row[0],
            'normalized_url': normalized_url,
            'title': row[1],
            'content': row[2],
            'content_hash': row[3],
            'fetched_at': row[4],
            'status_code': row[5],
            'content_type': row[6],
            'parser_version': self.parser_version,
            'source': 'live_web',
        }

    def _write_cached(self, document):
        with self._connect() as connection:
            connection.execute('''
                INSERT INTO web_documents (
                    normalized_url, parser_version, final_url, title, content,
                    content_hash, fetched_at, status_code, content_type
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(normalized_url, parser_version) DO UPDATE SET
                    final_url = excluded.final_url,
                    title = excluded.title,
                    content = excluded.content,
                    content_hash = excluded.content_hash,
                    fetched_at = excluded.fetched_at,
                    status_code = excluded.status_code,
                    content_type = excluded.content_type
            ''', (
                document['normalized_url'],
                document['parser_version'],
                document['url'],
                document['title'],
                document['content'],
                document['content_hash'],
                document['fetched_at'],
                document['status_code'],
                document['content_type'],
            ))
