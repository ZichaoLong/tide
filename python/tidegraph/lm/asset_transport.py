"""Strict bounded HTTP ranges; transient retries never weaken byte identity."""
import http.client
import os
import time
import urllib.error
import urllib.request


class TransientRead(Exception):
    """A retryable read failure, with no URL or authentication in its message."""


def network_error(error):
    """Describe connection failure without leaking a signed URL or proxy auth."""
    reason=error.reason if isinstance(error,urllib.error.URLError) else error
    number=getattr(reason,'errno',None)
    return type(error).__name__+'/'+type(reason).__name__+(f'/errno={number}' if isinstance(number,int) else '')


def receive(url, partial, total, digest, progress, budget, *, opener=None,
            chunk_bytes=32*1024**2, retries=6, timeout_seconds=60, event=lambda _: None, sleep=time.sleep):
    if type(chunk_bytes) is not int or chunk_bytes <= 0 or type(retries) is not int or retries < 0:
        raise ValueError('download chunk_bytes must be positive; retries must be nonnegative')
    if type(timeout_seconds) not in (int,float) or not 0<timeout_seconds<=180:
        raise ValueError('download timeout_seconds must be positive and at most 180')
    opener = opener or urllib.request.build_opener()
    offset = partial.stat().st_size if partial.exists() else 0
    failures = 0
    while offset < total:
        end = min(total, offset+chunk_bytes)-1
        start = offset
        headers = {'User-Agent': 'tidegraph-lm-assets/2', 'Accept-Encoding': 'identity',
                   'Range': f'bytes={start}-{end}'}
        try:
            try:
                response = opener.open(urllib.request.Request(url, headers=headers), timeout=timeout_seconds)
            except urllib.error.HTTPError as error:
                if error.code in {408, 429, 500, 502, 503, 504}:
                    raise TransientRead(f'HTTP {error.code}') from None
                raise RuntimeError(f'asset HTTP {error.code}') from None
            except (OSError, http.client.HTTPException) as error:
                raise TransientRead(network_error(error)) from None
            with response:
                # A small whole-file response may legitimately ignore Range.
                whole = start == 0 and end == total-1 and response.status == 200
                if not whole and (response.status != 206 or response.headers.get('Content-Range') != f'bytes {start}-{end}/{total}'):
                    raise RuntimeError('server did not honor exact requested Range/total; partial retained')
                if response.headers.get('Content-Encoding', 'identity') != 'identity':
                    raise RuntimeError('asset response must use identity Content-Encoding')
                length = response.headers.get('Content-Length')
                if length is not None and length != str(end-start+1):
                    raise RuntimeError('asset Content-Length differs from requested range')
                with partial.open('ab') as stream:
                    try:
                        while offset <= end:
                            try:
                                block = response.read(min(1024**2, end-offset+1))
                            except (OSError, http.client.HTTPException) as error:
                                raise TransientRead(network_error(error)) from None
                            if not block:
                                raise TransientRead('early EOF')
                            if len(block) > end-offset+1:
                                raise RuntimeError('asset body exceeds requested range')
                            stream.write(block); digest.update(block); offset += len(block)
                    finally:
                        stream.flush(); os.fsync(stream.fileno())
            budget(0)
            progress(offset)
        except TransientRead as error:
            failures += 1
            event(dict(kind='transient_read', failure=failures, offset=offset,
                       reason=str(error), retries_allowed=retries))
            progress(offset)
            if failures > retries:
                raise RuntimeError(f'asset transient retry budget exhausted after {failures} failures; partial retained') from None
            sleep(min(30, 2**failures))
        # Retry budget is per file, not reset after every successful segment.
    return offset
