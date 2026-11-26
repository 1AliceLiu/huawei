"""Publish cumulative count snapshots without blocking camera inference."""
import hashlib
from http.client import HTTPException
import json
import sqlite3
import threading
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class CountsPublisher(threading.Thread):
    def __init__(self, store, cameras, url, stop, token=""):
        super().__init__(name="counts-publisher", daemon=True)
        parsed = urlsplit(url)
        if (parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username
                or parsed.password or parsed.fragment or any(c.isspace() for c in url)
                or parsed.port == 0):
            raise ValueError("Counts webhook must be an HTTP(S) URL without credentials or fragment")
        if "\r" in token or "\n" in token:
            raise ValueError("Invalid counts webhook token")
        self.store, self.cameras, self.url, self.stop, self.token = store, cameras, url, stop, token
        self.destination = hashlib.sha256(url.encode()).hexdigest()
        self.client = build_opener(ProxyHandler({}), NoRedirect())

    def publish_once(self):
        snapshot = self.store.counts(self.cameras)
        body = json.dumps(snapshot, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        digest = hashlib.sha256(body).hexdigest()
        if self.store.counts_push_digest(self.destination) == digest:
            return False
        headers = {"Content-Type": "application/json", "Idempotency-Key": "counts:" + digest}
        if self.token:
            headers["Authorization"] = "Bearer " + self.token
        with self.client.open(Request(self.url, data=body, headers=headers, method="POST"), timeout=3) as response:
            if not 200 <= response.status < 300:
                raise HTTPError(self.url, response.status, "Webhook rejected snapshot", response.headers, None)
            status = response.status
        self.store.acknowledge_counts(self.destination, digest)
        print(f"[counts-push] revision={snapshot['revision']} status={status}", flush=True)
        return True

    def run(self):
        delay = 1
        while not self.stop.is_set():
            try:
                self.publish_once()
                delay = 1
            except (OSError, URLError, HTTPException, sqlite3.Error) as error:
                print(f"[counts-push] retry error={type(error).__name__} status={getattr(error, 'code', None)}", flush=True)
                delay = min(30, delay * 2)
            self.stop.wait(delay)
