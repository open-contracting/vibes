#!/usr/bin/env python3
"""
Download all files from a OneDrive personal shared folder (recursively).

No login or API key required. The folder has been migrated to SharePoint
Online, so the old anonymous api.onedrive.com endpoints return 401. Instead we
replicate what the OneDrive web app does:

  1. Mint an anonymous "Badger" token from api-badgerp.svc.ms.
  2. Resolve the share link to a driveItem, passing `Prefer: autoredeem` so the
     anonymous token is granted access to the shared folder.
  3. Walk the folder tree via the /children endpoint (paginated).
  4. Download every file via its pre-authenticated @content.downloadUrl,
     mirroring the folder structure locally.
"""

import argparse
import base64
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from http import HTTPStatus
from pathlib import Path

OUTPUT_DIR = "./data"

API = "https://my.microsoftpersonalcontent.com/_api/v2.0"
BADGER_TOKEN_URL = "https://api-badgerp.svc.ms/v1.0/token"  # noqa: S105  # non-password
BADGER_APP_ID = "00000000-0000-0000-0000-0000481710a4"  # OneDrive web app id

MAX_RETRIES = 6
TRANSIENT_STATUS = {429, 500, 502, 503, 504}

# Fields requested per item; @content.downloadUrl is the anonymous download link.
SELECT = "id,name,size,folder,file,@content.downloadUrl"


def _request(url, *, data=None, headers=None, method=None, timeout=60):
    req = urllib.request.Request(url, data=data, headers=headers or {}, method=method)  # noqa: S310
    return urllib.request.urlopen(req, timeout=timeout)  # noqa: S310


def _backoff(attempt, error):
    """Sleep before retrying a transient failure, honouring Retry-After."""
    delay = min(60, 2**attempt)
    if isinstance(error, urllib.error.HTTPError):
        retry_after = error.headers.get("Retry-After")
        if retry_after and retry_after.isdigit():
            delay = max(delay, int(retry_after))
    print(f"  ...transient {error}; retrying in {delay}s", file=sys.stderr)
    time.sleep(delay)


class OneDrive:
    """Anonymous client for a OneDrive shared folder; holds the Badger token across requests."""

    def __init__(self):
        self.token = None

    def mint_token(self):
        """Get a fresh anonymous Badger token."""
        resp = _request(
            BADGER_TOKEN_URL,
            data=json.dumps({"appId": BADGER_APP_ID}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        self.token = json.loads(resp.read())["token"]
        return self.token

    def api_get(self, url, *, data=None, method=None, prefer=None):
        """GET/POST an API endpoint, re-minting on 401 and backing off on throttling."""
        for attempt in range(MAX_RETRIES):
            headers = {
                "Authorization": "Badger " + self.token,
                "Accept": "application/json",
                "Referer": "https://onedrive.live.com/",
            }
            if prefer:
                headers["Prefer"] = prefer
            try:
                resp = _request(url, data=data, headers=headers, method=method)
                return json.loads(resp.read())
            except urllib.error.HTTPError as e:
                if e.code == HTTPStatus.UNAUTHORIZED:
                    self.mint_token()
                    continue
                if e.code in TRANSIENT_STATUS and attempt < MAX_RETRIES - 1:
                    _backoff(attempt, e)
                    continue
                raise
            except OSError as e:
                if attempt < MAX_RETRIES - 1:
                    _backoff(attempt, e)
                    continue
                raise
        return None

    def list_children(self, drive_id, item_id):
        """Yield all child items of a folder, following pagination."""
        select = urllib.parse.quote(SELECT, safe=",@")
        url = f"{API}/drives/{drive_id}/items/{item_id}/children?$top=200&$select={select}"
        while url:
            page = self.api_get(url)
            yield from page.get("value", [])
            url = page.get("@odata.nextLink")

    def walk_and_download(self, drive_id, item_id, rel_dir, stats, skip_existing, output_dir):
        for item in self.list_children(drive_id, item_id):
            name = item["name"]
            if "folder" in item:
                self.walk_and_download(
                    drive_id,
                    item["id"],
                    rel_dir / name,
                    stats,
                    skip_existing,
                    output_dir,
                )
                continue
            dest = output_dir / rel_dir / name
            stats["found"] += 1
            if skip_existing and dest.exists() and dest.stat().st_size == item.get("size"):
                print(f"  skip (exists): {rel_dir / name}")
                stats["skipped"] += 1
                continue
            url = item.get("@content.downloadUrl")
            if not url:
                print(f"  WARN no downloadUrl: {rel_dir / name}")
                continue
            print(f"  -> {rel_dir / name} ({item.get('size', 0):,} bytes)")
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_name(dest.name + ".part")
            for attempt in range(MAX_RETRIES):
                try:
                    with _request(url, headers={"Referer": "https://onedrive.live.com/"}) as resp, tmp.open("wb") as f:
                        while True:
                            chunk = resp.read(1 << 16)
                            if not chunk:
                                break
                            f.write(chunk)
                    tmp.replace(dest)
                    break
                except urllib.error.HTTPError as e:
                    if e.code in TRANSIENT_STATUS and attempt < MAX_RETRIES - 1:
                        _backoff(attempt, e)
                        continue
                    raise
                except OSError as e:
                    if attempt < MAX_RETRIES - 1:
                        _backoff(attempt, e)
                        continue
                    raise
            stats["downloaded"] += 1


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("share_url", help="a OneDrive personal share link (1drv.ms/...)")
    parser.add_argument("--output", default=OUTPUT_DIR)
    parser.add_argument("--no-skip", action="store_true", help="re-download existing files")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    client = OneDrive()
    print("Minting anonymous token...")
    client.mint_token()
    print("Resolving share link...")
    encoded = base64.b64encode(args.share_url.encode()).decode().rstrip("=")
    share_id = "u!" + encoded.replace("+", "-").replace("/", "_")
    endpoint = f"{API}/shares/{share_id}/driveitem?$select=id,name,parentReference"
    item = client.api_get(endpoint, data=b"", method="POST", prefer="autoredeem")
    drive_id, item_id = item["parentReference"]["driveId"], item["id"]
    print(f"Root folder: {item.get('name')!r}  (drive {drive_id}, item {item_id})")

    stats = {"found": 0, "downloaded": 0, "skipped": 0}
    try:
        client.walk_and_download(
            drive_id,
            item_id,
            Path(),
            stats,
            skip_existing=not args.no_skip,
            output_dir=output_dir,
        )
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)

    print(f"\nDone. {stats['found']} file(s) found, {stats['downloaded']} downloaded, {stats['skipped']} skipped.")


if __name__ == "__main__":
    main()
