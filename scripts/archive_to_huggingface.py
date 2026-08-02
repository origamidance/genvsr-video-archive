#!/usr/bin/env python3
"""Move one published GenVSR dataset version from R2/GitHub to Hugging Face."""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

import boto3
import requests
from huggingface_hub import HfApi


def required(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


FAMILY_ID = required("FAMILY_ID")
VERSION = int(required("VERSION"))
HF_REPOSITORY = required("HF_DATASET_REPOSITORY")
HF_TOKEN = required("HF_TOKEN")
R2_ACCOUNT_ID = required("R2_ACCOUNT_ID")
R2_BUCKET_NAME = required("R2_BUCKET_NAME")
MEDIA_BASE_URL = required("MEDIA_BASE_URL").rstrip("/")

JOB_KEY = f"_control/hf-archives/{FAMILY_ID}/v{VERSION}.json"
MANIFEST_KEY = f"datasets/{FAMILY_ID}/v{VERSION}/manifest.json"
CATALOG_KEY = "_control/catalog.json"


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


s3 = boto3.client(
    "s3",
    endpoint_url=f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com",
    region_name="auto",
    aws_access_key_id=required("R2_ACCESS_KEY_ID"),
    aws_secret_access_key=required("R2_SECRET_ACCESS_KEY"),
)
hf = HfApi(token=HF_TOKEN)


def get_json(key: str) -> dict:
    response = s3.get_object(Bucket=R2_BUCKET_NAME, Key=key)
    return json.loads(response["Body"].read())


def put_json(key: str, value: dict) -> None:
    value["updatedAt"] = now()
    s3.put_object(
        Bucket=R2_BUCKET_NAME,
        Key=key,
        Body=json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8"),
        ContentType="application/json",
        CacheControl="no-cache" if key.endswith("/manifest.json") else "no-store",
    )


def update_job(job: dict, status: str, *, current: str = "", error: str = "") -> None:
    job["status"] = status
    job["currentAsset"] = current
    job["error"] = error
    put_json(JOB_KEY, job)
    completed = sum(1 for item in job.get("files", []) if item.get("status") == "uploaded")
    print(f"[{status}] {completed}/{len(job.get('files', []))} {current}", flush=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_http(url: str, destination: Path, expected_size: int) -> None:
    for attempt in range(8):
        offset = destination.stat().st_size if destination.exists() else 0
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        try:
            with requests.get(url, headers=headers, stream=True, timeout=(30, 180), allow_redirects=True) as response:
                if offset and response.status_code == 200:
                    destination.unlink(missing_ok=True)
                    offset = 0
                elif response.status_code not in (200, 206):
                    raise RuntimeError(f"HTTP {response.status_code}")
                mode = "ab" if offset and response.status_code == 206 else "wb"
                with destination.open(mode) as handle:
                    for chunk in response.iter_content(8 * 1024 * 1024):
                        if chunk:
                            handle.write(chunk)
            if destination.stat().st_size == expected_size:
                return
            raise RuntimeError(f"downloaded {destination.stat().st_size}, expected {expected_size}")
        except Exception as exc:  # noqa: BLE001 - retry network failures with persisted bytes
            if attempt == 7:
                raise RuntimeError(f"Unable to download {url}: {exc}") from exc
            time.sleep(min(2**attempt, 30))


def download_file(item: dict, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if item.get("sourceUrl"):
        download_http(item["sourceUrl"], destination, int(item["size"]))
    else:
        s3.download_file(R2_BUCKET_NAME, item["key"], str(destination))
    actual = destination.stat().st_size
    if actual != int(item["size"]):
        raise RuntimeError(f"Size mismatch for {item['key']}: {actual} != {item['size']}")


def remote_sizes(paths: list[str]) -> dict[str, int]:
    sizes: dict[str, int] = {}
    for index in range(0, len(paths), 100):
        batch = paths[index : index + 100]
        try:
            entries = hf.get_paths_info(HF_REPOSITORY, batch, repo_type="dataset")
        except Exception:  # noqa: BLE001 - missing paths and transient metadata errors trigger a safe upload
            continue
        for entry in entries:
            path = getattr(entry, "path", "")
            size = getattr(entry, "size", None)
            if path and size is not None:
                sizes[path] = int(size)
    return sizes


def hf_retry_delay(error: Exception, attempt: int) -> int:
    message = str(error)
    match = re.search(r"retry after\s+(\d+)\s+seconds", message, flags=re.IGNORECASE)
    if match:
        return min(int(match.group(1)) + 5, 65 * 60)
    if "repository commits" in message and "128 per hour" in message:
        return 65 * 60
    return min(2**attempt * 15, 5 * 60)


def upload_folder_with_retry(folder: Path) -> None:
    for attempt in range(16):
        try:
            hf.upload_folder(
                folder_path=str(folder),
                repo_id=HF_REPOSITORY,
                repo_type="dataset",
                commit_message=f"Archive {FAMILY_ID} v{VERSION}",
                ignore_patterns=[".cache/**"],
            )
            return
        except Exception as exc:  # noqa: BLE001 - honor Hub limits and retry transient Xet/network failures
            if attempt == 15:
                raise RuntimeError(f"Unable to upload dataset folder: {exc}") from exc
            delay = hf_retry_delay(exc, attempt)
            print(f"Hugging Face upload paused for {delay}s: {exc}", flush=True)
            time.sleep(delay)


def upload_manifest_with_retry(payload: bytes, path_in_repo: str):
    for attempt in range(16):
        try:
            return hf.upload_file(
                path_or_fileobj=io.BytesIO(payload),
                path_in_repo=path_in_repo,
                repo_id=HF_REPOSITORY,
                repo_type="dataset",
                commit_message=f"Publish manifest for {FAMILY_ID} v{VERSION}",
            )
        except Exception as exc:  # noqa: BLE001 - final metadata commit must survive rate limits too
            if attempt == 15:
                raise RuntimeError(f"Unable to publish dataset manifest: {exc}") from exc
            delay = hf_retry_delay(exc, attempt)
            print(f"Hugging Face manifest commit paused for {delay}s: {exc}", flush=True)
            time.sleep(delay)


def encode_hf_proxy(path: str, revision: str) -> str:
    owner, repository = HF_REPOSITORY.split("/", 1)
    encoded_path = "/".join(quote(segment, safe="") for segment in path.split("/"))
    return (
        f"{MEDIA_BASE_URL}/hf/datasets/{quote(owner, safe='')}/{quote(repository, safe='')}"
        f"/resolve/{quote(revision, safe='')}/{encoded_path}"
    )


def write_dataset_card(folder: Path) -> bool:
    if hf.file_exists(HF_REPOSITORY, "README.md", repo_type="dataset"):
        return False
    card = """---
license: other
task_categories:
- video-classification
tags:
- video-super-resolution
- visual-comparison
pretty_name: GenVSR Video Benchmarks
---

# GenVSR Video Benchmarks

Public video inputs and restoration outputs used by the GenVSR Visual Comparator.
Each immutable experiment version contains aligned MP4 sources, posters, and a manifest.
Please consult the individual version manifest for source labels and comparison defaults.
"""
    (folder / "README.md").write_text(card, encoding="utf-8")
    return True


def update_catalog(job: dict) -> None:
    catalog = get_json(CATALOG_KEY)
    record = catalog.get("datasets", {}).get(f"{FAMILY_ID}:v{VERSION}")
    if not record:
        return
    if record.get("storage") in (None, "r2"):
        catalog["usedBytes"] = max(0, int(catalog.get("usedBytes", 0)) - int(job["totalBytes"]))
        record["bytes"] = max(0, int(record.get("bytes", 0)) - int(job["totalBytes"]))
        record["archivedBytes"] = int(job["totalBytes"])
    record["storage"] = "huggingface"
    put_json(CATALOG_KEY, catalog)


def main() -> None:
    job = get_json(JOB_KEY)
    if job.get("datasetRepository") != HF_REPOSITORY:
        raise RuntimeError("Archive job targets a different Hugging Face repository")
    if job.get("status") == "completed":
        print("Archive already completed.", flush=True)
        return

    hf.create_repo(HF_REPOSITORY, repo_type="dataset", private=False, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="genvsr-hf-") as temp_directory:
        temp_root = Path(temp_directory)
        include_card = write_dataset_card(temp_root)
        paths = [item["path"] for item in job["files"]]
        existing = remote_sizes(paths)
        pending: list[dict] = []
        for item in job["files"]:
            if existing.get(item["path"]) == int(item["size"]):
                item["status"] = "uploaded"
                continue
            item["status"] = "downloading"
            item["attempts"] = int(item.get("attempts", 0)) + 1
            update_job(job, "downloading", current=item["path"])
            local_path = temp_root / item["path"]
            download_file(item, local_path)
            item["sha256"] = sha256_file(local_path)
            item["status"] = "ready"
            pending.append(item)

        if pending or include_card:
            update_job(job, "uploading", current=f"{len(pending)} files in one batched upload")
            upload_folder_with_retry(temp_root)

        verified = remote_sizes(paths)
        for item in job["files"]:
            if verified.get(item["path"]) != int(item["size"]):
                raise RuntimeError(f"Remote size verification failed for {item['path']}")
            item["status"] = "uploaded"
            item["uploadedAt"] = item.get("uploadedAt") or now()
        update_job(job, "uploading")

    update_job(job, "finalizing")
    media_revision = hf.dataset_info(HF_REPOSITORY).sha
    manifest = get_json(MANIFEST_KEY)
    by_key = {item["key"]: item for item in job["files"]}
    for sequence in manifest["sequences"]:
        poster = by_key[sequence["posterKey"]]
        sequence["posterUrl"] = encode_hf_proxy(poster["path"], media_revision)
        for media in sequence.get("sources", {}).values():
            item = by_key[media["key"]]
            media["url"] = encode_hf_proxy(item["path"], media_revision)

    repository_url = f"https://huggingface.co/datasets/{HF_REPOSITORY}"
    prefix = f"datasets/{FAMILY_ID}/v{VERSION}"
    manifest["storage"] = {
        "provider": "huggingface",
        "repository": HF_REPOSITORY,
        "repositoryUrl": repository_url,
        "revision": media_revision,
        "archivedAt": now(),
        "bytes": int(job["totalBytes"]),
        "bytesFreed": int(job["totalBytes"]) if job.get("sourceProvider") == "r2" else 0,
        "sourceProvider": job.get("sourceProvider", "r2"),
    }
    commit = upload_manifest_with_retry(
        json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8"),
        f"{prefix}/manifest.json",
    )
    manifest_revision = commit.oid
    version_url = f"{repository_url}/tree/{manifest_revision}/{prefix}"
    manifest["storage"]["manifestRevision"] = manifest_revision
    manifest["storage"]["versionUrl"] = version_url
    put_json(MANIFEST_KEY, manifest)

    if job.get("sourceProvider") == "r2":
        for index in range(0, len(job["files"]), 1000):
            batch = job["files"][index : index + 1000]
            s3.delete_objects(
                Bucket=R2_BUCKET_NAME,
                Delete={"Objects": [{"Key": item["key"]} for item in batch], "Quiet": True},
            )
    update_catalog(job)
    job.update(
        {
            "status": "completed",
            "currentAsset": "",
            "error": "",
            "completedAt": now(),
            "revision": media_revision,
            "manifestRevision": manifest_revision,
            "repositoryUrl": repository_url,
            "versionUrl": version_url,
        }
    )
    put_json(JOB_KEY, job)
    print(version_url, flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:  # noqa: BLE001 - persist actionable failure for the web UI
        try:
            failed_job = get_json(JOB_KEY)
            update_job(failed_job, "failed", current=failed_job.get("currentAsset", ""), error=str(error))
        except Exception as status_error:  # noqa: BLE001
            print(f"Unable to persist failed status: {status_error}", file=sys.stderr)
        raise
