#!/usr/bin/env python3
"""Restore one immutable Hugging Face dataset version into Cloudflare R2."""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import boto3
from botocore.exceptions import ClientError
from huggingface_hub import hf_hub_download


def required(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


FAMILY_ID = required("FAMILY_ID")
VERSION = int(required("VERSION"))
HF_REPOSITORY = required("HF_DATASET_REPOSITORY")
HF_TOKEN = os.environ.get("HF_TOKEN", "").strip() or None
R2_BUCKET_NAME = required("R2_BUCKET_NAME")

JOB_KEY = f"_control/r2-pins/{FAMILY_ID}/v{VERSION}.json"
MANIFEST_KEY = f"datasets/{FAMILY_ID}/v{VERSION}/manifest.json"
CATALOG_KEY = "_control/catalog.json"


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


s3 = boto3.client(
    "s3",
    endpoint_url=f"https://{required('R2_ACCOUNT_ID')}.r2.cloudflarestorage.com",
    region_name="auto",
    aws_access_key_id=required("R2_ACCESS_KEY_ID"),
    aws_secret_access_key=required("R2_SECRET_ACCESS_KEY"),
)


def get_json(key: str) -> tuple[dict, str | None]:
    response = s3.get_object(Bucket=R2_BUCKET_NAME, Key=key)
    return json.loads(response["Body"].read()), response.get("ETag")


def put_json(key: str, value: dict, *, etag: str | None = None) -> None:
    parameters = {
        "Bucket": R2_BUCKET_NAME,
        "Key": key,
        "Body": json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8"),
        "ContentType": "application/json",
        "CacheControl": "no-cache" if key.endswith("/manifest.json") else "no-store",
    }
    if etag:
        parameters["IfMatch"] = etag
    s3.put_object(**parameters)


def update_job(job: dict, status: str, *, current: str = "", error: str = "") -> None:
    job["status"] = status
    job["currentAsset"] = current
    job["error"] = error
    job["updatedAt"] = now()
    put_json(JOB_KEY, job)
    completed = sum(1 for item in job.get("files", []) if item.get("status") == "verified")
    print(f"[{status}] {completed}/{len(job.get('files', []))} {current}", flush=True)


def r2_size(key: str) -> int | None:
    try:
        return int(s3.head_object(Bucket=R2_BUCKET_NAME, Key=key)["ContentLength"])
    except ClientError as error:
        if error.response.get("ResponseMetadata", {}).get("HTTPStatusCode") == 404:
            return None
        raise


def download_hf_file(repository: str, revision: str, path: str) -> Path:
    return Path(hf_hub_download(
        repo_id=repository,
        filename=path,
        repo_type="dataset",
        revision=revision,
        token=HF_TOKEN,
    ))


def restore_file(job: dict, item: dict) -> None:
    expected = int(item["size"])
    if r2_size(item["key"]) == expected:
        item["status"] = "verified"
        item["verifiedAt"] = item.get("verifiedAt") or now()
        return

    item["status"] = "copying"
    item["attempts"] = int(item.get("attempts", 0)) + 1
    update_job(job, "copying", current=item["path"])
    local_path = download_hf_file(job["datasetRepository"], job["revision"], item["path"])
    actual = local_path.stat().st_size
    if actual != expected:
        raise RuntimeError(f"HF size mismatch for {item['path']}: {actual} != {expected}")
    s3.upload_file(
        str(local_path),
        R2_BUCKET_NAME,
        item["key"],
        ExtraArgs={
            "ContentType": item.get("contentType") or "application/octet-stream",
            "CacheControl": "public, max-age=31536000, immutable",
        },
    )
    restored = r2_size(item["key"])
    if restored != expected:
        raise RuntimeError(f"R2 size mismatch for {item['key']}: {restored} != {expected}")
    item["status"] = "verified"
    item["verifiedAt"] = now()
    update_job(job, "copying", current=item["path"])


def activate_manifest(job: dict) -> None:
    manifest, etag = get_json(MANIFEST_KEY)
    storage = manifest.setdefault("storage", {})
    providers = storage.setdefault("providers", {})
    if "huggingface" not in providers:
        providers["huggingface"] = {
            key: storage[key]
            for key in (
                "repository", "repositoryUrl", "revision", "manifestRevision",
                "versionUrl", "archivedAt", "bytes", "sourceProvider",
            )
            if key in storage
        }
    providers["r2"] = {
        "bytes": int(job["totalBytes"]),
        "pinnedAt": now(),
        "verifiedAt": now(),
    }
    storage["provider"] = "huggingface"
    manifest["updatedAt"] = now()
    put_json(MANIFEST_KEY, manifest, etag=etag)


def complete_catalog(job: dict) -> None:
    for attempt in range(8):
        catalog, etag = get_json(CATALOG_KEY)
        record = catalog.setdefault("datasets", {}).setdefault(
            f"{FAMILY_ID}:v{VERSION}", {"bytes": 0, "storage": "huggingface"}
        )
        reserved = int(record.get("r2ReservedBytes", 0))
        already_used = int(record.get("r2Bytes", 0))
        total = int(job["totalBytes"])
        catalog["reservedBytes"] = max(0, int(catalog.get("reservedBytes", 0)) - reserved)
        catalog["usedBytes"] = int(catalog.get("usedBytes", 0)) + max(0, total - already_used)
        record.update({
            "r2ReservedBytes": 0,
            "r2Bytes": total,
            "r2PinStatus": "completed",
        })
        try:
            put_json(CATALOG_KEY, catalog, etag=etag)
            return
        except ClientError as error:
            status = error.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
            if status not in (409, 412) or attempt == 7:
                raise
    raise RuntimeError("Unable to update the R2 usage catalog")


def main() -> None:
    job, _ = get_json(JOB_KEY)
    if job.get("datasetRepository") != HF_REPOSITORY:
        raise RuntimeError("R2 pin job targets a different Hugging Face repository")
    if job.get("status") == "completed":
        print("R2 pin already completed.", flush=True)
        return

    update_job(job, "copying")
    for item in job.get("files", []):
        restore_file(job, item)

    update_job(job, "verifying")
    for item in job.get("files", []):
        actual = r2_size(item["key"])
        if actual != int(item["size"]):
            raise RuntimeError(f"Final R2 verification failed for {item['key']}")
        item["status"] = "verified"

    activate_manifest(job)
    complete_catalog(job)
    job["completedAt"] = now()
    update_job(job, "completed")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:  # noqa: BLE001 - persist background job failure for the management UI
        try:
            failed_job, _ = get_json(JOB_KEY)
            update_job(
                failed_job,
                "failed",
                current=failed_job.get("currentAsset", ""),
                error=str(error),
            )
        except Exception as status_error:  # noqa: BLE001
            print(f"Unable to persist failed status: {status_error}", file=sys.stderr)
        raise
