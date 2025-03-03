#!/usr/bin/env python3
"""
Cloud Storage Module for YouTube Frame Extractor

This module provides classes to upload/download files to AWS S3 or Google Cloud Storage.
It uses the project-wide config settings to determine which cloud provider to use,
and can be extended for Azure or other providers in the future.
"""

import os
from typing import Optional

import boto3
from botocore.exceptions import BotoCoreError, ClientError

try:
    from google.cloud import storage as gcs
except ImportError:
    # If GCS is not installed, this is okay as we might only be using AWS
    gcs = None

from ..logger import get_logger
from ..config import get_settings
from ..exceptions import CloudStorageError, StorageError

logger = get_logger(__name__)
settings = get_settings()


class CloudStorage:
    """
    A unified interface to handle file uploads/downloads to different cloud providers.
    Currently supports AWS S3 and Google Cloud Storage.
    
    Usage Example:
        storage = CloudStorage()
        storage.store_file(local_path="path/to/file.jpg", remote_path="my-frames/file.jpg")
        storage.retrieve_file(remote_path="my-frames/file.jpg", local_path="downloads/file.jpg")
    """

    def __init__(self, provider: Optional[str] = None):
        """
        Initialize the cloud storage interface.

        Args:
            provider: "aws" or "gcp". If None, uses settings.storage.cloud_provider.
        """
        self.provider = provider or settings.storage.cloud_provider
        if not self.provider:
            raise CloudStorageError("No cloud provider specified or configured.")

        self._client = None
        self._bucket_name = None

        self._init_provider()

    def _init_provider(self):
        """Initialize the appropriate cloud provider client."""
        provider = self.provider.lower()
        logger.info(f"Initializing CloudStorage for provider '{provider}'")

        if provider == "aws":
            self._init_aws_client()
        elif provider == "gcp":
            self._init_gcp_client()
        else:
            raise CloudStorageError(f"Unsupported cloud provider: {provider}", provider=provider)

    def _init_aws_client(self):
        """Initialize AWS S3 client."""
        bucket = settings.storage.aws_bucket
        region = settings.storage.aws_region

        if not bucket:
            raise CloudStorageError("AWS bucket not specified in settings", provider="aws")

        try:
            self._client = boto3.client("s3", region_name=region)
            self._bucket_name = bucket
            logger.info(f"AWS S3 client initialized for bucket '{bucket}' in region '{region}'")
        except (BotoCoreError, ClientError) as e:
            logger.error(f"Error initializing AWS S3 client: {str(e)}")
            raise CloudStorageError(f"Error initializing AWS S3 client: {str(e)}", provider="aws")

    def _init_gcp_client(self):
        """Initialize Google Cloud Storage client."""
        bucket = settings.storage.gcs_bucket
        if not gcs:
            raise CloudStorageError("google-cloud-storage is not installed", provider="gcp")
        if not bucket:
            raise CloudStorageError("GCS bucket not specified in settings", provider="gcp")

        try:
            self._client = gcs.Client()  # Uses default service account or env creds
            self._bucket_name = bucket
            logger.info(f"GCS client initialized for bucket '{bucket}'")
        except Exception as e:
            logger.error(f"Error initializing GCS client: {str(e)}")
            raise CloudStorageError(f"Error initializing GCS client: {str(e)}", provider="gcp")

    def store_file(self, local_path: str, remote_path: str) -> None:
        """
        Upload a local file to the configured cloud storage.

        Args:
            local_path: Path to the local file on disk.
            remote_path: Desired path/key in the cloud bucket.

        Raises:
            CloudStorageError: If upload fails.
        """
        provider = self.provider.lower()
        logger.info(f"Uploading '{local_path}' to provider '{provider}' at '{remote_path}'")

        if not os.path.exists(local_path):
            raise StorageError(f"Local file not found: {local_path}")

        if provider == "aws":
            self._upload_aws(local_path, remote_path)
        elif provider == "gcp":
            self._upload_gcp(local_path, remote_path)
        else:
            raise CloudStorageError(f"Unsupported provider: {provider}", provider=provider)

    def retrieve_file(self, remote_path: str, local_path: str) -> None:
        """
        Download a file from the configured cloud storage.

        Args:
            remote_path: Path/key in the cloud bucket.
            local_path: Local path to save the downloaded file.

        Raises:
            CloudStorageError: If download fails.
        """
        provider = self.provider.lower()
        logger.info(f"Downloading '{remote_path}' from provider '{provider}' to '{local_path}'")

        if provider == "aws":
            self._download_aws(remote_path, local_path)
        elif provider == "gcp":
            self._download_gcp(remote_path, local_path)
        else:
            raise CloudStorageError(f"Unsupported provider: {provider}", provider=provider)

    def list_files(self, prefix: str = "") -> list:
        """
        List files in the cloud storage bucket under a given prefix.

        Args:
            prefix: Prefix/folder path to list from.

        Returns:
            A list of file paths/keys in the bucket.

        Raises:
            CloudStorageError: If listing fails or is unsupported.
        """
        provider = self.provider.lower()
        logger.info(f"Listing files in provider '{provider}' with prefix '{prefix}'")

        if provider == "aws":
            return self._list_aws(prefix)
        elif provider == "gcp":
            return self._list_gcp(prefix)
        else:
            raise CloudStorageError(f"Unsupported provider: {provider}", provider=provider)

    # === AWS-specific methods ===

    def _upload_aws(self, local_path: str, remote_path: str):
        try:
            self._client.upload_file(local_path, self._bucket_name, remote_path)
            logger.info(f"Successfully uploaded '{local_path}' to 's3://{self._bucket_name}/{remote_path}'")
        except (BotoCoreError, ClientError) as e:
            logger.error(f"Error uploading to S3: {str(e)}")
            raise CloudStorageError(f"Error uploading to S3: {str(e)}", provider="aws")

    def _download_aws(self, remote_path: str, local_path: str):
        try:
            self._client.download_file(self._bucket_name, remote_path, local_path)
            logger.info(f"Successfully downloaded 's3://{self._bucket_name}/{remote_path}' to '{local_path}'")
        except (BotoCoreError, ClientError) as e:
            logger.error(f"Error downloading from S3: {str(e)}")
            raise CloudStorageError(f"Error downloading from S3: {str(e)}", provider="aws")

    def _list_aws(self, prefix: str = "") -> list:
        try:
            file_paths = []
            paginator = self._client.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=self._bucket_name, Prefix=prefix):
                for obj in page.get("Contents", []):
                    file_paths.append(obj["Key"])
            logger.info(f"Found {len(file_paths)} objects in 's3://{self._bucket_name}/{prefix}'")
            return file_paths
        except (BotoCoreError, ClientError) as e:
            logger.error(f"Error listing S3 bucket: {str(e)}")
            raise CloudStorageError(f"Error listing S3 bucket: {str(e)}", provider="aws")

    # === GCP-specific methods ===

    def _upload_gcp(self, local_path: str, remote_path: str):
        if not self._client:
            raise CloudStorageError("GCS client not initialized", provider="gcp")

        try:
            bucket = self._client.bucket(self._bucket_name)
            blob = bucket.blob(remote_path)
            blob.upload_from_filename(local_path)
            logger.info(f"Successfully uploaded '{local_path}' to 'gs://{self._bucket_name}/{remote_path}'")
        except Exception as e:
            logger.error(f"Error uploading to GCS: {str(e)}")
            raise CloudStorageError(f"Error uploading to GCS: {str(e)}", provider="gcp")

    def _download_gcp(self, remote_path: str, local_path: str):
        if not self._client:
            raise CloudStorageError("GCS client not initialized", provider="gcp")

        try:
            bucket = self._client.bucket(self._bucket_name)
            blob = bucket.blob(remote_path)
            blob.download_to_filename(local_path)
            logger.info(f"Successfully downloaded 'gs://{self._bucket_name}/{remote_path}' to '{local_path}'")
        except Exception as e:
            logger.error(f"Error downloading from GCS: {str(e)}")
            raise CloudStorageError(f"Error downloading from GCS: {str(e)}", provider="gcp")

    def _list_gcp(self, prefix: str = "") -> list:
        if not self._client:
            raise CloudStorageError("GCS client not initialized", provider="gcp")

        try:
            bucket = self._client.bucket(self._bucket_name)
            blobs = bucket.list_blobs(prefix=prefix)
            file_paths = [blob.name for blob in blobs]
            logger.info(f"Found {len(file_paths)} objects in 'gs://{self._bucket_name}/{prefix}'")
            return file_paths
        except Exception as e:
            logger.error(f"Error listing GCS bucket: {str(e)}")
            raise CloudStorageError(f"Error listing GCS bucket: {str(e)}", provider="gcp")
