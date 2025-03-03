#!/usr/bin/env python3
"""
Local Storage Module for YouTube Frame Extractor

This module provides a simple mechanism to store files locally
and handle file operations like copying, moving, or retrieving
metadata.
"""

import os
import shutil
from typing import Optional, List

from ..logger import get_logger
from ..config import get_settings
from ..exceptions import StorageError, FileWriteError

logger = get_logger(__name__)
settings = get_settings()


class LocalStorage:
    """
    A straightforward local file system storage implementation.
    Useful if you want to keep frames or metadata on the local disk
    in a structured way.

    Usage example:
        storage = LocalStorage(base_dir="/path/to/storage")
        storage.store_file("local_frames/frame1.jpg", "videos/video123/frame1.jpg")
    """

    def __init__(self, base_dir: Optional[str] = None):
        """
        Initialize local storage.

        Args:
            base_dir: The root directory for storage. Defaults to settings.storage.output_dir
        """
        self.base_dir = base_dir or settings.storage.output_dir
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir, exist_ok=True)
        logger.info(f"LocalStorage initialized with base_dir='{self.base_dir}'")

    def store_file(self, local_path: str, relative_path: str) -> str:
        """
        Store a local file into the storage structure by copying it to base_dir/relative_path.

        Args:
            local_path: Path to the source file on the local disk.
            relative_path: The sub-path in the storage directory structure.

        Returns:
            The absolute path where the file was stored.

        Raises:
            FileWriteError: If the file cannot be copied.
            StorageError: If the local source file does not exist.
        """
        if not os.path.exists(local_path):
            raise StorageError(f"Local source file not found: {local_path}")

        destination = os.path.join(self.base_dir, relative_path)
        os.makedirs(os.path.dirname(destination), exist_ok=True)

        try:
            shutil.copy2(local_path, destination)  # copies metadata as well
            logger.info(f"Stored file '{local_path}' at '{destination}'")
            return destination
        except Exception as e:
            logger.error(f"Error storing file to '{destination}': {str(e)}")
            raise FileWriteError(destination, context={"error": str(e)})

    def retrieve_file(self, relative_path: str, local_path: str) -> None:
        """
        Retrieve a stored file from base_dir/relative_path to a local destination.

        Args:
            relative_path: The sub-path inside the storage directory.
            local_path: The destination path on disk.

        Raises:
            StorageError: If the file is not found in storage.
            FileWriteError: If copying to local path fails.
        """
        source = os.path.join(self.base_dir, relative_path)
        if not os.path.exists(source):
            raise StorageError(f"File not found in local storage: {source}")

        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        try:
            shutil.copy2(source, local_path)
            logger.info(f"Retrieved file '{source}' to '{local_path}'")
        except Exception as e:
            logger.error(f"Error retrieving file to '{local_path}': {str(e)}")
            raise FileWriteError(local_path, context={"error": str(e)})

    def list_files(self, prefix: str = "") -> List[str]:
        """
        List all files under base_dir that match the given prefix.

        Args:
            prefix: Sub-directory or partial path under base_dir to filter.

        Returns:
            A list of file paths (relative to base_dir).
        """
        target_dir = os.path.join(self.base_dir, prefix)
        if not os.path.isdir(target_dir):
            logger.warning(f"Prefix directory '{target_dir}' does not exist")
            return []

        files_found = []
        for root, dirs, files in os.walk(target_dir):
            for f in files:
                full_path = os.path.join(root, f)
                relative = os.path.relpath(full_path, self.base_dir)
                files_found.append(relative)

        logger.info(f"Found {len(files_found)} files under prefix '{prefix}' in '{self.base_dir}'")
        return files_found

    def delete_file(self, relative_path: str) -> None:
        """
        Delete a file from local storage.

        Args:
            relative_path: The sub-path in the storage directory to delete.

        Raises:
            StorageError: If the file doesn't exist or cannot be deleted.
        """
        path_to_delete = os.path.join(self.base_dir, relative_path)
        if not os.path.exists(path_to_delete):
            raise StorageError(f"File not found: {path_to_delete}")

        try:
            os.remove(path_to_delete)
            logger.info(f"Deleted file '{path_to_delete}'")
        except Exception as e:
            logger.error(f"Error deleting file '{path_to_delete}': {str(e)}")
            raise StorageError(f"Could not delete file '{path_to_delete}': {str(e)}")
