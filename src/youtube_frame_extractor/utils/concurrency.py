#!/usr/bin/env python3
"""
Concurrency Utility Module for YouTube Frame Extractor

This module provides helper functions and classes for parallel execution,
including threaded or process-based concurrency with optional progress reporting.
"""

import sys
import math
import time
import traceback
from typing import Callable, Iterable, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..logger import get_logger
from ..config import get_settings
from ..exceptions import ConcurrencyError

logger = get_logger(__name__)
settings = get_settings()


def parallel_map(
    func: Callable[[Any], Any],
    items: Iterable[Any],
    max_workers: Optional[int] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None
) -> List[Any]:
    """
    Apply a function to a list of items in parallel using threads.

    Args:
        func: The function to apply to each item.
        items: An iterable of items.
        max_workers: Maximum number of threads to use (default determined by ThreadPoolExecutor).
        progress_callback: An optional function that takes (completed_count, total_count).

    Returns:
        A list of results in the same order as the input items.

    Raises:
        ConcurrencyError: If any thread raises an unhandled exception.
    """
    items_list = list(items)
    total_count = len(items_list)
    if total_count == 0:
        return []

    logger.info(f"Starting parallel_map with {total_count} items, max_workers={max_workers}")

    results = [None] * total_count
    exceptions = [None] * total_count

    # Submit tasks
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {}
        for i, item in enumerate(items_list):
            future = executor.submit(_task_wrapper, func, item, i)
            future_to_index[future] = i

        completed_count = 0
        for future in as_completed(future_to_index):
            idx = future_to_index[future]
            completed_count += 1

            # Progress callback
            if progress_callback:
                progress_callback(completed_count, total_count)

            try:
                result_or_error = future.result()
                if isinstance(result_or_error, Exception):
                    exceptions[idx] = result_or_error
                else:
                    results[idx] = result_or_error
            except Exception as e:
                logger.error(f"Unhandled exception in parallel_map: {str(e)}")
                exceptions[idx] = e

    # Check for any errors
    for i, exc in enumerate(exceptions):
        if exc is not None:
            msg = f"Error in parallel_map for item index {i}: {str(exc)}"
            logger.error(msg)
            raise ConcurrencyError(msg)

    return results


def chunked_parallel_map(
    func: Callable[[List[Any]], List[Any]],
    items: Iterable[Any],
    chunk_size: int,
    max_workers: Optional[int] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None
) -> List[Any]:
    """
    Similar to parallel_map but processes items in chunks. The `func` is expected
    to handle a batch of items at once for more efficient processing.

    Args:
        func: Function that takes a list (chunk) of items and returns a list of results.
        items: Iterable of items to chunk.
        chunk_size: How many items to batch together.
        max_workers: Maximum threads.
        progress_callback: (completed_count, total_count).

    Returns:
        A list of results, flattened in the original order.

    Raises:
        ConcurrencyError: If any chunk processing fails.
    """
    items_list = list(items)
    total_count = len(items_list)
    if total_count == 0:
        return []

    # Partition items into chunks
    chunks = []
    for i in range(0, total_count, chunk_size):
        chunk = items_list[i : i + chunk_size]
        chunks.append((i, chunk))  # store starting index

    logger.info(f"Starting chunked_parallel_map with total {total_count} items, chunk_size={chunk_size}")

    # Placeholder for results
    results = [None] * total_count
    exceptions = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_start_idx = {}
        for (start_idx, chunk_data) in chunks:
            future = executor.submit(_chunk_task_wrapper, func, chunk_data, start_idx)
            future_to_start_idx[future] = (start_idx, len(chunk_data))

        completed_chunks = 0
        total_chunks = len(chunks)
        for future in as_completed(future_to_start_idx):
            (start_idx, size) = future_to_start_idx[future]
            completed_chunks += 1

            # Progress callback
            if progress_callback:
                progress_callback(completed_chunks, total_chunks)

            try:
                sub_results_or_error = future.result()
                if isinstance(sub_results_or_error, Exception):
                    exceptions.append((start_idx, sub_results_or_error))
                else:
                    # Place sub-results into the right positions
                    for i, val in enumerate(sub_results_or_error):
                        results[start_idx + i] = val
            except Exception as e:
                logger.error(f"Exception in chunked_parallel_map: {str(e)}")
                exceptions.append((start_idx, e))

    if exceptions:
        first_err_idx, first_exc = exceptions[0]
        msg = f"Error in chunked_parallel_map (chunk starting at {first_err_idx}): {str(first_exc)}"
        logger.error(msg)
        raise ConcurrencyError(msg)

    return results


def _task_wrapper(func, item, index):
    """
    Internal helper to wrap the call to func(item). 
    Returns either the result or the raised exception (as an object).
    """
    try:
        return func(item)
    except Exception as e:
        logger.debug(f"Exception in _task_wrapper at index={index}: {str(e)}")
        return e


def _chunk_task_wrapper(func, chunk_data, start_idx):
    """
    Internal helper for chunked tasks. Returns a list of results or an exception object.
    """
    try:
        return func(chunk_data)
    except Exception as e:
        logger.debug(f"Exception in _chunk_task_wrapper at start_idx={start_idx}: {str(e)}")
        return e
