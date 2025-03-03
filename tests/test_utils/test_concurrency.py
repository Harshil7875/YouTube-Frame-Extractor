import pytest
from youtube_frame_extractor.utils import concurrency

# Assume that utils/concurrency.py defines a function:
# def map_parallel(func, data, max_workers=4, timeout=60):
#     # Implementation that applies func to each item in data in parallel
#     # and returns a list of results.
#     ...

def square(x):
    return x * x

def test_map_parallel():
    data = [1, 2, 3, 4]
    expected = [1, 4, 9, 16]
    
    # Run the function using a limited number of workers.
    results = concurrency.map_parallel(square, data, max_workers=2)
    
    # Assert that results match expected output.
    assert results == expected
