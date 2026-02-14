#!/usr/bin/env python3
"""Performance benchmarking script for Transistor Database.

Measures:
- Import time
- JSON load time
- Core model operations
"""
import time
import json
from pathlib import Path


def measure_import_time():
    """Measure time to import transistordatabase package."""
    start = time.perf_counter()
    import transistordatabase  # noqa: F401
    end = time.perf_counter()
    return (end - start) * 1000  # Convert to milliseconds


def measure_core_import_time():
    """Measure time to import core module."""
    start = time.perf_counter()
    from transistordatabase.core import Transistor  # noqa: F401
    end = time.perf_counter()
    return (end - start) * 1000


def measure_json_load_time():
    """Measure time to load a transistor from JSON."""
    from transistordatabase.core.repository import JsonTransistorLoader

    # Find a test JSON file
    test_data_dir = Path(__file__).parent / "tests" / "test_data" / "database"
    json_files = list(test_data_dir.glob("*.json"))

    if not json_files:
        return None, "No test JSON files found"

    test_file = json_files[0]
    loader = JsonTransistorLoader()

    start = time.perf_counter()
    transistor = loader.load_from_json(test_file)
    end = time.perf_counter()

    load_time = (end - start) * 1000
    return load_time, transistor.metadata.name


def measure_json_save_time():
    """Measure time to save a transistor to JSON."""
    from transistordatabase.core.repository import JsonTransistorLoader, TransistorFactory
    import tempfile

    # Create a minimal transistor with valid manufacturer
    transistor = TransistorFactory.create_empty_transistor("Benchmark_Test", "MOSFET")
    transistor.metadata.manufacturer = "infineon"
    transistor.metadata.housing_type = "to220"
    transistor.metadata.author = "Benchmark"

    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "benchmark.json"
        loader = JsonTransistorLoader()

        start = time.perf_counter()
        loader.save_to_json(transistor, test_file)
        end = time.perf_counter()

    return (end - start) * 1000


def measure_repository_operations():
    """Measure repository CRUD operations."""
    from transistordatabase.core.repository import JsonTransistorRepository, TransistorFactory
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        repo = JsonTransistorRepository(Path(tmpdir))

        # Create with valid metadata
        transistor = TransistorFactory.create_empty_transistor("Bench_MOSFET", "MOSFET")
        transistor.metadata.manufacturer = "infineon"
        transistor.metadata.housing_type = "to220"
        transistor.metadata.author = "Benchmark"
        start = time.perf_counter()
        repo.save(transistor)
        end = time.perf_counter()
        save_time = (end - start) * 1000

        # Read
        start = time.perf_counter()
        loaded = repo.get_by_name("Bench_MOSFET")
        end = time.perf_counter()
        read_time = (end - start) * 1000

        # List
        start = time.perf_counter()
        names = repo.list_all()
        end = time.perf_counter()
        list_time = (end - start) * 1000

        # Delete
        start = time.perf_counter()
        repo.delete("Bench_MOSFET")
        end = time.perf_counter()
        delete_time = (end - start) * 1000

    return {
        "save": save_time,
        "read": read_time,
        "list": list_time,
        "delete": delete_time,
    }


def main():
    """Run all benchmarks and print results."""
    print("=" * 70)
    print("Transistor Database Performance Benchmarks")
    print("=" * 70)
    print()

    # Import times
    print("Import Performance:")
    print("-" * 70)
    import_time = measure_import_time()
    print(f"  Full package import:     {import_time:8.2f} ms")

    core_import_time = measure_core_import_time()
    print(f"  Core module import:      {core_import_time:8.2f} ms")
    print()

    # JSON operations
    print("JSON Operations:")
    print("-" * 70)

    load_time, transistor_name = measure_json_load_time()
    if load_time is not None:
        print(f"  Load from JSON ({transistor_name}):")
        print(f"                           {load_time:8.2f} ms")
    else:
        print(f"  Load from JSON:          {transistor_name}")

    print("  Save to JSON:            (skipped - requires legacy adapter)")
    print()

    # Repository operations (skipped - require legacy adapter for save)
    print("Repository Operations (CRUD):")
    print("-" * 70)
    print("  (Skipped - require legacy adapter with all fields)")
    print()

    # Summary
    print("Performance Summary:")
    print("-" * 70)

    status = []
    if import_time < 2000:
        status.append(f"✓ Import time: {import_time:.0f}ms (target: <2000ms)")
    else:
        status.append(f"✗ Import time: {import_time:.0f}ms (target: <2000ms)")

    if load_time is not None:
        if load_time < 100:
            status.append(f"✓ JSON load: {load_time:.0f}ms (target: <100ms)")
        else:
            status.append(f"✗ JSON load: {load_time:.0f}ms (target: <100ms)")

    for line in status:
        print(f"  {line}")

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
