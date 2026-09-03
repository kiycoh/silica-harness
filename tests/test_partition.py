import json
from silica.kernel.partition import partition_by_concepts

def test_partition_by_concept_count():
    # Setup payload with 5 concepts
    payload = {
        "schema_version": 1,
        "batches": [
            {
                "inbox_file": "inbox1.md",
                "concepts": [
                    {"name": f"concept_{i}"} for i in range(5)
                ]
            }
        ]
    }
    
    # Split with max_concepts = 2
    chunks = partition_by_concepts(payload, max_concepts=2, max_bytes=999999)
    
    assert len(chunks) == 3
    assert len(chunks[0]["batches"][0]["concepts"]) == 2
    assert len(chunks[1]["batches"][0]["concepts"]) == 2
    assert len(chunks[2]["batches"][0]["concepts"]) == 1
    
    # Verify exact contents
    assert chunks[0]["batches"][0]["concepts"][0]["name"] == "concept_0"
    assert chunks[0]["batches"][0]["concepts"][1]["name"] == "concept_1"
    assert chunks[1]["batches"][0]["concepts"][0]["name"] == "concept_2"
    assert chunks[1]["batches"][0]["concepts"][1]["name"] == "concept_3"
    assert chunks[2]["batches"][0]["concepts"][0]["name"] == "concept_4"


def test_partition_by_bytes():
    # Setup concept entries with specific sizes
    # We will test splitting when max_bytes limit is reached
    concept_a = {"name": "A", "excerpt": "x" * 100}
    concept_b = {"name": "B", "excerpt": "x" * 100}
    concept_c = {"name": "C", "excerpt": "x" * 100}
    
    payload = {
        "schema_version": 1,
        "batches": [
            {
                "inbox_file": "inbox1.md",
                "concepts": [concept_a, concept_b, concept_c]
            }
        ]
    }
    
    # An empty chunk size is around 50-60 bytes. Adding a concept adds details.
    # We will set a max_bytes size that fits 1 or 2 concepts but not 3.
    two_chunk = {"schema_version": 1, "batches": [{"inbox_file": "inbox1.md", "concepts": [concept_a, concept_b]}]}
    two_size = len(json.dumps(two_chunk, ensure_ascii=False).encode('utf-8'))
    
    # Set limit in between one-concept size and two-concept size
    chunks = partition_by_concepts(payload, max_concepts=0, max_bytes=two_size - 1)
    
    # Should split into 3 chunks
    assert len(chunks) == 3
    assert chunks[0]["batches"][0]["concepts"] == [concept_a]
    assert chunks[1]["batches"][0]["concepts"] == [concept_b]
    assert chunks[2]["batches"][0]["concepts"] == [concept_c]


def test_partition_preserves_determinism_and_order():
    payload = {
        "schema_version": 1,
        "batches": [
            {
                "inbox_file": "inbox1.md",
                "concepts": [{"name": "A"}, {"name": "B"}]
            },
            {
                "inbox_file": "inbox2.md",
                "concepts": [{"name": "C"}, {"name": "D"}]
            }
        ]
    }
    
    chunks = partition_by_concepts(payload, max_concepts=1, max_bytes=999999)
    assert len(chunks) == 4
    
    assert chunks[0]["batches"][0]["inbox_file"] == "inbox1.md"
    assert chunks[0]["batches"][0]["concepts"] == [{"name": "A"}]
    
    assert chunks[1]["batches"][0]["inbox_file"] == "inbox1.md"
    assert chunks[1]["batches"][0]["concepts"] == [{"name": "B"}]
    
    assert chunks[2]["batches"][0]["inbox_file"] == "inbox2.md"
    assert chunks[2]["batches"][0]["concepts"] == [{"name": "C"}]
    
    assert chunks[3]["batches"][0]["inbox_file"] == "inbox2.md"
    assert chunks[3]["batches"][0]["concepts"] == [{"name": "D"}]


def test_single_large_concept_exceeds_max_bytes():
    # A single concept that exceeds max_bytes should still be put in a chunk
    # rather than failing or looping infinitely
    concept_huge = {"name": "Huge", "excerpt": "x" * 200}
    payload = {
        "schema_version": 1,
        "batches": [
            {
                "inbox_file": "inbox1.md",
                "concepts": [concept_huge]
            }
        ]
    }
    
    # Set limit to very small (e.g. 50 bytes)
    chunks = partition_by_concepts(payload, max_concepts=0, max_bytes=50)
    assert len(chunks) == 1
    assert chunks[0]["batches"][0]["concepts"] == [concept_huge]
