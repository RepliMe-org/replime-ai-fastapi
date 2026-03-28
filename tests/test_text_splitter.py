from core.text_splitter import split_transcript
from core.exceptions import ChunkingError

# --- happy path ---
print("TEST 1: Normal transcript")
sample = "هذا نص تجريبي. " * 100  # repeat to get enough characters
chunks = split_transcript(sample)
print(f"  chunks: {len(chunks)}")
print(f"  first chunk length: {len(chunks[0])} chars")
print(f"  sample: {chunks[0][:80]}")
assert len(chunks) > 1, "Should produce multiple chunks"
print("  PASSED\n")

# --- edge case: empty string ---
print("TEST 2: Empty transcript")
try:
    split_transcript("")
    print("  FAILED — should have raised ChunkingError")
except ChunkingError as e:
    print(f"  PASSED — got ChunkingError: {e}\n")

# --- edge case: very short text ---
print("TEST 3: Very short transcript")
chunks = split_transcript("مرحبا")
print(f"  chunks: {len(chunks)}")
assert len(chunks) == 1
print("  PASSED\n")

print("All text splitter tests passed.")