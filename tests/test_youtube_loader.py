from core.youtube_loader import load_transcript

url = "https://youtu.be/2KwMUvfW9bo?si=U8nl0fefha2vAaqW"

print("Loading transcript...")
transcript = load_transcript(url)

print(f"\nLength: {len(transcript)} characters")
print(f"\nFirst 500 chars:\n{transcript[:500]}")