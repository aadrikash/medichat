import os
import re

# Patterns to look for
SECRETS_PATTERNS = {
    "Groq Key": r"gsk_[a-zA-Z0-9]{40,}",
    "Pinecone Key": r"[a-z0-9-]{30,60}", # Generic pattern for UUID-style keys
}

def scan_files():
    found_secrets = False
    # Only scan relevant folders
    for root, dirs, files in os.walk("."):
        if "venv" in root or ".git" in root: continue
        
        for file in files:
            if file.endswith((".py", ".ipynb", ".env")):
                path = os.path.join(root, file)
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    for name, pattern in SECRETS_PATTERNS.items():
                        if re.search(pattern, content):
                            print(f"⚠️  POSSIBLE SECRET FOUND: {name} in {path}")
                            found_secrets = True
    
    if not found_secrets:
        print("✅ No hardcoded keys detected! You're good to go.")

if __name__ == "__main__":
    scan_files()