# create_env.py

import os

# Replace this with your actual NVIDIA API key
nvidia_api_key = "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"

# Create or overwrite the .env file
with open(".env", "w") as env_file:
    env_file.write(f"NVIDIA_API_KEY={nvidia_api_key}\n")

print(".env file created successfully!")