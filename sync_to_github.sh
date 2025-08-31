#!/bin/bash

# Navigate to your project folder (edit this if needed)
cd ~/ai-protest-detection || exit

# Stage all changes
git add .

# Commit with timestamp
COMMIT_MSG="Auto-sync: $(date '+%Y-%m-%d %H:%M:%S')"
git commit -m "$COMMIT_MSG"

# Push to main branch
git push origin main

echo "✅ Synced to GitHub with message: $COMMIT_MSG"
