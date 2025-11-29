# Clean the uv cache (This deletes the downloaded .DS_Store file):
uv cache clean

# Delete your current environment (It's likely half-broken):
Remove-Item -Recurse -Force .venv

# OneDrive's "cloud sync" system conflicts with uv's default behavior, which tries to create "hardlinks" (shortcuts) to save space. OneDrive sees these shortcuts and blocks them, triggering os error 396.
# The Fix
# Run the sync command with the copy mode flag to force uv to physically copy the files instead of linking them.
# Run this exact command in your terminal:
python -m uv sync --link-mode=copy

# This is the "Golden Rule" of Git that trips everyone up: .gitignore only ignores files that are NOT currently tracked.

# If you (or the system) committed .env in the past, Git is now "watching" it. Adding it to .gitignore afterwards does not make Git stop watching it; it just prevents it from being added again if it were new.

# Since your .env is white (not gray), Git is still tracking it.

# The Fix: Force Git to "Forget" the File
# Run these three commands in your terminal, one by one. This will remove the file from Git's memory but keep the file on your hard drive.

# 1. Remove from Git's Index:
git rm --cached .env

# 2. Verify it worked: Run this command:
git status
# You should see something like:
# deleted: .env (This means it's deleted from Git)
# You should NOT see .env listed under "Untracked files" (because it is now successfully ignored).

# 3. Commit the "Deletion": You must save this change for it to take effect.
git commit -m "Stop tracking .env file"

# After running these, the .env file in your VS Code explorer should instantly turn gray. 
#If it doesn't, reload your VS Code window (Ctrl+R or Cmd+R).