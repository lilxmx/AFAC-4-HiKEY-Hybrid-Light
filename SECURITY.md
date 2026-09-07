# Security

Do not commit API keys, access tokens, credentials, private datasets, or internal endpoints.

1. Copy `.env.example` to `.env` and fill in credentials locally.
2. Keep `.env` untracked; it is excluded by `.gitignore`.
3. If a secret is ever committed, revoke or rotate it immediately, then remove it from the complete Git history before publishing.

The public version intentionally omits runtime caches, generated outputs, and raw reference files.
