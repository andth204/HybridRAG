"""Mint an access token for an evaluation bot user.

Idempotent: re-running upserts the user row and issues a fresh access
token. Intended for benchmark scripts that need to authenticate against
the chat API without going through the Google OAuth flow.

Usage (from backend/ cwd inside the api container):

    python scripts/mint_eval_token.py

Prints the raw access token to stdout. The DB rows touched:

- ``users`` table — UPSERT a single ``eval.bot@utehy.local`` user.
- ``auth_tokens`` table — INSERT one access + one refresh token row.

Both writes are reversible (delete the user → cascade clears tokens).
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# Project import path — backend/ as root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.api.auth.service import issue_token_pair
from src.api.auth.tokens import AuthTokenRepo
from src.config.settings import settings
from src.hybridrag.chat.user import UserRepo


EMAIL = "eval.bot@utehy.local"
USERNAME = "Eval Bot"
GOOGLE_ID = "eval-bot-google-id"


async def _mint() -> str:
    user_repo = UserRepo(settings.DATABASE_URL)
    user = user_repo.upsert_google_user(
        google_id=GOOGLE_ID, email=EMAIL, username=USERNAME
    )
    token_repo = AuthTokenRepo(settings.DATABASE_URL)
    pair = await issue_token_pair(token_repo=token_repo, user=user)
    return pair.access_token


def main() -> int:
    token = asyncio.run(_mint())
    print(token)
    return 0


if __name__ == "__main__":
    sys.exit(main())
