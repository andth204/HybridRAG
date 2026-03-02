import sys
from typing import Optional
import psycopg2
from psycopg2.extras import RealDictCursor
from src.config.settings import settings
from src.hybridrag.chat.message import ChatMessageRepo
from src.hybridrag.chat.session import ChatSessionRepo


def get_user(*, email: Optional[str] = None, google_id: Optional[str] = None) -> Optional[dict]:
    if not email and not google_id:
        return None

    if email:
        sql = """
        SELECT id, google_id, email, username, created_at, updated_at
        FROM users
        WHERE email = %s
        """
        params = (email,)
    else:
        sql = """
        SELECT id, google_id, email, username, created_at, updated_at
        FROM users
        WHERE google_id = %s
        """
        params = (google_id,)

    with psycopg2.connect(settings.DATABASE_URL) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, params)
            return cur.fetchone()


def print_user_history(user: dict, session_limit: int = 20, msg_limit: int = 200) -> None:
    user_id = str(user["id"])
    session_repo = ChatSessionRepo(settings.DATABASE_URL)
    message_repo = ChatMessageRepo(settings.DATABASE_URL)
    sessions = session_repo.list_by_user(user_id=user_id, limit=session_limit)

    print("\n=== USER ===")
    print(f"id       : {user_id}")
    print(f"email    : {user.get('email')}")
    print(f"google_id: {user.get('google_id')}")
    print(f"username : {user.get('username')}")
    print(f"sessions : {len(sessions)}")

    if not sessions:
        print("\nNo chat session found for this user.")
        return

    total_messages = 0
    for idx, session in enumerate(sessions, start=1):
        messages = message_repo.load_history(session.id, limit=msg_limit)
        total_messages += len(messages)

        print(f"\n--- SESSION {idx} ---")
        print(f"session_id : {session.id}")
        print(f"title      : {session.title}")
        print(f"created_at : {session.created_at}")
        print(f"updated_at : {session.updated_at}")
        print(f"messages   : {len(messages)}")

        for i, msg in enumerate(messages, start=1):
            content_preview = msg.content.replace("\n", " ").strip()
            if len(content_preview) > 180:
                content_preview = content_preview[:180] + "..."
            print(f"  [{i}] {msg.role}: {content_preview}")
    print(f"\nTotal messages across sessions: {total_messages}")


def parse_args() -> tuple[Optional[str], Optional[str], int, int]:
    args = sys.argv[1:]
    email = None
    google_id = None
    session_limit = 20
    msg_limit = 200
    i = 0
    while i < len(args):
        arg = args[i]
        if arg == "--email" and i + 1 < len(args):
            email = args[i + 1]
            i += 2
            continue
        if arg == "--google-id" and i + 1 < len(args):
            google_id = args[i + 1]
            i += 2
            continue
        if arg == "--session-limit" and i + 1 < len(args):
            session_limit = int(args[i + 1])
            i += 2
            continue
        if arg == "--msg-limit" and i + 1 < len(args):
            msg_limit = int(args[i + 1])
            i += 2
            continue
        i += 1

    if not email and not google_id:
        email = "fake.user@utehy.local"
    return email, google_id, session_limit, msg_limit


if __name__ == "__main__":
    user_email, user_google_id, session_limit, msg_limit = parse_args()
    user = get_user(email=user_email, google_id=user_google_id)

    if not user:
        print("User not found.")
        print("Try:")
        print("  --email fake.user@utehy.local")
        print("  --google-id fake-google-id-001")
        raise SystemExit(1)

    print_user_history(user, session_limit=session_limit, msg_limit=msg_limit)