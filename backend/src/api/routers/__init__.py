from src.api.routers.auth import router as auth_router
from src.api.routers.chat import router as chat_router
from src.api.routers.files import router as files_router
from src.api.routers.health import router as health_router
from src.api.routers.search import router as search_router
from src.api.routers.users import router as users_router

__all__ = [
    "auth_router",
    "chat_router",
    "files_router",
    "health_router",
    "search_router",
    "users_router",
]