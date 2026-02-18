import asyncio
from src.hybridrag.router import SemanticRouter
from src.hybridrag.router.keywords import KeywordRouter
from src.hybridrag.router.samples import ROUTES
from src.config.settings import settings

# router = SemanticRouter(
#     routes=ROUTES,
#     embeddings_dir=settings.ROUTER_EMBEDDINGS_DIR
# )
# score, route_name = asyncio.run(router.guide("Thông tin điểm chuẩn"))
# print(f"Score: {score:.4f}, Route: {route_name}")

router = KeywordRouter(routes=ROUTES)
score, route_name = router.guide("Thông tin điểm chuẩn???")
print(f"Keyword Router -> Score: {score}, Route: {route_name}")