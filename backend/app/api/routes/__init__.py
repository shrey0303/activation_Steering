"""
API route composition — assembles all sub-routers into a single router.

main.py imports `from app.api.routes import router` which resolves
here, composing all domain-specific routers into one.
"""

from fastapi import APIRouter

from app.api.routes.health import router as health_router
from app.api.routes.models import router as models_router
from app.api.routes.generation import router as generation_router
from app.api.routes.analysis import router as analysis_router
from app.api.routes.concepts import router as concepts_router
from app.api.routes.patches import router as patches_router
from app.api.routes.features import router as features_router
from app.api.routes.remote import router as remote_router

router = APIRouter()

router.include_router(health_router)
router.include_router(models_router)
router.include_router(generation_router)
router.include_router(analysis_router)
router.include_router(concepts_router)
router.include_router(patches_router)
router.include_router(features_router)
router.include_router(remote_router)
