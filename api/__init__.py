"""FastAPI application and routes"""
# Removed app import to avoid circular dependencies
# Import app directly from api.main when needed
from . import models

__all__ = ['models']
