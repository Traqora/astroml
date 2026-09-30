"""FastAPI application and REST API layer for AstroML.

This module provides the web API interface including:
- REST endpoints for model inference and data queries
- GraphQL API for complex queries
- Authentication and authorization middleware
- Rate limiting and security features
- Health check and monitoring endpoints

Key components:
- app.py: Main FastAPI application
- routers/: Endpoint modules for different resources
- graphql/: GraphQL schema and resolvers
- auth/: Authentication and authorization
- middleware/: Request/response middleware

Dependencies:
- fastapi: Web framework
- pydantic: Data validation
- starlette: ASGI toolkit
"""
