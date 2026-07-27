"""GitHub Discussions API integration router."""
import httpx
import logging
from fastapi import APIRouter, HTTPException, Query
from typing import Optional
from datetime import datetime, timedelta
import os

router = APIRouter(prefix="/api/v1/discussions", tags=["discussions"])
logger = logging.getLogger(__name__)

GITHUB_API_BASE = "https://api.github.com"
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_OWNER = os.getenv("GITHUB_OWNER", "Traqora")
GITHUB_REPO = os.getenv("GITHUB_REPO", "astroml")
DISCUSSION_CACHE = {}
CACHE_TTL = 300  # 5 minutes


class DiscussionService:
    """Service for GitHub Discussions API integration."""

    def __init__(self):
        self.headers = {
            "Authorization": f"Bearer {GITHUB_TOKEN}" if GITHUB_TOKEN else "",
            "Accept": "application/vnd.github.v3+json",
        }

    async def fetch_discussions(
        self, category: Optional[str] = None, limit: int = 20
    ) -> list:
        """Fetch recent discussions from GitHub."""
        if not GITHUB_TOKEN:
            return []

        try:
            query = """
            query($owner:String!, $name:String!, $first:Int!) {
              repository(owner:$owner, name:$name) {
                discussions(first:$first, orderBy:{field:UPDATED_AT, direction:DESC}) {
                  edges {
                    node {
                      id
                      title
                      body
                      createdAt
                      updatedAt
                      author {
                        login
                      }
                      category {
                        name
                      }
                      comments {
                        totalCount
                      }
                    }
                  }
                }
              }
            }
            """

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{GITHUB_API_BASE}/graphql",
                    json={
                        "query": query,
                        "variables": {
                            "owner": GITHUB_OWNER,
                            "name": GITHUB_REPO,
                            "first": limit,
                        },
                    },
                    headers=self.headers,
                    timeout=10,
                )

            if response.status_code != 200:
                logger.error(f"GitHub API error: {response.text}")
                return []

            data = response.json()
            if "errors" in data:
                logger.error(f"GraphQL error: {data['errors']}")
                return []

            discussions = data.get("data", {}).get("repository", {}).get("discussions", {})
            return [
                {
                    "id": edge["node"]["id"],
                    "title": edge["node"]["title"],
                    "body": edge["node"]["body"][:200],  # Truncate for display
                    "createdAt": edge["node"]["createdAt"],
                    "updatedAt": edge["node"]["updatedAt"],
                    "author": edge["node"]["author"]["login"],
                    "category": edge["node"]["category"]["name"],
                    "commentCount": edge["node"]["comments"]["totalCount"],
                }
                for edge in discussions.get("edges", [])
                if not category or edge["node"]["category"]["name"] == category
            ]
        except Exception as e:
            logger.error(f"Error fetching discussions: {e}")
            return []

    async def get_discussion_categories(self) -> list:
        """Get available discussion categories."""
        if not GITHUB_TOKEN:
            return []

        try:
            query = """
            query($owner:String!, $name:String!) {
              repository(owner:$owner, name:$name) {
                discussionCategories(first:10) {
                  edges {
                    node {
                      name
                      description
                    }
                  }
                }
              }
            }
            """

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{GITHUB_API_BASE}/graphql",
                    json={
                        "query": query,
                        "variables": {
                            "owner": GITHUB_OWNER,
                            "name": GITHUB_REPO,
                        },
                    },
                    headers=self.headers,
                    timeout=10,
                )

            if response.status_code != 200:
                return []

            data = response.json()
            if "errors" in data:
                return []

            categories = data.get("data", {}).get("repository", {}).get("discussionCategories", {})
            return [
                {
                    "name": edge["node"]["name"],
                    "description": edge["node"]["description"],
                }
                for edge in categories.get("edges", [])
            ]
        except Exception as e:
            logger.error(f"Error fetching categories: {e}")
            return []


discussion_service = DiscussionService()


@router.get("/recent")
async def get_recent_discussions(
    category: Optional[str] = Query(None),
    limit: int = Query(20, ge=1, le=100),
):
    """Get recent discussions from GitHub."""
    cache_key = f"discussions:{category}:{limit}"
    
    # Check cache
    if cache_key in DISCUSSION_CACHE:
        cached_data, timestamp = DISCUSSION_CACHE[cache_key]
        if datetime.now() - timestamp < timedelta(seconds=CACHE_TTL):
            return {"discussions": cached_data, "cached": True}

    discussions = await discussion_service.fetch_discussions(category, limit)
    
    # Cache result
    DISCUSSION_CACHE[cache_key] = (discussions, datetime.now())
    
    return {"discussions": discussions, "cached": False}


@router.get("/categories")
async def get_discussion_categories():
    """Get available discussion categories."""
    cache_key = "discussion_categories"
    
    # Check cache
    if cache_key in DISCUSSION_CACHE:
        cached_data, timestamp = DISCUSSION_CACHE[cache_key]
        if datetime.now() - timestamp < timedelta(seconds=CACHE_TTL):
            return {"categories": cached_data, "cached": True}

    categories = await discussion_service.get_discussion_categories()
    
    # Cache result
    DISCUSSION_CACHE[cache_key] = (categories, datetime.now())
    
    return {"categories": categories, "cached": False}


@router.post("/search")
async def search_discussions(query: str, limit: int = 20):
    """Search discussions by query (simple text search on cached data)."""
    discussions = await discussion_service.fetch_discussions(limit=100)
    
    filtered = [
        d for d in discussions
        if query.lower() in d["title"].lower() or query.lower() in d["body"].lower()
    ][:limit]
    
    return {"results": filtered, "total": len(filtered)}


@router.get("/user-reputation/{username}")
async def get_user_reputation(username: str):
    """Get user reputation based on discussion activity."""
    if not GITHUB_TOKEN:
        raise HTTPException(status_code=400, detail="GitHub token not configured")

    try:
        query = """
        query($userName:String!) {
          user(login:$userName) {
            contributionsCollection {
              totalCommitContributions
              totalIssueContributions
              totalPullRequestContributions
            }
            followers {
              totalCount
            }
            repositories(first:1) {
              totalCount
            }
          }
        }
        """

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{GITHUB_API_BASE}/graphql",
                json={
                    "query": query,
                    "variables": {"userName": username},
                },
                headers={
                    "Authorization": f"Bearer {GITHUB_TOKEN}",
                    "Accept": "application/vnd.github.v3+json",
                },
                timeout=10,
            )

        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail="Failed to fetch user data")

        data = response.json()
        if "errors" in data:
            raise HTTPException(status_code=400, detail="GitHub API error")

        user = data.get("data", {}).get("user", {})
        contributions = user.get("contributionsCollection", {})
        
        # Calculate reputation score
        reputation_score = (
            contributions.get("totalCommitContributions", 0) * 10 +
            contributions.get("totalIssueContributions", 0) * 5 +
            contributions.get("totalPullRequestContributions", 0) * 15 +
            user.get("followers", {}).get("totalCount", 0) * 2 +
            user.get("repositories", {}).get("totalCount", 0) * 3
        )
        
        return {
            "username": username,
            "reputationScore": reputation_score,
            "contributions": {
                "commits": contributions.get("totalCommitContributions", 0),
                "issues": contributions.get("totalIssueContributions", 0),
                "pullRequests": contributions.get("totalPullRequestContributions", 0),
            },
            "followers": user.get("followers", {}).get("totalCount", 0),
            "repositories": user.get("repositories", {}).get("totalCount", 0),
        }
    except Exception as e:
        logger.error(f"Error fetching user reputation: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch reputation")
