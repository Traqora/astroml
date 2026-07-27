import time
from typing import List, Optional
import difflib
from api.schemas import SuggestionItem, SuggestionResponse

class AutocompleteService:
    def __init__(self):
        # Mock database of popular queries
        self.popular_queries = {
            "latest transactions": 1500,
            "show me recent transactions": 1200,
            "high value accounts": 900,
            "anomalous transactions": 800,
            "whale accounts": 700,
            "fraudulent transactions": 600,
            "transaction volume over time": 500,
            "recent blocks": 450,
            "active addresses": 400,
            "gas fees history": 300,
            "smart contract deployments": 250
        }

    def suggest(self, partial_query: str, max_results: int = 5) -> SuggestionResponse:
        """
        Returns suggestions for a partial query.
        Includes typo correction if no direct matches are found.
        """
        partial_lower = partial_query.lower()
        
        # 1. Exact prefix matching
        matches = [
            (q, pop) for q, pop in self.popular_queries.items() 
            if q.startswith(partial_lower)
        ]
        
        # 2. Substring matching if few prefix matches
        if len(matches) < max_results:
            substring_matches = [
                (q, pop) for q, pop in self.popular_queries.items() 
                if partial_lower in q and not q.startswith(partial_lower)
            ]
            matches.extend(substring_matches)
        
        is_correction = False
        corrected_query = None
        
        # 3. Typo correction if still no matches
        if not matches and len(partial_lower) > 3:
            # Find the closest query by difflib
            closest_keys = difflib.get_close_matches(partial_lower, self.popular_queries.keys(), n=1, cutoff=0.6)
            if closest_keys:
                closest_query = closest_keys[0]
                matches = [(closest_query, self.popular_queries[closest_query])]
                is_correction = True
                corrected_query = closest_query

        # 4. Rank by popularity
        matches.sort(key=lambda x: x[1], reverse=True)
        top_matches = matches[:max_results]
        
        suggestions = [
            SuggestionItem(
                query=q, 
                popularity=pop, 
                is_correction=is_correction
            ) 
            for q, pop in top_matches
        ]
        
        return SuggestionResponse(
            suggestions=suggestions,
            corrected_query=corrected_query
        )
