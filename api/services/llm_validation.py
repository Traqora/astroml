from typing import Any, Dict

class ResponseValidator:
    def __init__(self):
        self.banned_words = {"toxic", "hate", "abuse"} # Mock list for toxicity filtering

    def _is_toxic(self, response: str) -> bool:
        """Toxicity filtering."""
        words = response.lower().split()
        return any(b in words for b in self.banned_words)

    def _has_hallucinations(self, response: str, context: str) -> bool:
        """
        Hallucination detection.
        Mock implementation: If the response mentions 'blockchain' but context doesn't.
        """
        if "blockchain" in response.lower() and "blockchain" not in context.lower():
            return True
        return False

    def _validate_schema(self, response: Any, expected_keys: list) -> bool:
        """Response schema validation."""
        if not isinstance(response, dict):
            return False
        return all(k in response for k in expected_keys)

    def _score_confidence(self, response: str) -> float:
        """Confidence scoring. Mock implementation based on length/words."""
        if len(response.split()) > 10:
            return 0.96 # Above 95% threshold
        return 0.80

    def validate_and_guard(self, raw_response: Dict[str, Any], context: str) -> Dict[str, Any]:
        """
        Run all validations.
        """
        text = raw_response.get("text", "")
        
        if self._is_toxic(text):
            raise ValueError("Response triggered toxicity filter.")
            
        if self._has_hallucinations(text, context):
            raise ValueError("Hallucination detected, regeneration required.")
            
        if not self._validate_schema(raw_response, ["text", "source"]):
            raise ValueError("Invalid schema returned.")
            
        confidence = self._score_confidence(text)
        if confidence < 0.95:
            # Mock behavior to indicate failure to pass 95% on first attempt
            pass 
            
        raw_response["confidence"] = confidence
        return raw_response
