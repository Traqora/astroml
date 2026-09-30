#!/bin/bash
set -euo pipefail

echo "Running LLM safety scan..."

python3 -c "
from astroml.llm.safety import SafetyGuard, StrictnessLevel
print('Safety guard initialized successfully')

from astroml.llm.safety.filters import InputFilter, OutputFilter
print('Input/Output filters loaded')

from astroml.llm.safety.blocklist import BlocklistManager
print('Blocklist manager loaded')

from astroml.llm.safety.classifier import ContentClassifier, ContentCategory
print('Content classifier loaded')

from astroml.llm.safety.prompts import SAFETY_SYSTEM_PROMPT, get_safety_prompt
print('Safety prompts loaded')
assert len(SAFETY_SYSTEM_PROMPT) > 0, 'SAFETY_SYSTEM_PROMPT should not be empty'
print(f'Safety system prompt: {len(SAFETY_SYSTEM_PROMPT)} chars')
"

echo "Safety scan completed — all modules loaded successfully."
