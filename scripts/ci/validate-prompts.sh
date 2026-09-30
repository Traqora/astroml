#!/bin/bash
set -euo pipefail

echo "Validating LLM prompt templates..."

python3 -c "
from astroml.llm.prompts import PromptRegistry
registry = PromptRegistry()
templates = registry.list_templates()
print(f'Found {len(templates)} prompt templates')

errors = []
for name in templates:
    try:
        rendered = registry.render(name, {'test': 'value'})
        if not rendered or len(rendered) == 0:
            errors.append(f'{name}: rendered empty string')
    except Exception as e:
        errors.append(f'{name}: {e}')

if errors:
    print('Prompt validation FAILED:')
    for e in errors:
        print(f'  - {e}')
    exit(1)
else:
    print('All prompt templates validated successfully.')
"

echo "Prompt validation complete."
