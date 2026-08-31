# Secrets Scanning Guide

This document describes the secrets scanning implementation in AstroML and how to handle detected secrets and false positives.

## Overview

AstroML uses `detect-secrets` to automatically scan for leaked credentials in the codebase. This prevents accidental commits of API keys, passwords, tokens, and other sensitive information.

## Implementation

### Pre-Commit Hook

The `detect-secrets` hook is configured in `.pre-commit-config.yaml`:

```yaml
- repo: https://github.com/Yelp/detect-secrets
  rev: v1.4.0
  hooks:
    - id: detect-secrets
      args: [--baseline, .secrets.baseline]
      exclude: package.lock.json|go.sum|pnpm-lock.yaml|yarn.lock|poetry.lock
```

### Baseline File

The `.secrets.baseline` file stores known secrets that have been reviewed and whitelisted. This allows the scanner to ignore false positives while catching new secrets.

### CI Integration

Secrets scanning runs in CI via two workflows:

1. **Dedicated secrets scan** (.github/workflows/secrets-scan.yml): runs detect-secrets on every push and pull request. This workflow fails the build if any new secret not in the baseline is detected.
2. **Pre-commit workflow** (.github/workflows/pre-commit.yml): runs the full pre-commit hook suite, including the detect-secrets hook.

Both workflows will fail if new secrets are detected that aren't in the baseline.

## Usage

### Manual Scanning

Run a manual secrets scan:

```bash
make secrets-scan
```

Or directly:

```bash
detect-secrets scan --baseline .secrets.baseline --all-files
```

### Pre-Commit Hook

The pre-commit hook runs automatically before each commit:

```bash
git commit -m "your message"
# detect-secrets hook runs automatically
```

## Handling False Positives

### What is a False Positive?

A false positive occurs when `detect-secrets` identifies something as a secret that isn't actually sensitive. Common examples:

- Test API keys in fixtures
- Example passwords in documentation
- Random strings that happen to match secret patterns
- Hash values that look like keys

### Adding to Baseline

If a detected secret is a false positive:

1. **Review the detection**: Verify it's not actually sensitive
2. **Add to baseline**: Run the following to update the baseline:
   ```bash
   detect-secrets scan --baseline .secrets.baseline
   ```
3. **Commit the baseline**: Commit the updated `.secrets.baseline` file
4. **Document**: Add a comment in the code explaining why it's whitelisted

### Example: Whitelisting a Test Key

```python
# test_api.py
# This is a test key for unit tests only - not a real credential
# Whitelisted in .secrets.baseline
TEST_API_KEY = "sk_test_1234567890abcdef"
```

### Removing from Baseline

If a previously whitelisted secret should no longer be ignored:

1. **Edit the baseline**: Remove the entry from `.secrets.baseline`
2. **Re-scan**: Run `detect-secrets scan --baseline .secrets.baseline`
3. **Commit**: Commit the updated baseline

## Secret Types Detected

`detect-secrets` scans for various secret types:

- AWS Keys
- Azure Keys
- GitHub Tokens
- Private Keys (SSH, SSL)
- API Keys (Stripe, SendGrid, Twilio, etc.)
- JWT Tokens
- Basic Auth credentials
- High-entropy strings (likely to be secrets)

## Best Practices

### Preventing Secret Leaks

1. **Use environment variables**: Never hardcode secrets in code
2. **Use .env files**: Add `.env` to `.gitignore`
3. **Use secret management**: Use tools like HashiCorp Vault, AWS Secrets Manager
4. **Review commits**: Check diffs before pushing
5. **Enable pre-commit**: Ensure pre-commit hooks are installed

### Example: Using Environment Variables

**Bad:**
```python
api_key = "sk_live_1234567890abcdef"  # DON'T DO THIS
```

**Good:**
```python
import os
api_key = os.environ.get("API_KEY")
```

### Example: Using .env Files

```bash
# .env (gitignored)
API_KEY=sk_live_1234567890abcdef
DATABASE_URL=postgresql://user:pass@localhost/db
```

```python
# config.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    api_key: str
    database_url: str

    class Config:
        env_file = ".env"
```

## Secrets Cleanup Guide

If you accidentally commit a secret, follow these steps to remove it:

### Step 1: Revoke the Secret

**Immediately revoke the compromised secret:**
- Rotate API keys in the provider's dashboard
- Change passwords
- Revoke tokens
- Update any services using the secret

### Step 2: Remove from Current Branch

If the secret is only in your current branch:

```bash
# Remove the secret from the file
git checkout -- path/to/file.py
# Or edit the file to remove the secret
git add path/to/file.py
git commit --amend --no-edit
```

### Step 3: Remove from Git History

If the secret has been pushed:

**Option A: BFG Repo-Cleaner (Recommended)**
```bash
# Install BFG
brew install bfg  # macOS
# Or download from https://rtyley.github.io/bfg-repo-cleaner/

# Clean the repository
bfg --replace-text passwords.txt  # Create passwords.txt with the secret
bfg --delete-files file_with_secret.py

# Cleanup
git reflog expire --expire=now --all
git gc --prune=now --aggressive
```

**Option B: git filter-branch**
```bash
git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch path/to/file.py" \
  --prune-empty --tag-name-filter cat -- --all

# Force push
git push origin --force --all
git push origin --force --tags
```

**Option C: git-filter-repo (Modern alternative)**
```bash
pip install git-filter-repo
git filter-repo --invert-paths --path path/to/file.py
git push origin --force
```

### Step 4: Update Baseline

If the secret was added to the baseline:

1. Remove it from `.secrets.baseline`
2. Re-scan: `detect-secrets scan --baseline .secrets.baseline`
3. Commit the updated baseline

### Step 5: Notify Team

- Notify your team about the breach
- Document what happened
- Update security procedures if needed

## CI Failures

### When CI Fails on Secrets

If the CI job fails due to detected secrets:

1. **Check the logs**: Review which file/line triggered the detection
2. **Verify it's a secret**: Ensure it's actually sensitive
3. **Remove or whitelist**: Either remove the secret or add to baseline
4. **Update baseline**: If whitelisting, update `.secrets.baseline`
5. **Push fix**: Commit and push the changes

### Example CI Failure

```
ERROR: detect-secrets detected 1 new secret:
  File: api/config.py
  Line: 15
  Type: AWS Key
  Secret: AKIAIOSFODNN7EXAMPLE
```

**Resolution:**
```bash
# Remove the secret from code
# Add to environment variable instead
git add api/config.py
git commit -m "Remove hardcoded AWS key"
git push
```

## Configuration

### Excluding Files

Files can be excluded in `.pre-commit-config.yaml`:

```yaml
exclude: package.lock.json|go.sum|pnpm-lock.yaml|yarn.lock|poetry.lock
```

### Custom Plugins

Add custom detection rules in `.secrets.baseline`:

```json
{
  "plugins_used": [
    {
      "name": "CustomKeywordDetector",
      "path": "path/to/custom_detector.py"
    }
  ]
}
```

## Testing the Scanner

### Test with Fake Secret

To test the scanner:

1. Create a test file with a fake secret:
   ```python
   # test_secret.py
   fake_key = "sk_test_1234567890abcdef"
   ```

2. Run the scanner:
   ```bash
   detect-secrets scan test_secret.py
   ```

3. Verify it detects the secret

4. Clean up:
   ```bash
   rm test_secret.py
   ```

## Troubleshooting

### Scanner Not Running

If pre-commit hook doesn't run:

```bash
# Install pre-commit hooks
pre-commit install

# Verify installation
pre-commit run --all-files
```

### Baseline Out of Sync

If baseline is out of sync:

```bash
# Re-scan and update baseline
detect-secrets scan --baseline .secrets.baseline --all-files
```

### Too Many False Positives

If too many false positives:

1. Review the baseline and remove unnecessary entries
2. Consider adjusting detection thresholds
3. Exclude specific file patterns in `.pre-commit-config.yaml`

## Resources

- [detect-secrets Documentation](https://github.com/Yelp/detect-secrets)
- [OWASP Secret Scanning](https://owasp.org/www-community/Secrets_Management_Cheat_Sheet)
- [GitHub Secret Scanning](https://docs.github.com/en/code-security/secret-scanning)
- [BFG Repo-Cleaner](https://rtyley.github.io/bfg-repo-cleaner/)

## Contact

For questions about secrets scanning:
- Open an issue with the `security` label
- Contact the maintainers directly for urgent security issues
