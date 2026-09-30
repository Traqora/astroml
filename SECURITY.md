# Security Policy

## Supported Versions

AstroML provides security fixes for the current development branch and the latest published release line. Older release lines are not guaranteed to receive security patches unless maintainers explicitly announce otherwise.

| Support level | Status |
| --- | --- |
| Main branch | Supported |
| Latest tagged release | Supported |
| Older releases | Best effort only |

## Reporting a Vulnerability

Please report security issues privately rather than through public issues or pull requests.

Preferred options:
- Use GitHub Private Vulnerability Reporting if it is available for this repository.
- Otherwise, email the maintainers at security@traqora.com with a short summary and reproduction details.

Include the following in your report:
- Affected component or module
- Steps to reproduce the issue
- Expected vs. actual behavior
- Any proof-of-concept or relevant logs
- Your preferred contact method

Do not disclose the issue publicly until the maintainers confirm that a fix has been released or the report is otherwise safe to discuss.

## Response Expectations

- Acknowledge receipt within 5 business days.
- Triage and severity assessment should happen as quickly as possible, with critical issues receiving the highest priority.
- A mitigation plan or patch timeline will be shared once the issue is confirmed.
- Public disclosure will be coordinated after a fix is available and validated.

## Patch Release Cadence

Security fixes are handled as patch releases when practical. High-impact issues may trigger an out-of-band release. Security updates will be noted in release notes and repository advisories when applicable.

## Security Audit Checklist

Use this checklist for changes that affect authentication, storage, network exposure, or data handling:

- [ ] Secrets and API keys are stored in environment variables or a secret manager.
- [ ] Database credentials are not committed to source control.
- [ ] Network access is limited to required hosts and ports.
- [ ] New input handling validates and rejects unexpected values.
- [ ] Logging does not expose tokens, passwords, or private keys.
- [ ] Dependency changes are reviewed for known CVEs or vulnerable transitive packages.

## Known Security Considerations

### API key and secret storage

Store API credentials, database URLs, and other secrets in environment variables or another managed secret store. Avoid hard-coding secrets in configuration files, notebooks, or example scripts.

### Database access controls

Use least-privilege database accounts for application and tooling access. Restrict network access to the database host, prefer separate read/write roles where practical, and rotate credentials if they are ever exposed.

### Stellar key management

Treat private keys as sensitive secrets. Keep them out of source control, avoid committing them to examples or logs, and rotate them if an exposure is suspected.

## OWASP References

For general secure-development guidance, review the OWASP Top 10 and OWASP Application Security Verification Standard (ASVS).

## CVE Handling Process

1. Confirm the issue and determine whether it is a vulnerability, a misconfiguration, or a non-security bug.
2. Assess severity, affected versions, and the likelihood of exploitation.
3. Prepare and validate a fix in a private or restricted branch where appropriate.
4. Coordinate release timing, advisory text, and public disclosure with the maintainers.
5. Update documentation and release notes once the patch is available.
