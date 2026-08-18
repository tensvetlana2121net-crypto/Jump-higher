# Production security checklist

This checklist reduces risk; it cannot guarantee the absence of vulnerabilities.

## Before deployment

- [ ] Revoke the Telegram token previously stored in any Word document and issue a new token.
- [ ] Store the new token only in the server secret store or protected `.env` (`chmod 600`).
- [ ] Delete insecure copies, screenshots, chat messages, and unencrypted backups of secrets.
- [ ] Generate unique `SECRET_KEY`, `API_KEY`, and PostgreSQL password values.
- [ ] Confirm `.env` is ignored: `git check-ignore .env`.
- [ ] Enable GitHub 2FA, branch protection, secret scanning, CodeQL, and Dependabot.
- [ ] Patch the host OS, Docker, and all container images.
- [ ] Allow SSH keys only; disable password and root SSH login where practical.
- [ ] Allow inbound traffic only for SSH and the TLS reverse proxy.
- [ ] Confirm ports 5432 and 6379 are not public.

## Application verification

- [ ] `docker compose config` contains no unexpected public bindings.
- [ ] `/health` works locally; protected API routes reject missing or invalid `X-API-Key`.
- [ ] Oversized, malformed, long, and non-video uploads are rejected.
- [ ] Source videos disappear after completed, rejected, and failed analysis.
- [ ] Logs contain no tokens, API keys, database URLs, videos, or personal data.
- [ ] Free-tier quota remains correct under concurrent requests.
- [ ] Backup restoration and user-data deletion are tested.

## Incident response

1. Disable the affected bot/service and preserve minimal forensic evidence.
2. Revoke exposed credentials; do not merely edit the leaked file.
3. Audit Git history, CI logs, server logs, backups, and Telegram sessions.
4. Patch the root cause, redeploy from a clean commit, and verify containment.
5. Notify affected users and authorities when legally required.
