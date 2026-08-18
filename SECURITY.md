# Security Policy

## Supported versions

Only the current `main` branch is supported during the MVP stage.

## Reporting a vulnerability

Do not open a public issue containing credentials, personal data, videos, or an exploit.
Use GitHub private vulnerability reporting for this repository. Include the affected commit,
reproduction steps, impact, and any suggested mitigation.

## Secret handling

- Never commit `.env`, bot tokens, database passwords, API keys, videos, or database dumps.
- Production credentials must be unique and generated with a cryptographic random generator.
- Rotate a credential immediately if it appears in a document, chat, screenshot, terminal log,
  CI output, repository, or backup outside the approved secret store.
- The Telegram token must be revoked and reissued through BotFather after any suspected exposure.

## Data handling

Uploaded videos are untrusted personal data. Production is configured to delete source videos
after analysis. PostgreSQL backups and exported results require encryption, access control, a
retention policy, and tested deletion procedures.

## Operational baseline

- Run behind a patched host firewall; do not expose PostgreSQL or Redis.
- Keep the API bound to localhost unless it is behind an authenticated TLS reverse proxy.
- Enable two-factor authentication and branch protection on GitHub and Telegram owner accounts.
- Review Dependabot and CodeQL alerts before deployment.
- Back up PostgreSQL encrypted at rest and test restoration regularly.
