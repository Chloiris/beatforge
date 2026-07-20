# Security Policy

## Supported version

Security fixes are applied to the latest code on the default branch. Older snapshots and
unofficial builds are not maintained separately.

## Reporting a vulnerability

Please report vulnerabilities through GitHub's private vulnerability reporting page:

<https://github.com/Chloiris/beatforge/security/advisories/new>

Do not disclose an unresolved vulnerability in a public issue. Include the affected version or
commit, reproduction steps, impact, and any suggested mitigation. Please remove private audio,
lyrics, access tokens, local paths, and other personal data from the report.

## Local-only deployment warning

BeatForge Studio is a local workstation application. The API has no user authentication, access
control, or TLS termination and may process private audio and lyrics. Keep the Web and API ports
bound to the loopback interface (`127.0.0.1` or `localhost`) and do not expose them directly to a
LAN or the public internet.

Export packages may contain the original or separated audio, depending on the selected export
mode. Inspect an archive before sharing it. Model preparation commands contact their documented
upstream model registries; normal analysis is designed to use local files after preparation.
