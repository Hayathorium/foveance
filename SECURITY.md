# Security policy

## Scope
Foveance is a context-allocation library and an OpenAI-compatible reverse proxy. The most
security-relevant component is the proxy (`foveance.proxy`), which forwards requests to an
upstream model endpoint. The core library performs no network I/O.

## Reporting a vulnerability
Please report suspected vulnerabilities privately by opening a
[GitHub security advisory](https://github.com/aimaghsoodi/foveance/security/advisories/new) rather
than a public issue. Include a description, reproduction steps, and impact. We aim to
acknowledge within 5 business days.

## Hardening notes for deployers
- The proxy does not add authentication; place it behind your own auth/gateway and do not
  expose `/admin/stats` publicly.
- Treat upstream URLs and API keys as secrets; pass them via environment, not source.
- The store retains full item text in memory per conversation; size limits and eviction are
  the deployer's responsibility for long-lived processes.

## Supported versions
The latest minor release receives security fixes. Pre-1.0, only `main` and the latest tag are
supported.
