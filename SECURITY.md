# Security Policy

## Supported Versions

| Version | Supported          |
|---------|--------------------|
| 2.0.x   | Yes                |
| < 2.0   | No                 |

## Reporting a Vulnerability

If you discover a security vulnerability in GraphRAG SDK, please report it responsibly.

**Do not open a public GitHub issue for security vulnerabilities.**

Instead, please use one of the following methods:

1. **GitHub Security Advisories**: Use the [Report a vulnerability](https://github.com/FalkorDB/GraphRAG-SDK/security/advisories/new) feature on GitHub (preferred).
2. **Email**: Send details to security@falkordb.com.

### What to Include

- A description of the vulnerability
- Steps to reproduce the issue
- The potential impact
- Any suggested fixes (optional)

### What to Expect

- **Acknowledgment**: We will acknowledge your report within 48 hours.
- **Assessment**: We will assess the severity and impact within 5 business days.
- **Fix**: Critical vulnerabilities will be patched as quickly as possible.
- **Disclosure**: We will coordinate disclosure timing with you.

## Scope

This security policy covers the GraphRAG SDK Python package (`graphrag-sdk`). It does **not** cover:

- FalkorDB server (report to [FalkorDB](https://github.com/FalkorDB/FalkorDB))
- Third-party LLM providers (OpenAI, Anthropic, Cohere, etc.)
- Dependencies (report to the respective upstream projects)

## Security Best Practices

When using GraphRAG SDK:

- **Never commit API keys**: Use environment variables or a `.env` file (see `.env.example`).
- **Use network isolation**: Run FalkorDB behind a firewall or private network in production.
- **Enable authentication**: Configure FalkorDB with username/password via `ConnectionConfig`.
- **Review Cypher queries**: The SDK sanitizes generated Cypher queries, but review any custom queries you add.
