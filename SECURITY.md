# Security Policy

## Supported version

Security fixes are applied to the latest commit on the default branch. Older commits and locally modified deployments are not supported separately.

## Reporting a vulnerability

Do not disclose suspected vulnerabilities in a public issue. Use GitHub's **Security → Report a vulnerability** workflow for this repository and include:

- the affected endpoint, component, or dependency;
- reproduction steps and required configuration;
- expected and observed behavior;
- potential confidentiality, integrity, or availability impact.

Do not include uploaded documents, API keys, model credentials, or other private data in a report. Allow time for validation and a coordinated fix before public disclosure.

## Deployment boundary

The default Docker Compose configuration is a single-user, localhost deployment. Any network-accessible deployment must configure `RAG_API_KEY`, TLS at a trusted reverse proxy, request/body limits, and host-level storage protection.
