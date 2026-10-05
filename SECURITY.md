# Security

Please report vulnerabilities privately to <contact@inco.ai> rather than in a
public issue. Include the RichEngine version (`richengine --version`), macOS version,
and steps to reproduce. We will acknowledge your report and keep you informed
until it is resolved.

RichEngine serves on `127.0.0.1`, and authentication is off by default. Set
`RICHENGINE_API_KEY` before exposing the server beyond the local machine. Exposing
it without a key or a proxy is outside the threat model.
