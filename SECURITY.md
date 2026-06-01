# Security Policy

## Supported Versions

SilexCode is currently in early development. Security fixes are applied to the
`main` branch only. Tagged releases will be supported once a stable line is
declared.

| Version | Supported |
| ------- | --------- |
| main    | yes       |
| < 0.1   | no        |

## Reporting a Vulnerability

Please **do not** open a public GitHub issue for security-sensitive reports.

Instead, use [GitHub's private vulnerability reporting](https://github.com/CrazyDeagle/project/security/advisories/new)
to file a confidential advisory. You can expect:

- An acknowledgement within **5 business days**.
- A triage decision and severity assessment within **10 business days**.
- A coordinated disclosure timeline once a fix is available.

When reporting, please include:

- A clear description of the issue and its impact.
- A minimal reproduction (code, command line, or trace).
- The commit SHA or version you observed it on.
- Any suggested mitigation or patch, if you have one.

## Scope

In scope:

- The Python package `silexcode` and its CUDA extension.
- Build scripts (`setup.py`, `pyproject.toml`) and shipped shell scripts.
- CI workflows under `.github/workflows/`.

Out of scope:

- Issues that require physical access to the training host.
- Denial of service through resource exhaustion on the training process
  itself (this is an intrinsic property of training workloads).
- Third-party dependencies — please report those upstream.
