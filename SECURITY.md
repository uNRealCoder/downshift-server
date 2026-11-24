# Security policy

## Supported versions

Only the latest 1.x minor release gets security fixes. If you use an older 1.x release, update to the latest release to get a fix.

| Version | Security fixes |
|---|---|
| Latest 1.x minor | Yes |
| Older 1.x minor releases | No |
| Before 1.0 | No |

## Report a vulnerability

Do not open a public issue, pull request or discussion for a vulnerability.

1. Go to the [Security tab](https://github.com/uNRealCoder/downshift-server/security) of the repository.
2. Click **Report a vulnerability**.
3. Write the report. Include these items:
   - The downshift version (`downshift --version`), the Python version and the operating system.
   - The command or the code that starts the server, and the options.
   - The steps that show the problem, and a small model if the problem needs one.
   - The effect: what an attacker can do.

## What happens after you report

This project has one maintainer. There is no fixed time for a fix.

1. The maintainer replies to the report within 7 days, as a best effort.
2. The maintainer examines the report and tells you if it is a vulnerability.
3. If it is a vulnerability, the maintainer makes a fix in a private fork and makes a new release.
4. The maintainer publishes a GitHub security advisory after the release. The advisory gives credit to you, if you want credit.

Do not disclose the problem in public before the advisory is published.

## Scope

These items are in scope:

- The `downshift` package on PyPI and the code in this repository.
- The HTTP server of `downshift serve`, the `check` and `export` commands, and the Python API.

These items are not vulnerabilities. Each one is a known limit, and the documentation tells you about it:

- **Code in a pickled model file.** `--unsafe-load` runs `torch.load(weights_only=False)`. This runs code from the file. Use it only with a file from a source that you trust.
- **No API key.** If `DOWNSHIFT_API_KEY` is not set, the server does not check requests. It logs a warning at start.
- **Plain HTTP.** The API key is a shared secret in a header. Without `--ssl-certfile` or a proxy that ends TLS, other persons on the network can read it.
- **No rate limiting.** Downshift has no rate limiting and no limit for each client. Put a reverse proxy in front of it.
- **Open probes.** `GET /health` and `GET /ready` do not need the API key. This is by design, for Kubernetes.
- **Result cache timing.** A cache hit is faster than a miss. With more than one tenant, one client can learn if another client sent a row. Set `--result-cache-rows 0` to turn the cache off.
- **Forwarded headers.** If you set `--forwarded-allow-ips *` and other hosts can reach the port, they can set `X-Forwarded-For` and `X-Forwarded-Proto`.
- **Problems in a dependency.** Report a problem in torch, ONNX Runtime, FastAPI, uvicorn or another dependency to that project. If downshift uses the dependency in a way that is not safe, report it here.

Refer to [Production](docs/production.md) for the safe settings of a deployment.
