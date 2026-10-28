# Public Streamlit demo

The repository entry point is `app.py`. It imports the packaged platform, so
deploy the repository with its current source rather than copying that file alone.
`requirements.txt` installs the project with its `demo` dependencies. No API keys
or application secrets are required.

## Streamlit Community Cloud

1. Sign in at <https://share.streamlit.io/> and connect the GitHub account that
   owns this repository.
2. Create an app from an existing repository with these settings:

   | Setting | Value |
   |---|---|
   | Repository | `Vaishnavi-Raghupathi/Adaptive-ODE-Solver-with-Diagnostic-Driven-Model-Switching` |
   | Branch | `main` |
   | Entrypoint | `app.py` |
   | Python | `3.13` (the package also supports 3.11 and 3.12) |

3. Choose an available app subdomain, deploy, and inspect the build log.
4. Verify the actual returned public URL before sharing it. Open the app without
   signing in and run the example oscillator comparison. Also check independent
   trajectories and download a recommendation report.

Deployment depends on the host account being authenticated and authorized to
read the repository. Local preparation and a GitHub push do not establish that
an app is deployed. This document intentionally does not invent a public URL.

The checked-in Streamlit configuration limits each uploaded file to 10 MB and
disables Streamlit usage telemetry. Uploaded data are used in the running app's
session; the application does not deliberately save them to a database or write
them into the Git repository. This is a research demonstration, not a deployment
for confidential or regulated datasets. Hosting-provider behavior is governed by
the provider's own policies.

Cloud timing ranks reflect the shared machine at the time of each comparison.
They should not be presented as hardware-independent performance rankings.

## Local verification

```bash
python -m pip install '.[demo,dev]'
python -m pytest -q
python -m streamlit run app.py --server.address 127.0.0.1 --server.port 8512
```

The local URL is not externally shareable. A temporary tunnel would also depend
on the local computer staying online and is not a replacement for hosted deployment.

Host instructions: [deploy an app](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy).
