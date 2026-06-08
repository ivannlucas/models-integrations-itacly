# Basic Python WebApp Service Template


![Python](https://img.shields.io/badge/Python-3.12-3776AB.svg?logo=python&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Enabled-2496ED.svg?logo=docker&logoColor=white)

![Version](https://img.shields.io/badge/version-0.1.39-blue.svg)

![Status](https://img.shields.io/badge/Status-Work_in_Progress-orange.svg) / ![Status](https://img.shields.io/badge/Status-Production_Ready-brightgreen.svg) / ![Status](https://img.shields.io/badge/Status-Maintenance-lightgrey.svg) / ![Status](https://img.shields.io/badge/Status-Archived-red.svg)

A minimal Flask web service template with Swagger documentation, ready for
Docker, and CI with Bitbucket Pipelines and Sonar. Includes basic logging, monitoring, and unit tests.

## Overview

- Framework: Flask with Flasgger (Swagger UI)
- Purpose: Provide a lightweight starting point for REST services, with example endpoint, logging, metrics, and CI
  hooks.
- API Docs: Swagger UI served by the app (see API section)

## Tech Stack

- Language: Python 3.12+
- Web: Flask (`src.main.app`)
- API Docs: Flasgger (`swagger.yaml`)
- WSGI: Gunicorn (production example)
- Package manager: pip (`requirements.txt`)
- Tests: pytest + unittest-style test cases
- CI/CD: Bitbucket Pipelines (`bitbucket-pipelines.yml`), Sonar (`sonar-project.properties`)

## Requirements

- Python 3.12 or newer
- pip
- Optional: Docker, `gcloud` SDK for GCP deployment

Python dependencies are pinned in `requirements.txt`.

## Install project:

1. Create your new project in repo using Bitbucket (https://bitbucket.org/dashboard/overview) user interface.

2. Clone the newly created repo to your local machine.

   ```bash
    git clone <url_new_project> PROJECT_NAME
   ```

3. Navigate to project folder

    ```bash
     cd <PROJECT_NAME>
    ```

4. Add this template project as a remote

   ```bash
    git remote add template_repository git@bitbucket.org:luce_data/luce-python-cloud-template.git
   ```

5. Fetch the template project

   ```bash
    git fetch template_repository
   ```

6. Create a new main branch from the template main

   ```bash
    git checkout --orphan master template_repository/master
   ```

7. Modify settings following Luce guidelines

    - Sonar properties: projectKey, projectName, sources
    - Setup: name, description, author email, keywords
    - Requirements: add necessary libraries (with version)
    - Bitbucket Pipeline: replace TODOs as needed
    - README: update for your case
    - Changelog: restart changelog table to version 0.0.0


8. Commit and push the template files to your new master branch

   ```bash
    git commit -m "feat: Initial project setup from Luce Python template"
   ```

   ```bash
    git push --set-upstream origin master
   ```

9. Remove the template remote

   ```bash
    git remote remove template_repository
   ```

## Setup

1) Create and activate a virtual environment

    - Windows (PowerShell):
      ```
      python -m venv .venv
      .venv\Scripts\activate
      ```
    - Linux/Mac:
      ```
      python -m venv .venv
      source .venv/bin/activate
      ```

2) Install dependencies
    ```
    pip install -r requirements.txt
    ```

## Running the application

- Local development (runs the Flask dev server with metrics setup):
  ```bash
  python main.py
  ```
  Entrypoint flow: `main.py` → `src.main.main()` → runs `src.main.app`.

- Production (Gunicorn):
  ```bash
  gunicorn -c gunicorn.conf.py src.main:app
  ```
  Note: The Flask application object is `app` inside the module `src.main`.

- Docker:
  ```bash
  docker build -t basic_webapp .
  docker run -p 8080:8080 -e APP_ENV=dev -e HOST=0.0.0.0 -e PORT=8080 basic_webapp
  ```

- Google App Engine (App Engine config present at `app.yaml`):
    - TODO: Document exact `gcloud app deploy` steps and required service account/permissions.

- Custom:
    - TODO: Explain how to configure and run the application

## Environment variables

- `APP_ENV`: Environment where the application runs (e.g., `dev`, `test`, `prod`). Used to select configuration.
- `HOST`: Host interface to bind (e.g., `0.0.0.0`).
- `PORT`: Port to bind (e.g., `8080`).
- `LOG_LEVEL`: Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`).
- `LOG_TIMEZONE`: Timezone for logs (e.g., `UTC`, `Europe/Madrid`).
- `DEEP_LOG`: Enables deep logs (`0` or `1`).

Notes:

- Config and environment resolution is handled by helper functions in `src.utils.io` (see codebase).
- Logs are written under the `logs/` directory; metrics are written via `Monitoring`/`Metric` to a CSV (e.g.,
  `metrics_example.csv`).

## API and Swagger

- Base URL example during local run: `http://<HOST>:<PORT>/`
- Swagger UI: typically at `/apidocs/` (served by Flasgger). Example: `http://localhost:8080/apidocs/`
- Swagger spec file: `swagger.yaml` at repo root (loaded by the app at startup).

Available endpoints (from current implementation):

- `GET /` → returns a JSON welcome message.

## Tests

**Use one test file for each module. It should be named test_your_module_name.**

You can run them with either pytest command (in project root) or with a configuration on Pycharm:

Run on Pycharm:

1. Run > Edit Configurations...
2. Set Target: Script path
3. Path: path/to/project/<PROJECT_NAME>/tests
4. Working directory: path/to/project/<PROJECT_NAME>
5. Then, run the configuration

Run all tests:

```bash
python -m pytest
```

Run with coverage (package is `src`):

```bash
python -m pytest --cov=src
```

Run a specific test file or test:

```bash
python -m pytest tests/unit/test_main.py
python -m pytest tests/unit/test_main.py::TestIndexEndpoint::test_index
```

Pytest configuration/fixtures live in `conftest.py`.

## Useful scripts/commands

- Run app (dev): `python main.py`
- Run with Gunicorn: `gunicorn -c gunicorn.conf.py src.main:app`
- Run tests: `python -m pytest`
- Coverage: `python -m pytest --cov=src`
- Docker build/run: see commands above
- CI: configured in `bitbucket-pipelines.yml` (executed by Bitbucket Pipelines)
- Sonar: configuration in `sonar-project.properties` (execution handled in CI)

## Project structure

Current top-level layout:

```
basic_webapp/                 # The main project root directory
├── src/                 # The main Python package containing the application's source code
│   ├── main.py               # The core application file (e.g., Flask app initialization, routes)
│   ├── templates/            # Directory for HTML templates (e.g., html)
│   └── utils/                # Utility functions and helper modules (reusable code)
├── data/                     # For static data or configuration files needed by the app
│   └── config.yaml           # Application configuration (e.g., API keys, database settings)
├── deployment/               # All scripts and configs related to deployment (CI/CD, IaC)
│   ├── data/                 # Data files specific to deployment (e.g., database seeds)
│   ├── schemas/              # Validation schemas (e.g., JSON Schema) for configs, tables or APIs
│   ├── scripts/              # Automation scripts for the deployment process
│   │   ├── artifacts/        # Storage for build artifacts (compiled files, packages)
│   │   ├── infra/            # Infrastructure as Code (IaC) scripts (e.g., Terraform, shell)
│   │   └── utils/            # Helper scripts used to doc generation and quality review
├── logs/                     # Directory for application log files
├── tests/                    # Package containing all automated tests
│   ├── unit/                 # Unit tests (testing individual functions in isolation)
│   ├── integration/          # Integration tests (testing how components work together)
│   └── acceptance/           # Acceptance/E2E tests (testing full user flows)
├── .VERSION                  # Simple text file tracking the project's current version
├── app.yaml                  # Configuration file for Google App Engine deployment
├── bitbucket-pipelines.yml   # CI/CD configuration for Bitbucket Pipelines
├── conftest.py               # Pytest configuration file for shared test fixtures
├── Dockerfile                # Instructions to build a Docker container image for the app
├── gunicorn.conf.py          # Configuration file for the Gunicorn production WSGI server
├── LICENSE                   # Copyright/License project information
├── main.py                   # Project root entrypoint (delegates to src.main for Gunicorn)
├── requirements.txt          # List of Python package dependencies (installed via pip)
├── setup.py                  # Script to package the application as an installable Python library
├── sonar-project.properties  # Configuration for SonarQube/SonarCloud static code analysis
├── swagger.yaml              # OpenAPI (Swagger) specification for API documentation
└── README.md                 # This file! Project documentation.
```

Note: Some folders (e.g., `src/core`, `src/utils`, etc.) contain additional modules such as logging, metrics
and IO utilities.

## Deployment

- Bitbucket Pipelines automates CI/CD (see `bitbucket-pipelines.yml`).
- Docker-based deployment is possible using the provided `Dockerfile`.
- TODO: Provide concrete deployment steps and which environment variables are set by the platform.

## Architecture diagram

- TODO: Link to the architecture diagram of the application
- If you have a screenshot of the app or an architecture diagram, place it using `![Alt Text](path/to/image.png)`.

## Versioning

We use the internal policy referenced at:
https://sites.google.com/luceit.es/luceit/otros/devops/devops-politica_de_versionado

## Changelog

| Version | Date (last change) |  Developer   | Changes                      |
|:-------:|:-------------------|:------------:|
:-----------------------------|
| v0.1.39 | 08/06/2026         | Alejandro Flores | fix bitbucket-pipelines.yml |
| v0.1.38 | 08/06/2026         | Alejandro Flores | update .gitignore |
| v0.1.37 | 08/06/2026         | Alejandro Flores | Typed plugin responses; fix duplicate ModelEntry train fields |
| v0.1.36 | 08/06/2026         | Pablo    | Add docstrings to ml25 module |  
| v0.1.35 | 05/06/2026         | Alejandro Flores | Enable sonar.verbose=true (temporary) to debug SonarQube quality gate failure in CI |
| v0.1.34 | 05/06/2026         | Alejandro Flores | Fix duplicate train kwargs SyntaxError in main.py, refresh CLAUDE.md, wrap long lines in train_dto |
| v0.1.33 | 05/06/2026         | Alejandro Flores | Fix missing module docstrings, misplaced docstring in stats_dto, rename pH param, remove no-else-return |
| v0.1.32 | 05/06/2026         | Alejandro Flores | Delete dead wine_sulphite plugin files to fix SonarQube broken import bugs and 0% coverage |
| v0.1.31 | 05/06/2026         | Alejandro Flores | Remove uncovered except block from train endpoint to fix SonarQube |
| v0.1.30 | 05/06/2026         | Alejandro Flores | Add _reload_classifier test to reach SonarQube coverage threshold |
| v0.1.29 | 05/06/2026         | Alejandro Flores | ModelEntry train defaults to None, fallback resolved in router_factory |
| v0.1.28 | 05/06/2026         | Alejandro Flores | Fix ModelEntry train field defaults (SonarQube lambda smell) |
| v0.1.27 | 05/06/2026         | Alejandro Flores | Fix missing train fields in ModelEntry dataclass |
| v0.1.26 | 05/06/2026         | Alejandro Flores | Fix SyntaxError in ml25 model_loader |
| v0.1.25 | 05/06/2026         | Alejandro Flores | Fix post-merge: SyntaxError, lint, StatsResponse schema, tests |
| v0.1.24 | 05/06/2026         |   Pablo      | Artifact store fix tests		 |
| v0.1.23 | 05/06/2026         |   Pablo      | Artifact store fix				 |
| v0.1.22 | 05/06/2026         |   Pablo      | Artifact store mock				 |
| v0.1.21 | 05/06/2026         |   Pablo      | Artifact store mock				 |
| v0.1.20 | 28/05/2026         |   Trinidad   | Added modelo10-lacteo logic  |
| v0.1.0  | 28/05/2026         |  Ivan Lucas  | Plugin architecture, ML-25 wine-sulphite model, S3 artifact store, MODEL env var filtering |
| v0.0.1  | 17/02/2026         | javier.perez | Luce IT repository structure |
| v0.0.0  | 15/02/2026         |  Ivan Lucas  | Base project                 |


## Authors

![Maintained by](https://img.shields.io/badge/Maintained_by-Data_Team-blueviolet.svg)

- xxxx

## License

![License](https://img.shields.io/badge/License-Proprietary-red.svg) ![Confidential](https://img.shields.io/badge/Confidential-Luce_IT_Internal-critical.svg)

Copyright © 2025 Luce Innovative Technologies

All rights reserved.

This source code and its documentation are the confidential and proprietary
information of Luce Innovative Technologies ("Confidential Information").
You shall not disclose such Confidential Information and shall use it only
in accordance with the terms of the license agreement you entered into
with Luce Innovative Technologies.

Dissemination of this information or reproduction of this material is strictly
forbidden unless prior written permission is obtained from Luce Innovative Technologies.