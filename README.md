# 🛡️ Amani AML Project

A lightweight orchestration layer that runs **Sanctions/PEP scraping**, **Adverse Media screening**, and **Entity Resolution** in a clean two-phase pipeline, exposed through a single CLI command: `aml`.

This project wraps and coordinates two engines:

* **`sanction-pep-parser`** (Sanctions/PEP sources + resolver)
* **`adverse-media-parser`** (Adverse Media sources + screening)

---

##  What it does

### Two-phase execution model

* **Phase 1 (Parallel)**
  Runs **Sanctions/PEP scrapers** and **Adverse Media screening** at the same time.
* **Phase 2 (Sequential)**
  Runs **Entity Resolution** to generate final merged (“Golden”) records.



---

## ✅ Requirements

* Python **3.12+**
* `uv` (recommended)
* Windows / Linux supported

---

##  Install & Setup

From the project root:

```bash
uv sync
```

### Run without `uv run` (recommended)

Activate the environment:

**PowerShell**

```powershell
.\.venv\Scripts\Activate.ps1
```

Then you can run:

```bash
aml --help
```

---

##  CLI Usage

### Help

```bash
aml --help
```

### Run full pipeline

```bash
aml pipeline
```

### Pipeline options

```bash
aml pipeline --parallel 5
aml pipeline --no-media
aml pipeline --no-resolve
aml pipeline --no-sanctions
```

### Run individual components

```bash
aml sanctions --parallel 4
aml media
aml resolve
```

---

## List Available Sources

The `sources` command prints all supported data sources across both engines.

```bash
aml sources
```

### Options

| What you want        | Command                   |
| -------------------- | ------------------------- |
| All sources          | `aml sources`             |
| Sanctions / PEP only | `aml sources --sanctions` |
| Adverse Media only   | `aml sources --media`     |
| All (explicit)       | `aml sources --all`       |

To see command-specific help:

```bash
aml sources --help
```

---

##  Output / Data Lake

The orchestration sets the data lake path and prints it during execution:

* Project path: `settings.DATA_PATH`
* Underlying library path: `sanction_parser.core.config.settings.DATA_LAKE_DIR`

At the end of each run, the CLI prints:

* Total execution time
* Output location (Data Lake directory)

---

## Project Structure

```
test_project/
├─ dists/                           # Local wheels
│  ├─ adverse_media_parser-*.whl
│  └─ sanction_pep_parser-*.whl
├─ src/
│  └─ amani_aml/
│     ├─ __init__.py
│     ├─ cli.py                     # `aml` CLI
│     ├─ config.py                  # settings + logging
│     └─ services/
│        ├─ sanctions_svc.py        # runs ScraperRegistry sources
│        ├─ media_svc.py            # runs adverse media engine (thread-offloaded)
│        └─ resolver_svc.py         # runs EntityResolver (thread-offloaded)
├─ pyproject.toml
└─ uv.lock
```

---

##  Configuration

Edit `src/amani_aml/config.py` to control:

* `DATA_PATH`
* `SANCTIONS_CONCURRENCY`
* Logging level and format

The CLI automatically calls:

* `set_data_lake_path(settings.DATA_PATH)` to keep the library output aligned with your project data path.

---

##  Quick Smoke Test

After installation and activation:

```bash
aml --help
aml sources
aml pipeline --no-media --no-resolve
```




## Running the FastAPI Export Service

The API exposes Golden Records and provider endpoints.

### Start the API server

```bash
uvicorn sanction_parser.api.app:app
```

Server runs at:

```
http://127.0.0.1:8000
```

### Available endpoints

| Endpoint                    | Description                     |
| --------------------------- | ------------------------------- |
| `/docs`                     | Swagger UI                      |
| `/openapi.json`             | OpenAPI schema                  |
| `/health`                   | Health check                    |
| `/exports/meta`             | Golden export manifest          |
| `/exports/golden`           | Download Golden_Export.jsonl.gz |
| `/amani/meta`               | AmaniAI provider manifest       |
| `/amani/file`               | AmaniAI export file             |
| `/amani/update/{timestamp}` | AmaniAI delta updates           |

> These endpoints are used directly by the **AmaniAI provider**.

---

---

## 🧾 Packaging / Entry Point

This project is packaged to enable the `aml` command via:

```toml
[project.scripts]
aml = "amani_aml.cli:main"
```

The `src/` layout is enabled using:

```toml
[tool.setuptools]
package-dir = {"" = "src"}

[tool.setuptools.packages.find]
where = ["src"]
```

---
