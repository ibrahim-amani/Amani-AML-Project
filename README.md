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
  Runs **Entity Resolution** to generate final merged (“Golden”) records (Full + Delta).



---

##  Requirements

* Python **3.12.12**
* `uv` 

---

##  Install & Setup

From the project root:

```bash
uv sync
```

##  Install & Update `PlayWright`

```bash
playwright install
```

### Run without `uv run` (recommended)

Activate the environment:

```bash
.venv\scripts\activate
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

* Project path: `settings.DATA_LAKE_DIR`
* Underlying library path: `amani_aml.core.config.settings.DATA_LAKE_DIR`

At the end of each run, the CLI prints:

* Total execution time
* Output location (Data Lake directory)

---

## Project Structure

```
Amani-AML-Project
├── src/
│   ├── amani_aml/                  # Main application package
│   │   ├── api/                    # API layer (FastAPI)
│   │   │   ├── __init__.py
│   │   │   └── app.py              # API application entrypoint
│   │   │
│   │   ├── core/                   # Core application concerns
│   │   │   └── config.py           # Configuration, settings, logging
│   │   │
│   │   ├── pipelines/              # Orchestrated processing pipelines
│   │   │   ├── __init__.py
│   │   │   └── resolver.py         # Entity resolution pipeline
│   │   │
│   │   ├── services/               # Business logic / domain services
│   │   │   ├── __init__.py
│   │   │   ├── media_svc.py        # Adverse media processing
│   │   │   ├── resolver_svc.py     # Entity resolution service
│   │   │   └── sanctions_svc.py    # Sanctions & PEP screening
│   │   │
│   │   ├── utils/                  # Shared utility functions
│   │   │   ├── __init__.py
│   │   │   └── nat_to_iso.py       # Nationality → ISO mapping
│   │   │
│   │   ├── cli.py                  # Command-line interface (`aml`)
│   │   └── __init__.py
│   │
│   ├── __main__.py                 # Python module entrypoint
│   └── __init__.py
│
├── Dockerfile                      # Application container image
├── docker-compose.yml              # Local multi-service orchestration
├── pyproject.toml                  # Project metadata & dependencies
├── uv.lock                         # Locked dependency versions
├── README.md                       # Project documentation
├── .python-version                 # Python version pin
├── .dockerignore
└── .gitignore
```

---

##  Configuration

Edit `src/amani_aml/config.py` to control:

* `DATA_LAKE_DIR`
* `SANCTIONS_CONCURRENCY
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
aml api
```

Server runs at:

```
http://127.0.0.1:8000
```

### Available Endpoints

| Endpoint                    | Description                                                            |
| --------------------------- | ---------------------------------------------------------------------- |
| /                         | Health check                                                           |
| /amani/meta               | **FULL export manifest** (initial load metadata for Golden export)     |
| /amani/file               | **FULL export file download** (Golden_Export.jsonl.gz)               |
| /amani/delta/meta         | **DELTA export manifest** (new + updated records metadata)             |
| /amani/delta/file         | **DELTA export file download** (Golden_Export.delta.jsonl.gz)        |

---

> These endpoints are used directly by the **AmaniAI provider**.

---

