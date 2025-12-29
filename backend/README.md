# Backend (Django)

## Setup

1. Create venv and install deps:

```bash
python -m venv .venv
. .venv/Scripts/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. (Optional) Configure PostgreSQL via environment variables:

Set these in your shell (PowerShell example):

```powershell
$env:POSTGRES_DB = "project_db"
$env:POSTGRES_USER = "postgres"
$env:POSTGRES_PASSWORD = "your_password"
$env:POSTGRES_HOST = "127.0.0.1"
$env:POSTGRES_PORT = "5432"
```

3. Run migrations and start server:

```bash
python manage.py migrate
python manage.py runserver
```

- App serves pages from the frontend HTML templates in project root.
- Static files come from `css/`, `js/`, `img/`, `fonts/` directories.
- Prediction endpoint: POST `/predict/` with multipart form:
  - field `image`: uploaded image
  - field `symptoms`: JSON array of strings

Response:

```json
{"disease":"Unknown","confidence":0.0}
```

Replace the stubbed logic in `siteapp/views.py::predict` with your ML model.

