# Deploying to Render

The service runs on Render's free web plan (Docker) with a free hosted MySQL 8 database from
Aiven. Render has no managed MySQL. Total cost: free.

| Piece | Where | Notes |
|---|---|---|
| API (FastAPI + agent) | Render free web service, Docker, from `render.yaml` | 512 MB RAM (the app idles at ~160 MB); sleeps after 15 min idle, ~1 min to wake |
| MySQL 8 | Aiven free plan | 1 GB; TLS required; may be powered off when idle (power it on in the console) |
| Documents | committed in `data/raw/` | tcs.com blocks downloads from servers |
| Transcript index | committed in `data/chroma/gemini_models_gemini-embedding-001_v2/` | otherwise built in the background at startup (~4 min) |
| Secrets | Render dashboard (environment variables) | never committed |

---

## 1. Commit the transcript index (once, from your laptop)

Stop the local server first (Ctrl+C), then:

```
git pull
git add data/chroma/gemini_models_gemini-embedding-001_v2
git commit -m "chore: commit Gemini transcript index for deployment"
git push
```

## 2. Create the MySQL database on Aiven

1. Sign up at <https://aiven.io/free-mysql-database>. No card is needed.
2. Click **Create service**, choose **MySQL** and the **Free** plan, pick a region (e.g. Singapore or Mumbai if offered), then **Create**.
3. When it shows **Running**, open the service's **Overview** → **Connection information** and note:
   **Host**, **Port**, **User** (usually `avnadmin`) and **Password**.
4. On the same page, click **CA certificate → Download** (optional, but it lets the app verify the server).

The app creates the `tcs_forecast` database and its tables on first start.

## 3. Create the Render service

1. Sign up at <https://render.com> with your GitHub account.
2. **New → Blueprint** → select `Financial-Forecasting-Agent` → **Apply**. Render reads `render.yaml`.
3. Render asks for the values marked secret. Enter:

| Variable | Value |
|---|---|
| `GOOGLE_API_KEY` | your Gemini API key |
| `MYSQL_HOST` / `MYSQL_PORT` | from Aiven |
| `MYSQL_USER` / `MYSQL_PASSWORD` | from Aiven |
| `MYSQL_SSL_CA_CERT` | the full text of Aiven's `ca.pem` (open it in Notepad and paste everything), or leave empty: the connection is still encrypted, just not certificate-verified |
| `API_KEYS` | one or more secret keys you invent, comma-separated, e.g. `tcs-demo-7f3k9q2m` |

4. Deploy. The first build takes ~5–10 minutes. When the log shows `Application startup complete`, your URL is
   `https://tcs-forecasting-agent.onrender.com` (Render shows the exact one).

## 4. Test the deployment

Open `https://<your-app>.onrender.com`. It redirects to the interactive docs. Click **Authorize**, paste
your API key, then try **POST /forecast** with body `{}`.

PowerShell:

```powershell
$url = "https://<your-app>.onrender.com"
Invoke-RestMethod "$url/health"
Invoke-RestMethod -Method Post -Uri "$url/forecast" -Headers @{ "X-API-Key" = "tcs-demo-7f3k9q2m" } `
  -ContentType "application/json" -Body '{"task": "What are the main margin risks for TCS next quarter?"}' `
  -TimeoutSec 600 | ConvertTo-Json -Depth 8
```

curl:

```bash
curl -X POST https://<your-app>.onrender.com/forecast -H "X-API-Key: tcs-demo-7f3k9q2m" \
  -H "Content-Type: application/json" -d '{}'
```

## 5. Share it

Send people the URL and an API key. Rules that protect your quota:

- `POST /forecast` and `GET /forecast/{request_id}` need the `X-API-Key` header; `/docs`, `/health` and `/metrics` are open.
- Each key or IP gets at most `FORECAST_RATE_LIMIT_PER_HOUR` forecasts per hour (default 10). Beyond that it returns HTTP 429 with `Retry-After`.
- One key per person (`API_KEYS=key-for-alice,key-for-bob`) gives each person a separate limit. Remove a key and redeploy to revoke access.
- Every forecast still counts against your Gemini quota (~5 requests each). Enable billing on the Gemini project for heavier use.

## Updating and troubleshooting

- **Updates:** every `git push` to `main` redeploys automatically (`autoDeploy: true`).
- **First request after idle takes ~1 minute:** the free instance is waking up.
- **`/health` shows `database: unreachable`:** the Aiven free service may be powered off. Power it on in the Aiven console, and check `MYSQL_*`.
- **`vector_index_status: building`:** the index was not committed; wait ~4 minutes, or do step 1.
- **Logs:** Render dashboard → your service → **Logs** (live LLM and tool progress is printed there).
