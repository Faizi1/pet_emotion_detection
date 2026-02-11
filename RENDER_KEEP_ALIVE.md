# Keep Your Render Django App Running (Free Tier)

On Render’s **free tier**, web services spin down after **~15 minutes** of no traffic. The next request wakes the service (cold start), which can take **30–90+ seconds**. These options keep the app responsive or avoid cold starts.

---

## Option 1: External Ping (Free, Recommended)

Use an external service to **hit your app every 10–14 minutes**. That keeps the instance awake so real users don’t hit cold starts.

### Health endpoint

Your app now has a lightweight endpoint for pings and Render health checks:

- **URL:** `https://YOUR-RENDER-URL.onrender.com/api/health`
- **Method:** GET  
- **Response:** `200` with `{"status": "ok"}`

Use this URL (and only this path) for keep-alive pings so you don’t trigger heavy endpoints.

---

### A. UptimeRobot (easiest, free)

1. Go to [uptimerobot.com](https://uptimerobot.com) and create a free account.
2. **Add New Monitor**
   - Monitor Type: **HTTP(s)**
   - Friendly Name: e.g. `Pet Emotion API`
   - URL: `https://YOUR-RENDER-URL.onrender.com/api/health`
   - Monitoring Interval: **5 minutes** (free tier)
3. Save. UptimeRobot will request that URL every 5 minutes, so the free instance stays awake and you get basic uptime alerts.

---

### B. cron-job.org (free)

1. Go to [cron-job.org](https://cron-job.org) and sign up.
2. Create a new cron job:
   - **Title:** e.g. `Render keep-alive`
   - **URL:** `https://YOUR-RENDER-URL.onrender.com/api/health`
   - **Schedule:** every **10 minutes** (e.g. `*/10 * * * *`)
3. Save. The job will GET your health URL every 10 minutes from their servers.

---

### C. GitHub Actions (free, no extra account)

A workflow is already in the repo: `.github/workflows/keep-alive.yml`.

1. Edit that file and set the `RENDER_APP_URL` env (or replace `YOUR-RENDER-URL` in the `curl` command) to your real Render URL, e.g. `https://pet-emotion-api.onrender.com`.
2. Commit and push. GitHub will run the job every 10 minutes and GET your `/api/health` endpoint, so the free-tier instance stays awake.

---

## Option 2: Render health check (recommended anyway)

Use the same `/api/health` endpoint for Render’s own health check. This doesn’t prevent spin-down on the free tier, but it helps Render know when the app is up and can restart bad deploys.

1. In **Render Dashboard** → your **Web Service** → **Settings**.
2. **Health Check Path:** set to `/api/health`.
3. Save.

---

## Option 3: Paid plan (always on, no cold start)

If you need the app to **run 24/7 with no spin-down**:

- In Render, upgrade the **Web Service** to a **paid plan** (e.g. **Starter**).
- The service stays running continuously; no cold starts and no need for keep-alive pings for uptime.

---

## Summary

| Goal                         | What to do                                                                 |
|-----------------------------|----------------------------------------------------------------------------|
| Stay on free tier, no cold start | Use **Option 1** (UptimeRobot, cron-job.org, or GitHub Actions) to ping `/api/health` every 10–14 minutes. |
| Better reliability on Render     | Set **Health Check Path** to `/api/health` (Option 2).                    |
| No spin-down at all              | Upgrade to a paid Render plan (Option 3).                                 |

**Recommended:** Use **Option 1** (e.g. UptimeRobot at 5-minute interval) and **Option 2** (health check path). That keeps the free tier instance awake and gives you basic monitoring and healthier deploys.
