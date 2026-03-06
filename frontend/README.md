# HybridRAG Frontend

[![Node.js](https://img.shields.io/badge/Node.js-%5E20.19%20%7C%7C%20%3E%3D22.12-339933.svg)](#prerequisites)
[![Vue](https://img.shields.io/badge/Vue-3.5-42B883.svg)](#tech-stack)
[![Vite](https://img.shields.io/badge/Vite-7-646CFF.svg)](#tech-stack)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.9-3178C6.svg)](#tech-stack)

Modern Vue 3 frontend for the HybridRAG workspace, including chat, history, documents, users, and statistics views.

## Table of Contents

- [Highlights](#highlights)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Environment Variables](#environment-variables)
- [Quick Start](#quick-start)
- [Available Scripts](#available-scripts)
- [Troubleshooting](#troubleshooting)

## Highlights

- Vue 3 + TypeScript app bootstrapped with Vite for fast local development.
- Pinia-based state management across workspace modules.
- UI built with Naive UI, Tailwind CSS v4, and VueUse utilities.
- Streaming-ready SSE client (`fetch-event-source`) for chat integrations.
- Linting pipeline with both ESLint and Oxlint.

## Tech Stack

- Vue `3.5.x`
- Vite `7.x`
- TypeScript `5.9.x`
- Pinia `3.x`
- Naive UI `2.x`
- Tailwind CSS `4.x`

## Project Structure

```text
frontend/
  public/
  src/
    components/
      chat/
      documents/
      layout/
      statistics/
      users/
    router/
    services/
    stores/
    views/
  index.html
  vite.config.ts
  eslint.config.ts
  package.json
```

## Prerequisites

- Node.js: `^20.19.0` or `>=22.12.0`
- npm: bundled with Node.js

Verify local versions:

```bash
node -v
npm -v
```

## Environment Variables

Create `frontend/.env` and configure:

```env
VITE_API_BASE_URL=/api/v1
VITE_DEV_API_TARGET=http://localhost:8000
VITE_DEV_HOST=localhost
VITE_DEV_PORT=5173
VITE_GOOGLE_CLIENT_ID=your_google_oauth_client_id
```

Notes:

- `VITE_API_BASE_URL=/api/v1` works with Vite dev proxy.
- `VITE_DEV_API_TARGET` points to backend API host in local development.
- `VITE_DEV_HOST` + `VITE_DEV_PORT` define your frontend origin (default: `http://localhost:5173`).
- Frontend login uses Google auth only (`POST /api/v1/auth/google`).
- `Remember for 30 days` stores session in `localStorage`; unchecked mode stores session per-tab (`sessionStorage`).

## Quick Start

Run all commands from `frontend/`.

1. Install dependencies:

```bash
npm install
```

2. Start development server:

```bash
npm run dev
```

3. Open the local URL printed by Vite (typically `http://localhost:5173`).

## Available Scripts

| Command | Description |
| --- | --- |
| `npm run dev` | Start Vite dev server |
| `npm run build` | Run type-check then create production build |
| `npm run build-only` | Build production bundle without type-check |
| `npm run type-check` | Run `vue-tsc --build` |
| `npm run preview` | Preview the production build locally |
| `npm run lint` | Run Oxlint and ESLint with auto-fix |

## Troubleshooting

### `ENOENT` when running `npm install`

If you run `npm install` from the repository root, npm fails because there is no `package.json` there.

Use:

```bash
cd frontend
npm install
```

### Node engine mismatch

If you get an `engines` compatibility error, install a supported Node version (`^20.19.0` or `>=22.12.0`) and reinstall dependencies.

### Clean reinstall on Windows

If dependencies become inconsistent or lock files are stale:

```powershell
Remove-Item -Recurse -Force node_modules
Remove-Item -Force package-lock.json
npm install
```

### npm cache issues

```bash
npm cache verify
```

### Google sign-in skipped or popup not shown

If Google login does not open or is skipped, verify:

- `VITE_GOOGLE_CLIENT_ID` in `frontend/.env` matches backend `GOOGLE_CLIENT_ID` exactly.
- Your Google OAuth client has **Authorized JavaScript origins** including your frontend origin (for example `http://localhost:5173`).
- You restarted the Vite dev server after any `.env` change.
- The browser is not blocking Google popup/sign-in windows for your frontend origin.

### Google error `origin_mismatch`

Google rejects sign-in when the current frontend origin is not registered in your OAuth Client.

1. Open Google Cloud Console -> `APIs & Services` -> `Credentials`.
2. Open your OAuth 2.0 Client ID (Web application) that matches `VITE_GOOGLE_CLIENT_ID`.
3. In `Authorized JavaScript origins`, add your exact frontend origin(s), for example:

```text
http://localhost:5173
http://127.0.0.1:5173
```

4. Save, wait 1-5 minutes for propagation, then restart frontend and try again.
