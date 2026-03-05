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
