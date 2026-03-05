# Frontend Setup

README nay chi gom huong dan thiet lap va cai dat.

## 1) Yeu cau moi truong

- Node.js: `>= 22.12` (hoac `20.19`)
- npm: di kem Node.js

Kiem tra version:

```bash
node -v
npm -v
```

## 2) Cai dat

Trong thu muc `frontend`:

```bash
npm install
```

## 3) Chay development

```bash
npm run dev
```

Mac dinh Vite in URL local (thuong la `http://localhost:5173`).

## 4) Build production

```bash
npm run build
```

Lenh nay se:

- Type-check TypeScript
- Build bundle production bang Vite

## 5) Preview ban build

```bash
npm run preview
```

## 6) Lint code (tuy chon)

```bash
npm run lint
```

## 7) Loi thuong gap

### Loi version Node khong dung

Neu gap loi `engines` hoac package khong tuong thich:

1. Cap nhat Node len ban yeu cau
2. Xoa thu muc `node_modules` va file `package-lock.json`
3. Cai lai:

```bash
npm install
```

### Loi cache npm

```bash
npm cache verify
```
