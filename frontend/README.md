# HybridRAG Frontend

Frontend Vue 3 cho workspace HybridRAG. Ứng dụng này cung cấp giao diện chat, lịch sử hội thoại, tài liệu, thống kê và quản trị người dùng.

## Yêu cầu

- Node.js `^20.19.0` hoặc `>=22.12.0`
- npm đi kèm Node.js
- Backend API phải chạy được và truy cập được từ máy local

Kiểm tra nhanh:

```bash
node -v
npm -v
```

## Công nghệ chính

- Vue 3
- Vite 7
- TypeScript
- Pinia
- Naive UI
- Tailwind CSS 4

## Cài đặt

Chạy toàn bộ lệnh trong thư mục `frontend/`.

```bash
cd frontend
npm install
```

## Thiết lập môi trường

Tạo file `frontend/.env` với nội dung tối thiểu:

```env
VITE_API_BASE_URL=/api/v1
VITE_DEV_API_TARGET=http://localhost:8000
VITE_DEV_HOST=localhost
VITE_DEV_PORT=5173
VITE_GOOGLE_CLIENT_ID=your_google_oauth_client_id
```

Ý nghĩa từng biến:

- `VITE_API_BASE_URL`: base path mà frontend gọi API. Mặc định `/api/v1` để đi qua Vite proxy.
- `VITE_DEV_API_TARGET`: địa chỉ backend local mà Vite sẽ proxy sang.
- `VITE_DEV_HOST`: host của dev server frontend.
- `VITE_DEV_PORT`: port của dev server frontend.
- `VITE_GOOGLE_CLIENT_ID`: Google OAuth Client ID dùng cho đăng nhập.

## Thiết lập backend đi kèm

Frontend dev mode không gọi backend trực tiếp bằng CORS, mà đi qua Vite proxy:

```text
http://localhost:5173  ->  /api/*  ->  http://localhost:8000
```

Vì vậy trước khi chạy frontend, hãy đảm bảo:

- backend đang mở ở đúng địa chỉ mà `VITE_DEV_API_TARGET` trỏ tới
- backend đã startup xong và trả lời được request
- nếu backend chạy Docker và có warm-up model/reranker, cần đợi đến khi API thực sự `healthy`

Nếu backend chưa sẵn sàng, Vite có thể hiện lỗi kiểu:

```text
[vite] http proxy error: /api/v1/auth/google
Error: socket hang up
```

Đây thường là dấu hiệu backend chưa sẵn sàng hoặc bị đóng kết nối giữa chừng, không phải lỗi riêng của Vite.

## Thiết lập Google Login

Frontend đang dùng đăng nhập Google và gọi backend qua `POST /api/v1/auth/google`.

Cần đảm bảo:

1. `VITE_GOOGLE_CLIENT_ID` ở frontend khớp hoàn toàn với `GOOGLE_CLIENT_ID` ở backend.
2. Trong Google Cloud Console, OAuth Client phải có `Authorized JavaScript origins` chứa đúng origin frontend, ví dụ:

```text
http://localhost:5173
http://127.0.0.1:5173
```

3. Sau khi đổi `.env`, phải restart lại `npm run dev`.
4. Trình duyệt không chặn popup đăng nhập Google.

## Chạy local

Khởi động frontend:

```bash
npm run dev
```

Mặc định Vite sẽ mở ở:

```text
http://localhost:5173
```

## Build production

Build đầy đủ:

```bash
npm run build
```

Build không type-check:

```bash
npm run build-only
```

Xem bản build local:

```bash
npm run preview
```

## Scripts

| Command | Mô tả |
| --- | --- |
| `npm run dev` | Chạy Vite dev server |
| `npm run build` | Type-check rồi build production |
| `npm run build-only` | Build production không type-check |
| `npm run type-check` | Chạy `vue-tsc --build` |
| `npm run preview` | Preview bản build local |
| `npm run lint` | Chạy Oxlint và ESLint |

## Cấu trúc thư mục chính

```text
frontend/
  public/
  src/
    components/
    router/
    services/
    stores/
    views/
  index.html
  vite.config.ts
  package.json
```

## Lỗi thường gặp

### 1. `npm install` báo `ENOENT`

Bạn đang chạy lệnh sai thư mục. Hãy chạy trong `frontend/`:

```bash
cd frontend
npm install
```

### 2. Node version không đúng

Project yêu cầu:

```text
^20.19.0 hoặc >=22.12.0
```

Nếu dùng sai version, hãy đổi Node rồi cài lại dependency.

### 3. `socket hang up` khi login hoặc gọi API

Thường do một trong các nguyên nhân sau:

- backend chưa startup xong
- backend đang restart/crash
- `VITE_DEV_API_TARGET` trỏ sai
- backend đóng kết nối trước khi trả response

Kiểm tra nhanh:

- mở `http://localhost:8000/health/live`
- xem log backend
- xác nhận backend thật sự đã sẵn sàng trước khi login

### 4. Google báo `origin_mismatch`

Bạn chưa khai báo đúng frontend origin trong Google Cloud Console.

Ví dụ cần thêm:

```text
http://localhost:5173
http://127.0.0.1:5173
```

### 5. Đổi `.env` nhưng frontend không nhận

Vite chỉ đọc env khi start process. Sau khi sửa `.env`, cần dừng rồi chạy lại:

```bash
npm run dev
```
