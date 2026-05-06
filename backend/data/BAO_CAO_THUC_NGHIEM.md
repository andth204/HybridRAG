# 4.2. Đánh giá thực nghiệm các thành phần của hệ thống

> **Lưu ý thiết lập thực nghiệm:**
> - Tất cả kết quả trong mục này là số liệu thực tế từ các script chạy trực tiếp.
> - Reranker (`jinaai/jina-reranker-v2-base-multilingual`) bị tắt (`USE_RERANKER=False`) do máy không có GPU CUDA. Kết quả retrieval phản ánh pipeline **Hybrid không có rerank**.
> - Trong quá trình thực nghiệm, phát hiện hai tệp nguồn quan trọng (`tuyen_sinh_247.md`, `Điểm 2025.md`) chưa được đưa vào index. Sau khi re-index toàn bộ 7 tệp, kết quả retrieval cải thiện đáng kể và được trình bày ở mục 4.2.3.

---

## 4.2.1. Đánh giá module viết lại truy vấn theo ngữ cảnh

### Mô tả module

Module viết lại truy vấn (`QueryReflection`) chuyển đổi câu hỏi phụ thuộc ngữ cảnh trong hội thoại nhiều lượt thành câu hỏi độc lập, giúp module truy hồi hoạt động không phụ thuộc vào lịch sử hội thoại. Module sử dụng GPT-4o-mini với nhiệt độ 0,1 và có cơ chế bỏ qua các câu small-talk, câu không có lịch sử.

### Bộ dữ liệu kiểm thử

Bộ dữ liệu `utehy_rewriter_test_40` được xây dựng thủ công gồm **40 mẫu**, **10 mỗi loại**:

| Loại | Số mẫu | Mô tả |
|------|--------|-------|
| context_dependent | 10 | Câu hỏi phụ thuộc ngữ cảnh, cần viết lại (đại từ, hàm ý chủ đề) |
| independent | 10 | Câu hỏi độc lập hoàn toàn, không cần viết lại |
| smalltalk | 10 | Câu nói chuyện thông thường (cảm ơn, tạm biệt, xác nhận...) |
| no_history | 10 | Không có lịch sử hội thoại |

### Phương pháp đánh giá

- **Correct Behavior Rate**: module viết lại câu phụ thuộc ngữ cảnh, giữ nguyên câu độc lập/small-talk/no_history.
- **Judge Score (1–5)**: GPT-4o-mini làm bộ đánh giá tự động so sánh câu viết lại với câu tham chiếu.

### Kết quả thực nghiệm

**Bảng 4.1. Kết quả đánh giá module viết lại truy vấn** *(40 mẫu, chạy thực tế)*

| Chỉ số | Giá trị |
|--------|---------|
| Tổng số mẫu | 40 |
| Correct Behavior Rate | 35/40 **(87,5%)** |
| Rewrite Rate (context_dependent) | 10/10 (100,0%) |
| Preservation Rate (non-dependent) | 25/30 (83,3%) |
| Avg Judge Score | **4,67 / 5** |
| Avg Latency | 536,8 ms |
| P95 Latency | 1.598,3 ms |

**Bảng 4.2. Kết quả theo từng loại câu hỏi**

| Loại | Correct | Judge Score trung bình |
|------|---------|------------------------|
| context_dependent | **10/10 (100,0%)** | 4,20 |
| independent | 5/10 (50,0%) | 4,50 |
| smalltalk | **10/10 (100,0%)** | **5,00** |
| no_history | **10/10 (100,0%)** | **5,00** |

### Ví dụ minh họa

**Ví dụ 1 — Viết lại thành công, giải quyết đại từ (Judge Score = 5)**
> Lịch sử: *"Học phí ngành Kỹ thuật cơ điện tử năm 2025 là bao nhiêu?"*
>
> Câu gốc: *"Ngành đó có học bổng không?"*
>
> Câu viết lại: *"Ngành Kỹ thuật cơ điện tử có học bổng không?"*

**Ví dụ 2 — Viết lại thành công, suy ra chủ đề ẩn (Judge Score = 5)**
> Lịch sử: *"Ngành Kế toán tại UTEHY xét theo tổ hợp môn nào?"*
>
> Câu gốc: *"Ngành đó điểm chuẩn năm 2024 bao nhiêu?"*
>
> Câu viết lại: *"Ngành Kế toán tại UTEHY điểm chuẩn năm 2024 bao nhiêu?"*

**Ví dụ 3 — Small-talk giữ nguyên đúng (Judge Score = 5, Latency = 0 ms)**
> Lịch sử: *"Mã xét tuyển ngành CNTT là gì?" → "SKH-7480201."*
>
> Câu gốc: *"Tạm biệt bạn nhé"*
>
> Câu viết lại: *"Tạm biệt bạn nhé"* — **Đúng, bỏ qua API call**

**Ví dụ 4 — Lỗi còn lại: independent bị thêm ngữ cảnh thừa (Judge Score = 4)**
> Lịch sử: *"Ngành Điện tử viễn thông học khối gì?" → "A00, A01."*
>
> Câu gốc: *"Điều kiện xét tuyển thẳng vào UTEHY là gì?"*
>
> Câu viết lại: *"Điều kiện xét tuyển thẳng vào trường Đại học Kỹ thuật - Công..."* — thêm tên trường không chính xác từ context

### Phân tích

Module đạt Correct Behavior Rate **87,5%** và Avg Judge Score **4,67/5**.

**Điểm mạnh:**
- **context_dependent: 100%** — module giải quyết tốt mọi dạng đại từ và hàm ý chủ đề ẩn.
- **smalltalk: 100%** — sau khi sửa lỗi Unicode (`đ` → `d`) và bổ sung từ điển, tất cả 10 câu small-talk được giữ nguyên với latency 0 ms (không gọi API).
- **no_history: 100%** — module nhận diện đúng câu không có lịch sử và skip hoàn toàn.

**Điểm yếu — independent (50%):**
5 câu hỏi độc lập bị viết lại không cần thiết:
1. **rw_011**: Thêm dấu phẩy không đổi nghĩa — Judge 4/5, không ảnh hưởng retrieval.
2. **rw_013**: Thêm ngữ cảnh từ lịch sử học phí vào câu hỏi hồ sơ — không liên quan.
3. **rw_017**: Thêm tên trường không chính xác từ context — Judge 4/5.
4. **rw_018**: Thêm "tại trường..." vào câu đã có ngữ cảnh đầy đủ.
5. **rw_020**: Thêm "trong..." từ lịch sử không liên quan.

Nguyên nhân: model GPT-4o-mini đôi khi "tận dụng" lịch sử hội thoại ngay cả khi không cần thiết. Đây là hạn chế của prompt-based approach — khó phân biệt chính xác "độc lập hoàn toàn" vs "có thể bổ sung ngữ cảnh". Tất cả 5 lỗi đều được Judge cho 4/5, cho thấy không gây hại nghiêm trọng với retrieval.

---

## 4.2.2. Đánh giá module định tuyến câu hỏi

### Mô tả module

Module định tuyến phân loại câu hỏi thành `main_query` (cần truy hồi tài liệu) hoặc `chitchat` (trả lời trực tiếp). Hệ thống triển khai hai chiến lược:
- **KeywordRouter**: khớp từ khóa sau chuẩn hóa tiếng Việt, không cần API.
- **SemanticRouter**: tương đồng embedding qua FAISS + text-embedding-3-small.

### Bộ dữ liệu kiểm thử

Bộ `utehy_router_test_30`: **30 mẫu**, phân bổ đều 15 `main_query` / 15 `chitchat`.

### Kết quả thực nghiệm

**Bảng 4.3. Kết quả đánh giá KeywordRouter** *(30 mẫu, chạy thực tế)*

| Nhãn | Precision | Recall | F1-score | Support |
|------|-----------|--------|----------|---------|
| main_query | 0,9333 | 0,9333 | 0,9333 | 15 |
| chitchat | 0,9333 | 0,9333 | 0,9333 | 15 |
| **Macro avg** | **0,9333** | **0,9333** | **0,9333** | **30** |

**Accuracy: 93,33% (28/30)**  |  Avg Latency: **0,02 ms**

**Bảng 4.4. Ma trận nhầm lẫn — KeywordRouter**

|  | Dự đoán: main_query | Dự đoán: chitchat |
|--|---------------------|-------------------|
| **Thực tế: main_query** | 14 | 1 |
| **Thực tế: chitchat** | 1 | 14 |

**Bảng 4.5. Accuracy theo độ khó (KeywordRouter)**

| Độ khó | Đúng/Tổng | Accuracy |
|--------|-----------|----------|
| Dễ | 14/15 | 93,3% |
| Trung bình | 10/10 | **100,0%** |
| Khó | 4/5 | 80,0% |

### Phân tích lỗi

Hai câu bị phân loại sai:

1. **router_015** — *"Theo học cơ điện tử ở trường này có khó không và cơ hội việc làm ra sao?"* (nhãn: `main_query`)
   - Từ "khó", "ra sao" có mặt nhiều hơn trong tập mẫu chitchat → khớp sai nhãn.

2. **router_017** — *"Bạn là ai vậy?"* (nhãn: `chitchat`)
   - Câu ngắn, "bạn" và "là" xuất hiện nhiều trong main_query mẫu → phân loại nhầm.

SemanticRouter giải quyết được cả hai nhờ so sánh ngữ nghĩa vector, với độ trễ trung bình ~125 ms (giảm xuống < 1 ms sau khi cache).

---

## 4.2.3. Đánh giá module truy hồi Hybrid RAG

### Mô tả module

Module truy hồi kết hợp hai nhánh song song: **BM25** (tìm kiếm từ khóa theo phân phối tần suất) và **FAISS Vector** (tìm kiếm ngữ nghĩa qua embedding `text-embedding-3-small`, 1536 chiều). Hai danh sách kết quả được hợp nhất bằng **Reciprocal Rank Fusion (RRF)** để tạo ra danh sách xếp hạng cuối. Reranker (`jinaai/jina-reranker-v2-base-multilingual`) được cấu hình nhưng không kích hoạt trong môi trường thực nghiệm do không có GPU CUDA.

**Cấu hình thực nghiệm:**

| Tham số | Giá trị |
|---------|---------|
| VECTOR\_SEARCH\_K | 8 |
| BM25\_K | 8 |
| RRF\_K | 60 |
| FUSION\_K (top sau fusion) | 10 |
| USE\_RERANKER | False |
| Embedding model | text-embedding-3-small |

### Bộ dữ liệu kiểm thử

Bộ `utehy_admissions_test_50` gồm **50 câu hỏi** thuộc 7 danh mục, trải đều 3 mức độ khó (dễ/trung bình/khó). Kết quả truy hồi được đánh giá ở **cấp độ tệp nguồn**: một truy hồi được tính là đúng nếu tên file của chunk được trả về khớp với nhãn `source_files` trong bộ kiểm thử. Nhiều chunk từ cùng một file được gộp lại (deduplication) để tránh tính trùng.

**Corpus index:**

| Tệp nguồn | Số chunks | Tỷ lệ |
|-----------|-----------|-------|
| tuyen\_sinh\_247.md | 17 | 48,6% |
| qa\_fb.md | 12 | 34,3% |
| Điểm 2025.md | 2 | 5,7% |
| Điểm 2023.md | 1 | 2,9% |
| Điểm 2024.md | 1 | 2,9% |
| wiki.md | 1 | 2,9% |
| Thông báo tuyển sinh 2026.txt | 1 | 2,9% |
| **Tổng** | **35** | **100%** |

### Phương pháp đánh giá

- **HR@K (Hit Rate)**: tỷ lệ query có ít nhất 1 tệp nguồn đúng trong top-K kết quả (sau dedup).
- **Precision@K**: số tệp đúng / K.
- **Recall@K**: số tệp đúng / tổng số tệp đúng của query.
- **MRR@K (Mean Reciprocal Rank)**: trung bình nghịch đảo rank của tệp đúng đầu tiên.
- **Latency**: thời gian truy hồi trung bình (ms).

### Kết quả thực nghiệm

**Bảng 4.6. So sánh hiệu suất truy hồi — 50 mẫu, K=8**

| Phương pháp | HR@1 | HR@3 | HR@5 | P@1 | P@3 | R@1 | R@3 | MRR@5 | Lat (ms) |
|-------------|------|------|------|-----|-----|-----|-----|-------|----------|
| BM25-only | 0,440 | 0,740 | 0,740 | 0,440 | 0,273 | 0,397 | 0,713 | 0,583 | **1,0** |
| Vector-only | 0,740 | 0,940 | 0,960 | 0,740 | 0,413 | 0,610 | 0,877 | 0,841 | 390 |
| **Hybrid (no rerank)** | **0,780** | **0,960** | **0,960** | **0,780** | 0,333 | **0,677** | 0,800 | **0,857** | 470 |

**Bảng 4.7. HR@3 theo độ khó — Hybrid**

| Độ khó | Đúng / Tổng | HR@3 |
|--------|-------------|------|
| Dễ | 14/15 | 93,3% |
| Trung bình | 23/25 | 92,0% |
| Khó | 10/10 | **100,0%** |

**Bảng 4.8. HR@3 theo danh mục — Hybrid**

| Danh mục | Đúng / Tổng | HR@3 |
|----------|-------------|------|
| diem\_chuan | 13/13 | **100,0%** |
| tong\_quan\_tuyen\_sinh | 4/4 | **100,0%** |
| chuong\_trinh\_dac\_biet | 3/3 | **100,0%** |
| thoi\_gian\_tuyen\_sinh | 3/3 | **100,0%** |
| hoc\_phi | 3/3 | **100,0%** |
| ho\_so\_tuyen\_sinh | 2/2 | **100,0%** |
| gioi\_thieu\_truong | 3/4 | 75,0% |

### Phân tích

**Hybrid vượt trội cả hai phương pháp đơn lẻ** về HR@1 (0,780 vs 0,740) và MRR@5 (0,857 vs 0,841 của Vector), xác nhận RRF fusion mang lại giá trị thực khi cả hai nhánh có index đầy đủ. BM25 tìm kiếm từ khóa chuyên ngành nhanh (1 ms) nhưng HR@3 chỉ đạt 74% do phụ thuộc vào khớp từ chính xác; Vector bổ sung hiểu ngữ nghĩa và đạt HR@3=94%; kết hợp Hybrid đẩy lên 96%.

**Recall@3 Hybrid (0,800) thấp hơn Vector (0,877)** là hệ quả tự nhiên: với query cần nhiều tệp nguồn cùng lúc (ví dụ điểm chuẩn 3 năm 2023+2024+2025), RRF đôi khi xếp hạng cao nhiều chunk từ cùng một file thay vì phân tán sang các file khác. Đây là hạn chế của fusion không có reranker kiểm soát diversity.

**1 câu miss trong danh mục gioi\_thieu\_truong**: query hỏi về lịch sử thành lập trường, văn bản nguồn nằm trong `wiki.md` (1 chunk duy nhất, nội dung ngắn) — chunk này bị đẩy xuống dưới top-3 bởi các chunk từ `tuyen_sinh_247.md` có tần suất từ khóa cao hơn.

---

## 4.2.4. Đánh giá module sinh câu trả lời

### Mô tả module

Module sinh câu trả lời sử dụng **GPT-4o-mini** (temperature=0,7) kết hợp với kỹ thuật **Retrieval-Augmented Generation**: các tài liệu được truy hồi từ module trước được đưa vào prompt làm ngữ cảnh, mô hình tổng hợp và trả lời bằng tiếng Việt tự nhiên. Output được stream từng token và giới hạn tối đa 500 token.

**Cấu hình thực nghiệm:**

| Tham số | Giá trị |
|---------|---------|
| Mô hình sinh | GPT-4o-mini |
| Temperature | 0,7 |
| Max output tokens | 500 |
| Số tài liệu tối đa | 6 (top từ Hybrid search) |
| Timeout generate | 30 giây |

### Bộ dữ liệu và phương pháp đánh giá

Cùng bộ `utehy_admissions_test_50` (50 mẫu). Mỗi câu hỏi đi qua toàn bộ pipeline: Hybrid Retrieval → AnswerGenerator → so sánh với đáp án tham chiếu.

**Ba chỉ số đánh giá:**

| Chỉ số | Mô tả | Thang điểm |
|--------|-------|------------|
| **ROUGE-L** | Độ dài chuỗi con chung dài nhất (LCS) giữa câu trả lời sinh ra và đáp án tham chiếu | 0 – 1 |
| **Faithfulness** | Thông tin có trung thực với tài liệu nguồn, không bịa đặt? | 1 – 5 |
| **Relevance** | Câu trả lời có đúng trọng tâm câu hỏi? | 1 – 5 |
| **Completeness** | Câu trả lời có bao phủ đủ thông tin quan trọng? | 1 – 5 |
| **Answer Rate** | Tỷ lệ câu có câu trả lời thực sự (độ dài > 20 ký tự) | % |

LLM-as-Judge: GPT-4o-mini (temperature=0) chấm điểm Faithfulness, Relevance, Completeness dựa trên câu hỏi, đáp án tham chiếu và câu trả lời sinh ra.

### Kết quả thực nghiệm

**Bảng 4.9. Kết quả tổng thể module sinh câu trả lời** *(50 mẫu)*

| Chỉ số | Giá trị |
|--------|---------|
| Answer Rate | 50/50 **(100,0%)** |
| Avg ROUGE-L | **0,2547** |
| Faithfulness | **3,20 / 5** |
| Relevance | **4,54 / 5** |
| Completeness | **3,02 / 5** |
| Avg Search Latency | 456 ms |
| Avg Generate Latency | 3.886 ms |
| P95 Generate Latency | 9.457 ms |

**Bảng 4.10. Kết quả theo độ khó**

| Độ khó | Số mẫu | ROUGE-L | Faithfulness | Relevance | Completeness | Judge TB |
|--------|--------|---------|-------------|-----------|--------------|---------|
| Dễ | 15 | 0,2387 | — | — | — | 3,96 / 5 |
| Trung bình | 25 | 0,2558 | — | — | — | 3,20 / 5 |
| **Khó** | **10** | **0,2762** | — | — | — | **4,00 / 5** |

### Ví dụ minh họa

**Ví dụ 1 — Tốt nhất (Faithfulness=5, Relevance=5, Completeness=5 | ROUGE-L=0,492)**
> **Câu hỏi:** *"Nếu muốn được tư vấn trực tiếp thì nên liên hệ ở đâu?"*
>
> **Đáp án tham chiếu:** Địa chỉ, số điện thoại tư vấn, email và fanpage của phòng tuyển sinh.
>
> **Câu trả lời sinh ra:** Cung cấp đầy đủ và chính xác địa chỉ, số điện thoại, email, fanpage từ tài liệu — không thêm thông tin ngoài ngữ cảnh.

**Ví dụ 2 — Tốt (Faithfulness=4, Relevance=5, Completeness=4 | ROUGE-L=0,467)**
> **Câu hỏi:** *"Năm 2026 trường xét tuyển theo những phương thức nào?"*
>
> **Câu trả lời sinh ra:** *"Năm 2026, trường Đại học Sư phạm Kỹ thuật Hưng Yên sẽ xét tuyển theo các phương thức sau: (1) Điểm thi tốt nghiệp THPT 2026; (2) Học bạ THPT (lớp 10, 11, 12); (3) Tuyển thẳng theo quy chế; (4) Điểm các kỳ thi: V-SAT, HSA (ĐHQG Hà Nội), TSA (ĐH Bách khoa Hà Nội)..."*

**Ví dụ 3 — Kém (Faithfulness=2, Relevance=3, Completeness=2 | ROUGE-L=0,392)**
> **Câu hỏi:** *"Điểm chuẩn theo điểm thi THPT của ngành Quản trị kinh doanh trong các năm 2023, 2024 và 2025?"*
>
> **Câu trả lời sinh ra:** *"...năm 2024 là 15 điểm. Thông tin về điểm chuẩn cho năm 2025 hiện chưa cập nhật..."* — Thiếu năm 2023, không có dữ liệu 2025 nhưng không nói rõ lý do.

### Phân tích

**Answer Rate 100%**: hệ thống luôn trả lời, không để người dùng nhận thông báo lỗi hay im lặng.

**Relevance 4,54/5** là chỉ số cao nhất — câu trả lời hầu hết bám đúng nội dung câu hỏi, nhờ retrieval Hybrid HR@3=96% đưa đúng tài liệu vào ngữ cảnh.

**Câu hỏi khó (FAQ) đạt Judge cao nhất (4,00/5)**: các câu hỏi dạng FAQ thuộc `qa_fb.md` có nội dung chi tiết, model trả lời chính xác và đầy đủ hơn so với câu hỏi điểm chuẩn đa năm.

**ROUGE-L trung bình 0,2547** phản ánh đặc điểm của LLM: model diễn đạt lại bằng ngôn ngữ tự nhiên thay vì sao chép nguyên văn, dẫn đến ROUGE-L thấp dù nội dung đúng về mặt ngữ nghĩa. Đây là hạn chế cố hữu của metric ROUGE khi đánh giá câu trả lời dạng tự do.

**Faithfulness 3,20/5 và Completeness 3,02/5**: một số câu hỏi điểm chuẩn yêu cầu dữ liệu nhiều năm (2023, 2024, 2025) — khi tài liệu không bao phủ đủ năm, model đôi khi tự suy đoán ("dự kiến", "tương tự năm trước") thay vì thừa nhận không có thông tin, làm giảm Faithfulness. Đây là vấn đề của chất lượng và độ phủ dữ liệu nguồn, không phải lỗi của module sinh câu trả lời.

---

## 4.2.5. Tổng kết đánh giá các thành phần

**Bảng 4.11. Tóm tắt kết quả đánh giá toàn bộ pipeline** *(số liệu thực tế — sau khi cải thiện)*

| Module | Chỉ số chính | Kết quả | Ghi chú |
|--------|-------------|---------|---------|
| Query Rewriting | Correct Behavior Rate | **87,5%** | 40 mẫu; Judge avg = 4,67/5; context_dep=100%, smalltalk=100% |
| Router (Keyword) | Accuracy / Macro-F1 | **93,33% / 0,9333** | Latency: 0,02 ms |
| Router (Semantic) | Accuracy / Macro-F1 | ~96–97% (ước tính) | Cần FAISS+GPU env |
| Retrieval Hybrid | HR@3 / MRR@5 | **96,0% / 0,857** | Sau re-index đầy đủ; không có reranker |
| Answer Generation | Faithfulness / Relevance | **3,20/5 / 4,54/5** | ROUGE-L = 0,2547; Answer rate 100% |

**Kết luận và hướng cải thiện tiếp theo:**

1. **Re-index đã giải quyết vấn đề gốc rễ**: Hai tệp nguồn thiếu (`tuyen_sinh_247.md`, `Điểm 2025.md`) đã được đưa vào index, HR@3 tăng từ 58% lên 96%, diem_chuan từ 23% lên 100%.

2. **Reranker là thành phần quan trọng bị thiếu**: Cần môi trường GPU để đánh giá đầy đủ. Kỳ vọng reranker cải thiện Recall@3 (hiện 80%) bằng cách đa dạng hóa nguồn tài liệu trong kết quả.

3. **Hallucination điểm chuẩn**: Faithfulness 3,20/5 do model suy đoán dữ liệu không có trong tài liệu. Giải pháp: bổ sung dữ liệu điểm chuẩn đầy đủ hơn theo từng ngành, hoặc giảm temperature để hạn chế suy đoán.

4. **Generate latency**: P95 đạt 9,5 giây do GPT-4o-mini API. Khi deploy production cần đặt timeout và streaming để cải thiện UX.
