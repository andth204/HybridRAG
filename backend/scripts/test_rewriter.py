import asyncio
from src.hybridrag.rewriter import query_reflection

async def rewrite_has_history_main_query():
    chat_history = [
        {"role": "user", "content": "Trường có những ngành nào đang tuyển sinh vậy?"},
        {"role": "assistant", "content": "Trường hiện đang tuyển sinh các ngành như Công nghệ thông tin, Kỹ thuật phần mềm, Quản trị kinh doanh, Kế toán, Ngôn ngữ Anh và một số ngành khác thuộc khối kỹ thuật và kinh tế."},
        {"role": "user", "content": "Mình thích học ngành liên quan đến lập trình, phát triển phần mềm thì nên chọn ngành nào?"},
        {"role": "assistant", "content": "Bạn nên chọn ngành Kỹ thuật phần mềm hoặc Công nghệ thông tin, cả hai ngành đều tập trung vào lập trình và phát triển phần mềm."},
        {"role": "user", "content": "Ngành này học trong bao lâu và ra trường làm gì?"},
        {"role": "assistant", "content": "Thời gian đào tạo là 4 năm, sau khi tốt nghiệp có thể làm lập trình viên, phân tích hệ thống, quản lý dự án công nghệ thông tin tại các công ty phần mềm."},
        {"role": "user", "content": "Trường có chương trình thực tập không?"},
        {"role": "assistant", "content": "Trường có chương trình thực tập bắt buộc trong học kỳ cuối, sinh viên sẽ được giới thiệu đến các doanh nghiệp đối tác để thực hành."},
        {"role": "user", "content": "Học phí ngành này là bao nhiêu?"},
        {"role": "assistant", "content": "Học phí ngành Công nghệ thông tin và Kỹ thuật phần mềm là 15 triệu đồng/năm, tổng cộng khoảng 60 triệu cho toàn khóa học."},
        {"role": "user", "content": "Trường có hỗ trợ việc làm sau tốt nghiệp không?"},
        ]

    # Current vague query
    current_query = "thu7?"

    for msg in chat_history:
        if msg["role"] == "user":
            print(f"  - {msg['content']}")
    print(f"\n<> Current query: {current_query}")
    print(f"\n-> Refined query: {await query_reflection.reflect(current_query, chat_history)}")

# Run the async function
asyncio.run(rewrite_has_history_main_query())