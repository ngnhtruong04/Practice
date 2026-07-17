from langchain_core.documents import Document

# 1. Khởi tạo một Document Object chuẩn hóa trong LangChain
doc = Document(
    page_content="Mỗi nhân viên chính thức được hưởng 20 ngày nghỉ phép năm có hưởng lương.",
    metadata={
        "source": "hr_policy.pdf",
        "page": 5,
        "department": "HR"
    }
)

# 2. Truy xuất thông tin
print("--- Truy xuất Document Object ---")
print(f"Nội dung văn bản (page_content): {doc.page_content}")
print(f"Siêu dữ liệu (metadata): {doc.metadata}")
print(f"Nguồn tài liệu: {doc.metadata['source']} - Trang: {doc.metadata['page']}")