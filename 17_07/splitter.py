import os
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
# Kế thừa từ Chapter 2 bằng cách nhập hàm nạp dữ liệu từ loader.py
from loader import load_source, analyze_documents

def split_documents(documents: list, chunk_size: int = 500, chunk_overlap: int = 50, strategy: str = "recursive") -> list:
    """
    Hàm chia nhỏ tài liệu (chunking) dựa trên chiến lược được lựa chọn.
    
    Tham số:
    - documents: Danh sách các Document Object đầu vào.
    - chunk_size: Kích thước tối đa của mỗi đoạn (tính bằng số ký tự).
    - chunk_overlap: Độ dài phần gối đầu (overlap) giữa các đoạn liền kề.
    - strategy: Chiến lược chia ("recursive" hoặc "character").
    """
    if not documents:
        print("Không có tài liệu đầu vào để chia nhỏ.")
        return []
        
    if strategy == "recursive":
        # Sử dụng RecursiveCharacterTextSplitter (Ưu tiên ngắt dòng tự nhiên: \n\n, \n, khoảng trắng)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    elif strategy == "character":
        # Sử dụng CharacterTextSplitter (Ngắt cứng theo một ký tự xác định, mặc định là \n\n)
        splitter = CharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separator="\n\n"
        )
    else:
        raise ValueError(f"Chiến lược '{strategy}' không được hỗ trợ. Sử dụng 'recursive' hoặc 'character'.")
        
    chunks = splitter.split_documents(documents)
    return chunks

def analyze_chunks(chunks: list):
    """Phân tích các đoạn tài liệu sau khi chia nhỏ."""
    if not chunks:
        print("Danh sách chunks trống.")
        return
        
    print("\n" + "="*50)
    print(f"🧩 KẾT QUẢ PHÂN TÍCH CHUNKS: Đã tạo thành công {len(chunks)} Chunks")
    print("="*50)
    
    lengths = [len(chunk.page_content) for chunk in chunks]
    max_len = max(lengths) if lengths else 0
    min_len = min(lengths) if lengths else 0
    avg_len = sum(lengths) / len(lengths) if lengths else 0
    
    print(f"📊 Thống kê ký tự:")
    print(f"  - Chunk dài nhất: {max_len} ký tự")
    print(f"  - Chunk ngắn nhất: {min_len} ký tự")
    print(f"  - Độ dài trung bình: {avg_len:.1f} ký tự")
    print("-" * 40)
    
    # Hiển thị mẫu 3 chunks đầu tiên
    for idx, chunk in enumerate(chunks[:3]):
        print(f"\n📦 Chunk #{idx + 1}:")
        print(f"  - Nguồn gốc: {chunk.metadata.get('source', 'N/A')} (Trang {chunk.metadata.get('page', 'N/A')})")
        print(f"  - Độ dài: {len(chunk.page_content)} ký tự")
        print(f"  - Nội dung: {chunk.page_content.strip().replace('\n', ' ')[:150]}...")
        print("-" * 30)
        
    if len(chunks) > 3:
        print(f"... Và {len(chunks) - 3} chunks khác.")
    print("="*50)

if __name__ == "__main__":
    print("=== Chạy thử nghiệm splitter.py ===")
    
    # Đường dẫn file test
    pdf_path = "/workspace/scratch/data/sample.pdf"
    
    if os.path.exists(pdf_path):
        print(f"Nạp dữ liệu từ {pdf_path} thông qua loader.py...")
        try:
            # Tái sử dụng hàm load_source của loader.py
            raw_docs = load_source(pdf_path)
            
            # Thử nghiệm Recursive Character Text Splitter (Chiến lược khuyên dùng)
            print("\n1. Thực hiện chia nhỏ bằng RecursiveCharacterTextSplitter (Size=300, Overlap=30):")
            chunks_rec = split_documents(raw_docs, chunk_size=300, chunk_overlap=30, strategy="recursive")
            analyze_chunks(chunks_rec)
            
        except Exception as e:
            print(f"Có lỗi xảy ra khi chia nhỏ tài liệu: {e}")
    else:
        print(f"Không tìm thấy file thử nghiệm tại '{pdf_path}'. Vui lòng chạy loader.py hoặc tạo file trước.")
