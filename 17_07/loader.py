import os
from urllib.parse import urlparse
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, WebBaseLoader

def is_url(source_path: str) -> bool:
    """Kiểm tra xem đường dẫn đầu vào có phải là một URL hay không."""
    try:
        result = urlparse(source_path)
        return all([result.scheme, result.netloc])
    except Exception:
        return False

def detect_source_type(source_path: str) -> str:
    """Tự động nhận diện loại nguồn đầu vào: PDF, DOCX, URL hoặc lỗi."""
    if is_url(source_path):
        return "url"
    
    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Lỗi: Không tìm thấy tệp tin tại đường dẫn: '{source_path}'")
        
    ext = os.path.splitext(source_path)[1].lower()
    if ext == ".pdf":
        return "pdf"
    elif ext == ".docx":
        return "docx"
    elif ext == ".txt":
        return "txt"
    else:
        raise ValueError(f"Lỗi: Định dạng tệp tin '{ext}' không được hỗ trợ. Chỉ hỗ trợ .pdf, .docx, và URL website.")

def load_source(source_path: str) -> list[Document]:
    """
    Hàm chính để nạp dữ liệu từ nhiều nguồn khác nhau.
    Tự động xác định loader phù hợp dựa trên đường dẫn tệp tin hoặc URL.
    """
    source_type = detect_source_type(source_path)
    
    try:
        if source_type == "pdf":
            loader = PyPDFLoader(source_path)
            docs = loader.load()
            # Bổ sung metadata tùy chỉnh để theo dõi nguồn dễ dàng hơn trong RAG
            for doc in docs:
                doc.metadata["loader_type"] = "PyPDFLoader"
            return docs
            
        elif source_type == "docx":
            loader = Docx2txtLoader(source_path)
            docs = loader.load()
            for doc in docs:
                doc.metadata["loader_type"] = "Docx2txtLoader"
            return docs
            
        elif source_type == "url":
            loader = WebBaseLoader(source_path)
            docs = loader.load()
            for doc in docs:
                doc.metadata["loader_type"] = "WebBaseLoader"
            return docs
            
        elif source_type == "txt":
            from langchain_community.document_loaders import TextLoader
            loader = TextLoader(source_path, encoding="utf-8")
            docs = loader.load()
            for doc in docs:
                doc.metadata["loader_type"] = "TextLoader"
            return docs
            
    except Exception as e:
        raise RuntimeError(f"Lỗi trong quá trình nạp tài liệu từ {source_path}: {str(e)}")

def analyze_documents(documents: list[Document]):
    """Phân tích và hiển thị thông tin tóm tắt về danh sách các Document Object."""
    if not documents:
        print("Danh sách Document trống.")
        return
        
    print("\n" + "="*50)
    print(f"📊 KẾT QUẢ PHÂN TÍCH: Đã nạp thành công {len(documents)} Document(s)")
    print("="*50)
    
    total_chars = 0
    for idx, doc in enumerate(documents):
        char_count = len(doc.page_content)
        total_chars += char_count
        print(f"\n📄 Document #{idx + 1}:")
        print(f"  - Nguồn (Source): {doc.metadata.get('source', 'N/A')}")
        print(f"  - Trang (Page): {doc.metadata.get('page', 'N/A')}")
        print(f"  - Loại Loader: {doc.metadata.get('loader_type', 'N/A')}")
        print(f"  - Độ dài nội dung: {char_count} ký tự")
        print(f"  - Bản xem trước (Preview): {doc.page_content[:150].strip().replace('\n', ' ')}...")
        print("-" * 40)
        
    print(f"\n📈 TỔNG CỘNG TOÀN BỘ FILES: {total_chars} ký tự.")
    print("="*50)

if __name__ == "__main__":
    print("=== Chạy thử nghiệm loader.py ===")
    
    # 1. Thử nghiệm với file PDF ảo
    pdf_path = "/workspace/scratch/data/sample.pdf"
    if os.path.exists(pdf_path):
        print(f"\n1. Thử nghiệm nạp tệp tin PDF: {pdf_path}")
        try:
            pdf_docs = load_source(pdf_path)
            analyze_documents(pdf_docs)
        except Exception as e:
            print(f"Lỗi nạp PDF: {e}")
            
    # 2. Thử nghiệm với file DOCX ảo
    docx_path = "/workspace/scratch/data/sample.docx"
    if os.path.exists(docx_path):
        print(f"\n2. Thử nghiệm nạp tệp tin DOCX: {docx_path}")
        try:
            docx_docs = load_source(docx_path)
            analyze_documents(docx_docs)
        except Exception as e:
            print(f"Lỗi nạp DOCX: {e}")
            
    # 3. Thử nghiệm với URL (Sẽ bỏ qua nếu không có mạng hoặc lỗi)
    test_url = "https://example.com"
    print(f"\n3. Thử nghiệm nạp URL: {test_url}")
    try:
        url_docs = load_source(test_url)
        analyze_documents(url_docs)
    except Exception as e:
        print(f"Lỗi chạy thử nghiệm URL (đúng hành vi khi chạy offline): {e}")
