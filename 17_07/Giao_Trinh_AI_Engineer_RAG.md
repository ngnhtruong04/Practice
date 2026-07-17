# GIÁO TRÌNH ĐÀO TẠO AI ENGINEER: XÂY DỰNG HỆ THỐNG RAG CHUYÊN NGHIỆP VỚI LANGCHAIN
## Lộ Trình Đào Tạo Hướng Dự Án Thực Tế (Project-Based Learning)

---

### TỔNG QUAN DỰ ÁN CUỐI KHÓA (MINI PROJECT)
*   **Bối cảnh thực tế:** Các lập trình viên (Developers) và kỹ sư kiểm thử (QA Engineers) trong doanh nghiệp phải tốn quá nhiều thời gian để tra cứu, tìm kiếm và đối chiếu thông tin trong tài liệu Yêu cầu Sản phẩm (Product Requirements Document - PRD) dài 50 trang định dạng PDF [2].
*   **Mục tiêu:** Xây dựng hệ thống **RAG (Retrieval-Augmented Generation) Hỏi Đáp Tài Liệu PRD** có khả năng trả lời chính xác các câu hỏi nghiệp vụ, triệt tiêu hoàn toàn hiện tượng ảo tưởng (hallucination) của LLM bằng cách chỉ trả lời dựa trên ngữ cảnh được truy xuất [3, 15].
*   **Công nghệ áp dụng:** LangChain, ChromaDB, OpenAI API, PyPDFLoader, Text Splitters, Embeddings Model, Streamlit/Gradio [3, 4].
*   **Tính năng cốt lõi:**
    1.  Hỗ trợ tải lên (Upload) file PDF qua giao diện Chat UI thân thiện [3, 4].
    2.  Trích xuất và phân tích cú pháp tài liệu tự động (Document Parsing & Loading) [3].
    3.  Chia nhỏ tài liệu một cách thông minh (Semantic & Character Chunking) [3, 10].
    4.  Tạo vector biểu diễn ngữ nghĩa (Embeddings Generation) [3, 11].
    5.  Lưu trữ vector và siêu dữ liệu (Metadata) vào cơ sở dữ liệu vector ChromaDB [3, 11].
    6.  Tìm kiếm tương đồng ngữ nghĩa (Semantic Search) cực kỳ chính xác [3, 4].
    7.  Giao diện trò chuyện hỏi đáp (Chat UI) hỗ trợ hiển thị trích dẫn nguồn (Source Citation) và số trang chi tiết trong tài liệu gốc (Page Number) [3, 4, 13].
    8.  Cơ chế kiểm soát an toàn nghiêm ngặt: Trả về **"Information not found"** nếu ngữ cảnh truy xuất không chứa đủ thông tin để trả lời câu hỏi [3].

---

### LỘ TRÌNH GIÁO TRÌNH BẮT BUỘC
*   **Chapter 1 — LangChain & RAG Fundamentals** (Tìm hiểu kiến trúc tổng quan hệ thống và RAG Workflow) [9].
*   **Chapter 2 — Document Loaders** (Xây dựng thành phần nạp dữ liệu đa định dạng, đầu ra là file `loader.py`) [9, 10].
*   **Chapter 3 — Chunking** (Thiết kế giải pháp chia cắt dữ liệu thông minh, đầu ra là file `splitter.py`) [10].
*   **Chapter 4 — Embeddings & Vector Database** (Chuyển đổi text thành vector và xây dựng CSDL ChromaDB, tạo file `embedding.py`, `vectordb.py`) [10, 11].
*   **Chapter 5 — Retriever Pipeline** (Thiết kế bộ lọc thông minh thu hồi chính xác thông tin, tạo file `retriever.py`) [11].
*   **Chapter 6 — Build Retriever Flow** (Gắn kết toàn bộ pipeline thành một luồng RetrievalQA hoàn chỉnh, tạo file `rag_pipeline.py`) [12].
*   **Chapter 7 — Context Synthesis** (Nâng cấp luồng tổng hợp ngữ cảnh nâng cao, tạo file `synthesis.py`) [12, 13].
*   **Chapter 8 — Full Q&A Chatbot** (Hoàn thiện giao diện ứng dụng chatbot hỏi đáp trực quan, tạo file `app.py`) [13].

---

## CHAPTER 1 — LANGCHAIN & RAG FUNDAMENTALS

### 1. GIỚI THIỆU
*   **Vấn đề cần giải quyết:** Các mô hình ngôn ngữ lớn (LLM) tuy rất thông minh nhưng chỉ có thể xử lý các tri thức được học trong quá trình huấn luyện (training) [17]. Chúng hoàn toàn không biết về các thông tin nội bộ của công ty, các tài liệu PDF mới cập nhật, cơ sở dữ liệu bảo mật riêng hay cụ thể là tài liệu PRD sản phẩm 50 trang của dự án hiện tại [17, 18].
*   **Tại sao cần học:** Chương này giúp người học hiểu được bản chất của kỹ thuật **RAG (Retrieval-Augmented Generation)** - giải pháp cầu nối đưa dữ liệu thực tế ngoài đời thực vào LLM mà không cần tinh chỉnh (fine-tuning) mô hình rất tốn kém [18, 19].
*   **Vị trí trong Mini Project:** Đây là bước xây dựng bản vẽ thiết kế hệ thống tổng thể. Bạn sẽ hiểu rõ bức tranh lớn và vai trò của từng mắt xích trước khi bắt tay vào lập trình từng module cụ thể trong các chương sau.
*   **Sau chương này hệ thống sẽ hoàn thiện thêm:** Sơ đồ kiến trúc luồng dữ liệu (Data Ingestion Pipeline) và Luồng truy vấn (Query Pipeline) hoàn chỉnh của hệ thống Hỏi đáp PRD.
*   **Kết quả đạt được:** Người học làm chủ tư duy hệ thống RAG, phân biệt được ứng dụng LLM truyền thống và hiện đại, hiểu rõ cách Document Object biểu diễn dữ liệu trong LangChain [30, 43].

### 2. KIẾN THỨC NỀN
*   **LangChain là gì?** Là một framework mã nguồn mở giúp xây dựng các ứng dụng tích hợp LLM bằng cách kết nối LLM với các nguồn dữ liệu ngoài, cơ sở dữ liệu, APIs và các công cụ bổ trợ [17].
*   **RAG (Retrieval-Augmented Generation) là gì?** Là kỹ thuật tối ưu hóa đầu ra của LLM, bằng cách truy vấn thông tin từ một nguồn tri thức đáng tin cậy bên ngoài (tri thức nghiệp vụ riêng) rồi đưa thông tin đó làm ngữ cảnh (context) kèm theo câu hỏi gốc của người dùng gửi cho LLM sinh câu trả lời [17, 21, 27].
*   **Document Object trong LangChain:** Là cấu trúc dữ liệu chuẩn hóa đại diện cho một khối văn bản, bao gồm hai thuộc tính cốt lõi: `page_content` (chứa nội dung chữ thuần túy) và `metadata` (từ điển chứa thông tin mô tả nguồn, số trang, tác giả...) [24, 25, 31, 33].

### 3. LÝ THUYẾT CHUYÊN SÂU
#### Kiến trúc ứng dụng LLM Truyền thống vs Hiện đại (RAG) [21, 22]

**Mô hình truyền thống:**
$$\text{User Question} \longrightarrow \text{LLM} \longrightarrow \text{Generic/Outdated Answer (Dễ bị ảo tưởng)}$$

**Mô hình hiện đại sử dụng RAG với LangChain:**
1.  **Giai đoạn Ingestion (Nạp dữ liệu):**
    $$\text{Tài liệu gốc (PDF/Word/Web)} \longrightarrow \text{Document Loader} \longrightarrow \text{Document Object} \longrightarrow \text{Chunking} \longrightarrow \text{Embedding} \longrightarrow \text{Vector DB} [22]$$
2.  **Giai đoạn Retrieval & Generation (Truy vấn & Sinh nội dung):**
    $$\text{User Question} \longrightarrow \text{Retriever (Tìm trong Vector DB)} \longrightarrow \text{Relevant Context (Các đoạn chứa nội dung liên quan nhất)} [22]$$
    $$\text{User Question} + \text{Relevant Context} \longrightarrow \text{System Prompt} \longrightarrow \text{LLM} \longrightarrow \text{Grounded Answer (Câu trả lời trích nguồn)}$$

#### Các thành phần chính của hệ thống RAG [25, 26, 27]:
1.  **Data Source:** Tài liệu thô (PRD, HR Policy, Wiki...) [25].
2.  **Document Loader:** Chuyển đổi dữ liệu thô thành Document Object tiêu chuẩn của LangChain [24, 26, 31].
3.  **Text Splitter (Chunking):** Cắt nhỏ tài liệu lớn thành các phần nhỏ (chunks) để vừa với cửa sổ ngữ cảnh (context window) của LLM [20, 26].
4.  **Embedding Model:** Chuyển đổi các đoạn văn bản (text chunks) thành vector số thực (numerical vectors) mang đậm ngữ nghĩa [21, 26].
5.  **Vector Database:** Lưu trữ vector kèm theo siêu dữ liệu (metadata) để hỗ trợ tìm kiếm siêu tốc [21, 26].
6.  **Retriever:** Thực hiện tìm kiếm độ tương đồng cosine (Cosine Similarity) để rút trích các đoạn ngữ cảnh phù hợp nhất dựa trên câu hỏi [21, 26].
7.  **LLM Generation:** Đọc câu hỏi và ngữ cảnh để đưa ra câu trả lời cuối cùng, triệt tiêu ảo tưởng [27].

*   **Ưu điểm của RAG:** Dữ liệu luôn cập nhật theo thời gian thực; kiểm soát được nguồn tri thức đầu vào; hỗ trợ trích dẫn nguồn (citation) giúp tăng độ tin cậy [21, 22, 35].
*   **Nhược điểm:** Tốc độ phản hồi phụ thuộc vào hiệu năng truy vấn vector và API của LLM; nếu bước retrieval lấy sai thông tin, LLM sẽ trả lời sai (Garbage In - Garbage Out).
*   **Best Practices:** Luôn giữ nguyên metadata gốc (source, page) từ bước Loader cho đến khi lưu vào Vector DB để phục vụ Citation [25, 43].
*   **Lỗi thường gặp & Cách Debug:** Mất Metadata trong quá trình xử lý chuỗi khiến Chatbot không thể hiển thị số trang [43]. Debug bằng cách ghi log kiểm tra cấu trúc của danh sách `Document` ở mỗi bước trung gian.

### 4. VÍ DỤ MINH HỌA
Dưới đây là ví dụ cơ bản minh họa cách khởi tạo một cấu trúc dữ liệu `Document Object` thủ công trong LangChain và cách trích xuất dữ liệu từ nó [40].

```python
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
```

### 5. THỰC HÀNH (PROJECT-BASED)

#### Exercise 1: Thiết kế Sơ đồ Luồng dữ liệu Hệ thống RAG Hỏi đáp PRD

##### Mục tiêu
Hiểu sâu sắc và tự thiết kế kiến trúc phân rã luồng xử lý dữ liệu (Data Ingestion Pipeline) cho dự án cuối khóa.

##### Kiến thức áp dụng
*   Kiến trúc tổng quan RAG Pipeline [25].
*   Vòng đời Document Object trong LangChain [38].

##### Vai trò trong Mini Project
Xây dựng tài liệu kiến trúc làm kim chỉ nam để lập trình chính xác các module `loader.py`, `splitter.py`, `embedding.py` và `app.py` ở các bài tiếp theo.

##### File
`docs/architecture_diagram.md`

##### Code
*(Mô tả sơ đồ kiến trúc hệ thống bằng định dạng văn bản Markdown Mermaid)*

```markdown
# Kiến Trúc Luồng Dữ Liệu RAG - Hệ Thống Hỏi Đáp PRD

## 1. Luồng Nạp Dữ Liệu (Data Ingestion Pipeline)
[PRD.pdf] ---> (PyPDFLoader) ---> [Document Objects (1 Page/Doc)]
                                             |
                                             v
[Document Objects] ---> (RecursiveCharacterTextSplitter) ---> [Document Chunks (500 chars)]
                                                                      |
                                                                      v
[Document Chunks] ---> (OpenAI Embeddings) ---> [Vector Embeddings (1536 dims)]
                                                                      |
                                                                      v
[Vector Embeddings] ---> (ChromaDB VectorStore) ---> [Disk / Memory Store]

## 2. Luồng Truy Vấn & Hỏi Đáp (Retrieval & Generation Pipeline)
[User Question] ---> (OpenAI Embeddings) ---> [Query Vector]
                                                   |
                                                   v
[Query Vector] ---> (ChromaDB Similarity Search) ---> [Top-K Relevant Chunks]
                                                             |
                                                             v
[User Question] + [Top-K Relevant Chunks (Metadata: Page, Source)] ---> [System Prompt]
                                                                             |
                                                                             v
                                                                      (OpenAI GPT-4o)
                                                                             |
                                                                             v
                                                                [Answer + Citation Source]
```

##### Hướng dẫn chạy
Sao chép đoạn code Markdown Mermaid trên vào bất kỳ trình đọc Markdown nào hỗ trợ Mermaid (như GitHub, VS Code) để hiển thị sơ đồ trực quan.

##### Kết quả mong đợi
Một sơ đồ thiết kế rõ ràng giúp hình dung chính xác cách dữ liệu PRD thô đi qua các bước chuyển đổi để trở thành tri thức mà chatbot có thể tìm kiếm và trích xuất.

##### Giải thích chi tiết
Sơ đồ mô tả 2 pha hoạt động độc lập của RAG: 
1.  **Pha Ingestion (Offline):** Chạy một lần duy nhất khi nạp tài liệu PRD vào hệ thống để điền đầy tri thức vào ChromaDB.
2.  **Pha Retrieval & Generation (Online):** Chạy mỗi khi người dùng gửi một câu hỏi lên giao diện chatbot.

##### Liên kết với bài tiếp theo
Chương tiếp theo (Chapter 2) sẽ hiện thực hóa hộp xử lý đầu tiên: **Document Loader** để đọc và parse các định dạng tài liệu PDF, DOCX, Web thành `Document Objects` [9, 10].

### 6. QUY TẮC KẾ THỪA
Sơ đồ kiến trúc thiết kế trong Chương 1 sẽ trực tiếp làm đặc tả kỹ thuật để lập trình cấu trúc dữ liệu cho `loader.py` ở Chương 2.

---

## CHAPTER 2 — DOCUMENT LOADERS

### 1. GIỚI THIỆU
*   **Vấn đề cần giải quyết:** Tri thức thực tế của doanh nghiệp nằm rải rác dưới rất nhiều định dạng khác nhau: tài liệu PRD dạng PDF, chính sách nhân sự dạng Word (.docx), hay tài liệu kỹ thuật trên các trang web nội bộ [23, 26, 56]. Hệ thống RAG cần một bộ lọc vạn năng để biến tất cả các tài liệu thô này thành một định dạng đồng nhất [31, 32].
*   **Tại sao cần học:** Giúp AI Engineer nắm vững cơ chế chuyển đổi dữ liệu không cấu trúc thành cấu trúc dữ liệu chuẩn hóa của LangChain [24, 31, 45].
*   **Vị trí trong Mini Project:** Đây là module đầu tiên trong Data Ingestion Pipeline của dự án cuối khóa [46]. Nếu không nạp được dữ liệu thô, toàn bộ các bước sau như sinh embedding hay truy vấn đều không thể thực hiện.
*   **Sau chương này hệ thống sẽ hoàn thiện thêm:** Hoàn thành module `loader.py` tự động phát hiện loại tài liệu và trích xuất nội dung chữ kèm siêu dữ liệu đầy đủ [10].
*   **Kết quả đạt được:** Biết cách viết code sử dụng `PyPDFLoader`, `Docx2txtLoader`, và `WebBaseLoader`; phân biệt và biết khi nào nên dùng `load()` vs `lazy_load()` để tối ưu hóa bộ nhớ RAM [48, 50, 51, 53, 54].

### 2. KIẾN THỨC NỀN
*   **Document Loader:** Thành phần đọc dữ liệu từ nguồn ngoài và xuất ra danh sách các `Document Object` [24, 44].
*   **Lazy Loading:** Cơ chế nạp dữ liệu tuần tự theo kiểu "generator" (chỉ tải phần dữ liệu cần xử lý lên RAM tại một thời điểm), cực kỳ hữu ích khi xử lý các file dữ liệu khổng lồ (vài GB) nhằm tránh tràn bộ nhớ RAM (Out Of Memory) [48, 50].

### 3. LÝ THUYẾT CHUYÊN SÂU
#### Khảo sát các loại Document Loader thông dụng:
1.  **PyPDFLoader (Thư viện hạt nhân `pypdf`):**
    *   *Cách hoạt động:* Đọc cấu trúc nhị phân của tệp PDF, chia tài liệu theo từng trang vật lý [48, 67].
    *   *Ưu điểm:* Tự động gán metadata số trang (`page`) vào từng Document Object, hỗ trợ đắc lực cho việc trích dẫn nguồn (Citation) [51, 52, 67].
    *   *Nhược điểm:* Thất bại hoàn toàn với các file PDF dạng ảnh quét (scanned PDF) [52, 70].
2.  **Docx2txtLoader (Thư viện hạt nhân `docx2txt`):**
    *   *Cách hoạt động:* Giải nén cấu trúc file DOCX (bản chất là một file ZIP chứa XML) và trích xuất dòng văn bản thuần [75, 76, 77].
    *   *Ưu điểm:* Trích xuất nhanh, giữ được trật tự dòng văn bản [77].
    *   *Nhược điểm:* Thông thường toàn bộ file Word chỉ được nạp thành **một Document duy nhất** thay vì chia theo trang như PDF; cấu trúc bảng biểu và hình ảnh có thể bị mất định dạng [54, 78, 83].
3.  **WebBaseLoader (Thư viện hạt nhân `BeautifulSoup`):**
    *   *Cách hoạt động:* Gửi yêu cầu HTTP GET đến URL, nhận về HTML thô và dùng BeautifulSoup để lọc bỏ các thẻ tags, chỉ giữ lại văn bản thuần [55, 93, 94].
    *   *Ưu điểm:* Giúp hệ thống RAG nhanh chóng tiếp cận kho tri thức trực tuyến [55, 91].
    *   *Nhược điểm:* Chứa nhiều dữ liệu nhiễu (Quảng cáo, Header, Footer) và không thể xử lý các website render động bằng JavaScript (Single Page Applications - React/Vue) [55, 56, 99].

| Loader | Đầu vào | Thư viện đi kèm | Metadata mặc định | Trường hợp khuyên dùng |
| :--- | :--- | :--- | :--- | :--- |
| **PyPDFLoader** | Tệp `.pdf` | `pypdf` | `source`, `page` | Tài liệu đặc tả sản phẩm (PRD), báo cáo phân tích [51, 52, 56]. |
| **Docx2txtLoader** | Tệp `.docx` | `docx2txt` | `source` | Chính sách nhân sự, biên bản cuộc họp, hợp đồng [53, 56, 77]. |
| **WebBaseLoader** | Website URL | `beautifulsoup4` | `source` (URL) | Tài liệu hướng dẫn sử dụng (Documentation), bài viết blog [54, 55, 56]. |

#### Lỗi thường gặp & Cách Debug trong thực tế [29, 42]:
1.  **Lỗi PDF dạng ảnh quét (Scanned PDF):** Loader chạy không lỗi nhưng kết quả trả về là chuỗi rỗng (`page_content=""`) [29, 70].
    *   *Cách Debug:* Đếm số lượng ký tự của page_content, nếu bằng 0 thì chuyển hướng sử dụng giải pháp OCR (như Tesseract OCR, AWS Textract) trước khi nạp vào LangChain [29, 70].
2.  **Lỗi BadZipFile với tệp Word:** Người dùng đổi đuôi một file lỗi thành `.docx` rồi tải lên [85].
    *   *Cách Debug:* Bọc khối lệnh load trong khối `try-except BadZipFile` để bắt lỗi định dạng file không chuẩn và đưa ra cảnh báo thân thiện trên giao diện [85].

### 4. VÍ DỤ MINH HỌA
Ví dụ dưới đây thực hiện nạp đồng thời một trang tài liệu PDF và trích xuất thông tin trang của nó [28].

```python
from langchain_community.document_loaders import PyPDFLoader

# Khởi tạo loader cho tệp PDF mẫu
file_path = "data/sample.pdf"
loader = PyPDFLoader(file_path)

# Nạp toàn bộ trang tài liệu
documents = loader.load()

# Hiển thị thông tin trang đầu tiên
if documents:
    first_page = documents[0]
    print(f"Tổng số trang nạp được: {len(documents)}")
    print(f"Nội dung trang 1:\n{first_page.page_content[:200]}...")
    print(f"Siêu dữ liệu trang 1: {first_page.metadata}")
```

### 5. THỰC HÀNH (PROJECT-BASED)

#### Exercise 1: Xây dựng Bộ nạp tài liệu đa nguồn chuyên nghiệp (`loader.py`)

##### Mục tiêu
Viết một script Python hoàn chỉnh có khả năng tự động nhận diện định dạng đầu vào (PDF, DOCX hoặc URL) và sử dụng đúng Document Loader tương ứng để trích xuất dữ liệu, bảo đảm tính nhất quán của metadata [106, 107].

##### Kiến thức áp dụng
*   `PyPDFLoader`, `Docx2txtLoader`, `WebBaseLoader` [57].
*   Kỹ thuật chuẩn hóa metadata và xử lý ngoại lệ (Exception Handling) trong Python [115-116].

##### Vai trò trong Mini Project
Đây chính là module **Document Loader** (hộp thứ nhất) trong Data Ingestion Pipeline của dự án cuối khóa.

##### File
`loader.py`

##### Code
```python
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
            # Bổ sung thông tin phân biệt loại loader vào metadata
            for doc in docs:
                doc.metadata["loader_type"] = "PyPDFLoader"
            return docs
            
        elif source_type == "docx":
            loader = Docx2txtLoader(source_path)
            docs = loader.load()
            for doc in docs:
                doc.metadata["loader_type"] = "Docx2txtLoader"
                doc.metadata["page"] = 1  # DOCX không chia trang, mặc định gán trang 1
            return docs
            
        elif source_type == "url":
            loader = WebBaseLoader(source_path)
            docs = loader.load()
            for doc in docs:
                doc.metadata["loader_type"] = "WebBaseLoader"
                doc.metadata["page"] = 1  # Web không chia trang, mặc định gán trang 1
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
    # Mã chạy thử nghiệm để chứng minh tính năng
    print("=== Chạy thử nghiệm loader.py ===")
    
    # 1. Thử nghiệm với file PDF ảo (đã tạo sẵn trong thư mục học liệu)
    pdf_path = "data/sample.pdf"
    if os.path.exists(pdf_path):
        print(f"\n[TEST 1] Nạp tệp tin PDF: {pdf_path}")
        try:
            pdf_docs = load_source(pdf_path)
            analyze_documents(pdf_docs)
        except Exception as e:
            print(f"Lỗi: {e}")
            
    # 2. Thử nghiệm với file DOCX ảo (đã tạo sẵn trong thư mục học liệu)
    docx_path = "data/sample.docx"
    if os.path.exists(docx_path):
        print(f"\n[TEST 2] Nạp tệp tin DOCX: {docx_path}")
        try:
            docx_docs = load_source(docx_path)
            analyze_documents(docx_docs)
        except Exception as e:
            print(f"Lỗi: {e}")
```

##### Hướng dẫn chạy
1.  Đảm bảo đã cài đặt đầy đủ thư viện qua lệnh: `pip install langchain langchain-community pypdf docx2txt beautifulsoup4` [109].
2.  Chạy trực tiếp file python này: `python loader.py`

##### Kết quả mong đợi
Màn hình console hiển thị chi tiết số lượng document nạp được từ cả file PDF mẫu (chia làm nhiều trang rõ rệt kèm số trang) và file DOCX (nạp thành 1 document lớn, được gán metadata tùy chỉnh).

##### Giải thích chi tiết
*   Hàm `is_url` sử dụng bộ parse URL chuẩn của Python để xác định kiểu nguồn [111].
*   Hàm `load_source` đóng vai trò là một "Loader Factory" thu nhỏ, giúp tối giản hóa việc tương tác của các module phía sau [117].
*   Việc gán thủ công `"page": 1` cho tệp DOCX và Web giúp cấu trúc dữ liệu luôn đồng nhất ở các chương sau, tránh lỗi `KeyError` khi chatbot gọi `doc.metadata["page"]` để hiển thị Citation [43, 67].

##### Liên kết với bài tiếp theo
Bài sau (Chapter 3 - Chunking) sẽ import trực tiếp file `loader.py` này để nạp dữ liệu thô trước khi chuyển qua bước chia cắt văn bản [13].

### 6. QUY TẮC KẾ THỪA
Mã nguồn của `loader.py` được đóng gói kỹ lưỡng và sẽ được tái sử dụng nguyên vẹn bằng cú pháp `from loader import load_source` ở tất cả các chương tiếp theo của giáo trình [13].

---

## CHAPTER 3 — CHUNKING (TEXT SPLITTING)

### 1. GIỚI THIỆU
*   **Vấn đề cần giải quyết:** Sau khi đã load thành công file PRD 50 trang thành các Document Object chứa toàn bộ văn bản [2, 20]. Chúng ta gặp một rào cản lớn: các LLM đều có giới hạn về cửa sổ ngữ cảnh (Context Window) [20, 26]. Hơn nữa, việc gửi toàn bộ cuốn PRD 50 trang vào LLM cho mỗi câu hỏi vừa gây lãng phí chi phí token, vừa làm giảm độ chính xác của câu trả lời (hiện tượng LLM bị "bối rối" giữa hàng vạn thông tin không liên quan) [20].
*   **Tại sao cần học:** Giúp học viên hiểu được tầm quan trọng của việc chia nhỏ tài liệu một cách khoa học [20, 26]. Học cách tìm điểm cân bằng (Trade-offs) giữa kích thước đoạn (Chunk Size) và độ gối đầu (Chunk Overlap) [10].
*   **Vị trí trong Mini Project:** Đây là bước tiền xử lý dữ liệu cực kỳ quan trọng trước khi sinh Embeddings và nạp vào cơ sở dữ liệu vector [22, 24].
*   **Sau chương này hệ thống sẽ hoàn thiện thêm:** Module `splitter.py` có khả năng băm nhỏ toàn bộ văn bản PRD dài thành hàng trăm đoạn ngữ cảnh nhỏ gọn, chất lượng cao, giữ nguyên ý nghĩa [10].
*   **Kết quả đạt được:** Nắm chắc lý thuyết và ứng dụng của các bộ chia `CharacterTextSplitter`, `RecursiveCharacterTextSplitter`, hiểu khái niệm `Semantic Chunking` và cách tính toán tối ưu các tham số [10].

### 2. KIẾN THỨC NỀN
*   **Chunk Size:** Độ dài tối đa (tính theo số ký tự hoặc token) của một đoạn văn bản sau khi chia nhỏ [10].
*   **Chunk Overlap:** Độ dài của phần văn bản được lặp lại giữa hai đoạn văn bản liền kề nhau [10].
*   **Hallucination (Sự ảo tưởng):** Hiện tượng LLM tự bịa ra thông tin không có trong thực tế do thiếu ngữ cảnh chính xác hoặc do ngữ cảnh đầu vào quá dài và loãng [3, 4].

### 3. LÝ THUYẾT CHUYÊN SÂU
#### Khảo sát các chiến lược chia nhỏ văn bản (Chunking Strategies) [10]:

1.  **CharacterTextSplitter:**
    *   *Cách hoạt động:* Chia cắt văn bản dựa trên một ký tự phân tách cố định được cấu hình trước (ví dụ: chia cứng theo ký tự xuống dòng song song `\n\n`) [10].
    *   *Đặc điểm:* Đơn giản nhưng rất dễ làm mất ngữ cảnh nếu đoạn văn bản giữa các ký tự phân tách quá dài [10].
2.  **RecursiveCharacterTextSplitter:**
    *   *Cách hoạt động:* Đây là bộ chia khuyên dùng mặc định cho hầu hết các tác vụ RAG văn bản [10]. Nó sử dụng một danh sách các ký tự phân tách theo thứ tự ưu tiên giảm dần: `["\n\n", "\n", " ", ""]` [10]. Nó sẽ cố gắng giữ các đoạn văn (ngắt bởi `\n\n`) cùng nhau, nếu đoạn văn vẫn quá lớn so với `Chunk Size`, nó sẽ ngắt theo dòng (`\n`), rồi đến ngắt theo từ (` `), và cuối cùng là ngắt theo ký tự (` `) [10].
    *   *Ưu điểm:* Giữ các đoạn văn mang tính logic và các câu văn nguyên vẹn nhất có thể, tránh việc một câu bị chặt đôi giữa chừng [10].
3.  **Semantic Chunking:**
    *   *Cách hoạt động:* Sử dụng mô hình Embedding để đo độ tương đồng ngữ nghĩa giữa các câu văn liên tiếp. Điểm cắt sẽ được đặt tại những nơi có sự thay đổi đột ngột về mặt ngữ nghĩa (độ tương đồng giảm mạnh) [10].
    *   *Ưu điểm:* Đoạn trích cực kỳ nhất quán về mặt nội dung.
    *   *Nhược điểm:* Tốn thời gian xử lý và chi phí API do phải gọi mô hình Embedding cho từng câu văn trong giai đoạn tiền xử lý [10].

#### Sự ảnh hưởng của Chunk Size và Chunk Overlap (Trade-offs) [10, 16]:

*   **Nếu Chunk Size quá nhỏ (ví dụ < 200 ký tự):**
    *   *Ưu điểm:* Tiết kiệm token gửi lên LLM, truy xuất cực nhanh.
    *   *Nhược điểm:* Ngữ cảnh bị xé vụn, mất đi mối liên kết thông tin giữa các ý trong văn bản. LLM không đủ thông tin để trả lời câu hỏi phức tạp.
*   **Nếu Chunk Size quá lớn (ví dụ > 2000 ký tự):**
    *   *Ưu điểm:* Ngữ cảnh đầy đủ, bảo đảm giữ nguyên ý của tác giả.
    *   *Nhược điểm:* Chi phí token tăng vọt; tốc độ phản hồi chậm; dễ kéo theo nhiều thông tin không liên quan làm loãng độ chính xác của câu trả lời từ LLM.
*   **Vai trò của Chunk Overlap:**
    *   Giúp duy trì tính liên tục của thông tin giữa các đoạn liền kề [10]. Nếu một thông tin quan trọng nằm ngay tại ranh giới cắt của bộ chia, việc có `Chunk Overlap` (thường khuyên dùng từ 10% - 20% kích thước `Chunk Size`) sẽ đảm bảo thông tin đó xuất hiện trọn vẹn ở ít nhất một chunk [10].

#### Lỗi thường gặp & Cách Debug [16]:
*   **Lỗi mất dấu định dạng Markdown/Bullet Points:** Khi chia nhỏ, các ký tự đánh dấu danh sách như `*`, `-` hoặc `1.` bị tách rời khỏi nội dung phía sau, làm LLM không hiểu đó là một danh sách liệt kê.
    *   *Cách Debug & Giải pháp:* Sử dụng `MarkdownHeaderTextSplitter` để chia nhỏ dựa trên các tiêu đề Heading (`#`, `##`, `###`) của file PRD trước khi áp dụng bộ chia ký tự.

### 4. VÍ DỤ MINH HỌA
Ví dụ dưới đây thể hiện sự khác biệt rõ rệt khi băm văn bản bằng `RecursiveCharacterTextSplitter`.

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

text = """LangChain là một framework tuyệt vời cho AI Engineer.
Nó giúp kết nối LLM với các nguồn dữ liệu ngoài.
Trong chương này chúng ta học về kỹ thuật chia nhỏ văn bản (Chunking).
Chia nhỏ văn bản giúp tối ưu hóa dung lượng ngữ cảnh gửi vào mô hình LLM."""

# Khởi tạo bộ chia với kích thước nhỏ để dễ quan sát hành vi
splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20
)

chunks = splitter.create_documents([text])
for idx, chunk in enumerate(chunks):
    print(f"Chunk #{idx+1} ({len(chunk.page_content)} ký tự): {repr(chunk.page_content)}")
```

### 5. THỰC HÀNH (PROJECT-BASED)

#### Exercise 2: Thiết kế bộ chia nhỏ tài liệu tối ưu (`splitter.py`)

##### Mục tiêu
Xây dựng module `splitter.py` kế thừa trực tiếp từ `loader.py` để lấy danh sách Document thô và băm nhỏ thành các Chunks tối ưu dựa trên chiến lược `RecursiveCharacterTextSplitter` [10, 13].

##### Kiến thức áp dụng
*   `RecursiveCharacterTextSplitter` và các tham số cấu hình [10].
*   Cơ chế kế thừa mã nguồn bằng cách `import` module tự viết [13].

##### Vai trò trong Mini Project
Đây chính là module **Chunking** (hộp thứ hai) trong Data Ingestion Pipeline của dự án cuối khóa [22].

##### File
`splitter.py`

##### Code
```python
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
# KẾ THỪA: Import trực tiếp bộ nạp tài liệu từ Chapter 2 [13]
from loader import load_source

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
    
    # Đường dẫn file test (Tái sử dụng file PDF đã nạp ở Chapter 2)
    pdf_path = "data/sample.pdf"
    
    if os.path.exists(pdf_path):
        print(f"Nạp dữ liệu từ {pdf_path} thông qua loader.py...")
        try:
            # Tái sử dụng hàm load_source của loader.py [13]
            raw_docs = load_source(pdf_path)
            
            # Thực hiện chia nhỏ bằng RecursiveCharacterTextSplitter (Size=300, Overlap=30)
            print("\nTiến hành băm nhỏ tài liệu:")
            chunks = split_documents(raw_docs, chunk_size=300, chunk_overlap=30, strategy="recursive")
            analyze_chunks(chunks)
            
        except Exception as e:
            print(f"Có lỗi xảy ra khi chia nhỏ tài liệu: {e}")
    else:
        print(f"Không tìm thấy file thử nghiệm tại '{pdf_path}'. Vui lòng chạy loader.py hoặc kiểm tra đường dẫn.")
```

##### Hướng dẫn chạy
1.  Đảm bảo file `loader.py` nằm cùng thư mục với `splitter.py` [13].
2.  Chạy chương trình: `python splitter.py`

##### Kết quả mong đợi
Toàn bộ văn bản từ file PDF mẫu được cắt thành các đoạn nhỏ dưới 300 ký tự. Các đoạn trích dẫn đều giữ nguyên thông tin metadata nguồn (`source`) và số trang gốc (`page`), bảo đảm không bị mất mát thông tin cấu trúc [43].

##### Giải thích chi tiết
*   Việc kế thừa `load_source` giúp lập trình viên không phải viết lại code đọc file phức tạp [13, 14].
*   Cơ chế chia của `RecursiveCharacterTextSplitter` giúp giữ cấu trúc câu không bị băm nát ở các từ đơn lẻ [10].
*   Hàm `analyze_chunks` cung cấp các số liệu thống kê trực quan (độ dài max, min, trung bình) giúp lập trình viên nhanh chóng đánh giá xem cấu hình `chunk_size` hiện tại có bị quá lớn hay quá nhỏ so với tài liệu PRD thực tế hay không [16].

##### Liên kết với bài tiếp theo
Bài sau (Chapter 4 - Embeddings & Vector Database) sẽ sử dụng trực tiếp hàm `split_documents` từ `splitter.py` này để chia nhỏ tài liệu PRD, sau đó nạp toàn bộ danh sách chunks thu được vào mô hình Embedding của OpenAI và lưu trữ vào ChromaDB [13, 14].

### 6. QUY TẮC KẾ THỪA
Chương sau (Chapter 4) sẽ gọi `from splitter import split_documents` để lấy trực tiếp luồng văn bản đã được băm nhỏ tối ưu [13].

---

## LỘ TRÌNH CHI TIẾT CÁC CHƯƠNG TIẾP THEO (CHAPTER 4 - 8)

Để hoàn thiện hệ thống RAG Hỏi Đáp PRD, các chương tiếp theo sẽ được xây dựng tuần tự theo nguyên tắc kế thừa nghiêm ngặt [14]:

### Chapter 4 — Embeddings & Vector Database [10, 11]
*   **Mục tiêu:** Chuyển hóa toàn bộ văn bản sau khi chia nhỏ từ `splitter.py` thành các vector đa chiều mang ngữ nghĩa và lưu trữ vào ChromaDB [11, 13, 14].
*   **Đầu ra:** File `embedding.py` và `vectordb.py`.
*   **Kế thừa:** Sử dụng dữ liệu chunks được sinh ra từ `splitter.py` [13, 14].

### Chapter 5 — Retriever Pipeline [11]
*   **Mục tiêu:** Xây dựng cơ chế tìm kiếm tương đồng ngữ nghĩa. Định cấu hình tìm kiếm top-K kết quả phù hợp nhất và thiết lập bộ lọc metadata [11].
*   **Đầu ra:** File `retriever.py`.
*   **Kế thừa:** Sử dụng database ChromaDB đã khởi tạo ở `vectordb.py` [14].

### Chapter 6 — Build Retriever Flow [12]
*   **Mục tiêu:** Sử dụng cấu trúc `RetrievalQA` của LangChain để ghép nối tất cả các mảnh ghép rời rạc trước đó thành một luồng xử lý thông tin tự động hoàn chỉnh [12].
*   **Đầu ra:** File `rag_pipeline.py`.
*   **Kế thừa:** Nhập và gọi các hàm từ `loader.py`, `splitter.py`, `vectordb.py`, và `retriever.py` [12, 13, 14].

### Chapter 7 — Context Synthesis [12, 13]
*   **Mục tiêu:** Nghiên cứu và tối ưu hóa các chiến lược tổng hợp ngữ cảnh phức tạp: Stuff Chain, Map-Reduce, Refine để đối phó với các câu hỏi đòi hỏi sự tổng hợp thông tin từ nhiều trang khác nhau trong PRD [12].
*   **Đầu ra:** File `synthesis.py`.

### Chapter 8 — Full Q&A Chatbot [13]
*   **Mục tiêu:** Thiết kế giao diện Chatbot bằng Streamlit hoặc Gradio hỗ trợ người dùng upload trực tiếp file PRD PDF, theo dõi trạng thái phân tích, đặt câu hỏi hỏi đáp và hiển thị câu trả lời kèm theo trích dẫn số trang và file nguồn cụ thể [13].
*   **Đầu ra:** File `app.py`.
