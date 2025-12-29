import os
from bs4 import BeautifulSoup # این کتابخونه رو باید داشته باشی: pip install beautifulsoup4
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from sentence_transformers import SentenceTransformer

# تنظیمات اولیه
encoder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
client = QdrantClient(url="http://localhost:6333")
COLLECTION_NAME = "shia_ai_corpus"

def clean_html(html_content):
    """این همون گام اوله: تمیز کردن کدهای اضافه"""
    soup = BeautifulSoup(html_content, "html.parser")
    # حذف اسکریپت‌ها و استایل‌ها
    for script_or_style in soup(["script", "style"]):
        script_or_style.decompose()
    return soup.get_text(separator=' ', strip=True)

def ingest_folder(folder_path, source_type):
    if not os.path.exists(folder_path):
        print(f"❌ پوشه {folder_path} پیدا نشد!")
        return

    for filename in os.listdir(folder_path):
        if filename.endswith(".htm") or filename.endswith(".html"):
            path = os.path.join(folder_path, filename)
            with open(path, 'r', encoding='utf-8') as f:
                raw_content = f.read()
                text_content = clean_html(raw_content)
                # این رو جایگزین اون خط chunks قبلی کن:
                overlap = 200 # ۲۰۰ کاراکتر همپوشانی برای اینکه هیچ حدیثی از وسط قطع نشه
                chunks = [text_content[i:i+1500] for i in range(0, len(text_content), 1500 - overlap)]
                
                points = []
                for i, chunk in enumerate(chunks):
                    if len(chunk) < 50: continue
                    vector = encoder.encode(chunk).tolist()
                    points.append(PointStruct(
                        id=hash(filename + str(i)) % (10**10),
                        vector=vector,
                        payload={"text": chunk, "source_type": source_type, "book": filename}
                    ))
                
                # --- اصلاح اینجاست: ارسال در بسته‌های ۱۰۰ تایی ---
                for j in range(0, len(points), 100):
                    batch = points[j:j+100]
                    client.upsert(collection_name=COLLECTION_NAME, points=batch)
                
                print(f"✅ فایل {filename} با موفقیت تزریق شد.")

if __name__ == "__main__":
    # اول کل دیتابیس قبلی رو پاک کن که داده‌های کثیف حذف بشن
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={"size": 384, "distance": "Cosine"}
    )
    # ۲. حالا هر سه تا پوشه رو با برچسب مخصوص خودشون وارد می‌کنیم
    print("🚀 شروع عملیات تزریق داده‌ها...")
    
    ingest_folder("shia_source", "Shia")
    ingest_folder("sunni_source", "Sunni")
    ingest_folder("common_source", "Common") # این همون پوشه سوم که یادمون رفته بود
    
    print("✨ تموم شد! حالا دیتابیس مثل آینه تمیز و پر از اطلاعاته.")