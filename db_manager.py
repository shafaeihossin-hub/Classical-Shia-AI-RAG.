from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

# لود کردن مدل هوشمند برای درک معنای جملات
encoder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# اتصال به دیتابیس محلی
client = QdrantClient(url="http://localhost:6333")
COLLECTION_NAME = "shia_ai_corpus"

def query_database(user_query: str):
    try:
        # ۱. تبدیل سوال کاربر به وکتور
        query_vector = encoder.encode(user_query).tolist()

        # ۲. استفاده از متد جدید و قدرتمند query_points
        response = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=20  # گرفتن ۵ منبع برتر
        )

        # ۳. استخراج نتایج
        if not response.points:
            return ""

        context = ""
        for hit in response.points:
            p = hit.payload
            text = p.get("text", "")
            source = p.get("source_type", "Unknown")
            book = p.get("book", "Unknown")
            
            # چیدمان زیبا برای ارائه به هوش مصنوعی
            context += f"\n[منبع: {source} | کتاب: {book}]\nمتن: {text}\n---"
        
        return context

    except Exception as e:
        return f"خطای سیستمی در جستجو: {str(e)}"