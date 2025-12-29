import requests
import json
from db_manager import query_database 
import ollama

OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "deepseek-r1:8b" 

def generate_response(prompt: str) -> str:
    try:
        data = {"model": MODEL_NAME, "prompt": prompt, "stream": False}
        response = requests.post(OLLAMA_API_URL, json=data)
        return response.json().get("response", "No response received.")
    except Exception as e:
        return f"خطا در اتصال به اولاما: {e}"

def get_smart_queries(user_question):
    """اینجا همون مغز کارآگاهه که سوالات فرعی می‌سازه"""
    print("🔍 کارآگاه داره نقشه می‌کشه...")
    prompt = f"Generate 2 short search keywords in Persian related to: {user_question}. Just keywords, separated by comma."
    res = ollama.generate(model=MODEL_NAME, prompt=prompt)
    keywords = res['response'].strip().split(',')
    return [user_question] + [k.strip() for k in keywords]

def shia_ai_rag_query(user_question: str) -> str:
    # ۱. تولید سوالات هوشمند برای جستجوی عمیق‌تر
    queries = get_smart_queries(user_question)
    
    combined_context = ""
    print("--- 1. در حال شخم زدن دیتابیس با متد Agentic... ---")
    
    for q in queries:
        print(f"🔎 جستجو برای: {q}")
        combined_context += query_database(q) + "\n---\n"

    # ۲. دستورات لاتی و تخصصی تو (همون که فرستادی)
    system_instruction = """
    You are a master scholar, an expert in Islamic History, Hadith sciences, Imamate. 
    Your Tone: Street-Smart & Informal (Lati).
    Rule: Use ONLY the Context. If context is irrelevant, say: "داداش چیزی پیدا نکردم."
    Be aggressive and blunt against baseless claims.
    """
    
    full_prompt = f"SYSTEM:\n{system_instruction}\n\nCONTEXT:\n{combined_context}\n\nUSER QUESTION: {user_question}"
    
    print("--- 2. در حال تحلیل نهایی و پاتک زدن... ---")
    return generate_response(full_prompt)

if __name__ == "__main__":
    print("--- سیستم پاسخگویی (نسخه کارآگاه هوشمند) فعال شد ---")
    user_query = input("سوال رو بپرس رفیق: ")
    print(shia_ai_rag_query(user_query))