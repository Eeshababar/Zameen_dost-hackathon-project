# app.py — Smart Zameen Dost (Gradio + Urdu OCR + RAG)
# =============================================================================
# Deploy on Hugging Face Spaces (Gradio). Add keys in "Settings → Variables & secrets":
#   - ROBOFLOW_API_KEY
#   - WEATHER_API_KEY
#   - GROQ_API_KEY
#
# System deps (installed via packages.txt):
#   tesseract-ocr, tesseract-ocr-urd, poppler-utils, ffmpeg
# =============================================================================

import os
import io
import re
import json
import tempfile
from datetime import datetime, timedelta

import gradio as gr
import requests

# Whisper + TTS
import whisper
from gtts import gTTS

# Roboflow inference client
from inference_sdk import InferenceHTTPClient

# RAG stack
import PyPDF2
import nltk
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# OCR stack
from pdf2image import convert_from_bytes
import pytesseract

# Groq client
from groq import Groq


# ===========================
# 🔐 API keys (from Secrets)
# ===========================
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY", "").strip()
WEATHER_API_KEY  = os.getenv("WEATHER_API_KEY", "").strip()
GROQ_API_KEY     = os.getenv("GROQ_API_KEY", "").strip()

# Optional: point pytesseract to the binary if not on PATH (usually not needed on Spaces)
TESSERACT_CMD = os.getenv("TESSERACT_CMD", "").strip()
if TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

# ===========================
# 🔧 Clients / Endpoints
# ===========================
ROBOFLOW_API_URL = "https://serverless.roboflow.com"
WEATHER_URL_NOW  = "https://api.openweathermap.org/data/2.5/weather"
WEATHER_URL_3H   = "https://api.openweathermap.org/data/2.5/forecast"

rf_client = None
if ROBOFLOW_API_KEY:
    try:
        rf_client = InferenceHTTPClient(api_url=ROBOFLOW_API_URL, api_key=ROBOFLOW_API_KEY)
    except Exception as e:
        print(f"[WARN] Roboflow client init failed: {e}")

groq_client = None
if GROQ_API_KEY:
    try:
        groq_client = Groq(api_key=GROQ_API_KEY)
    except Exception as e:
        print(f"[WARN] Groq client init failed: {e}")

# ===========================
# 🔧 Global limits
# ===========================
K_RETRIEVE = 3
PER_DOC_CHARS = 700
MAX_CONTEXT_CHARS = 4000
MAX_OUTPUT_TOKENS = 512

# ===========================
# 🔇 NLTK (quiet)
# ===========================
try:
    nltk.download("punkt", quiet=True)
    nltk.download("stopwords", quiet=True)
except Exception:
    pass

# ===========================
# 🧩 Helpers
# ===========================
ARABIC_DIGITS_MAP = str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789")
ZERO_WIDTH_RE = re.compile(r"[\u200B-\u200F\u202A-\u202E]")

def safe_str(x) -> str:
    if x is None:
        return ""
    if isinstance(x, (bool, int, float)):
        return str(x)
    return str(x)

def normalize_mixed_text(text: str) -> str:
    s = safe_str(text)
    s = ZERO_WIDTH_RE.sub("", s)
    s = s.translate(ARABIC_DIGITS_MAP)
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"[^\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF\w\s.,;:!?()\-]", " ", s)
    return s

def _limit_chars(s: str, n: int) -> str:
    s = str(s or "")
    return s if len(s) <= n else (s[:n] + " …")

def _clip_context(snippets, max_chars: int) -> str:
    out, used = [], 0
    for snip in snippets:
        snip = str(snip or "")
        if used + len(snip) > max_chars:
            snip = snip[: max(0, max_chars - used)]
        if snip:
            out.append(snip)
            used += len(snip)
        if used >= max_chars:
            break
    return "\n\n".join(out)

def extract_numerical_data(text: str):
    t = safe_str(text)
    info = {}
    prices = re.findall(r"[$Rs\.]\s*(\d+(?:,\d{3})*(?:\.\d{2})?)", t)
    if prices:
        info["prices"] = prices
    perc = re.findall(r"(\d+(?:\.\d+)?)\s*%", t)
    if perc:
        info["percentages"] = perc
    yields = re.findall(
        r"(\d+(?:\.\d+)?)\s*(tons?|kg|quintals?|maunds?)\s*(?:per|/)?\s*(acre|hectare|ایکڑ)",
        t,
        re.IGNORECASE,
    )
    if yields:
        info["yields"] = yields
    return info

# ===========================
# 🌐 Urdu translation helper
# ===========================
def translate_weather(description: str) -> str:
    mapping = {
        "clear sky": "صاف آسمان",
        "few clouds": "ہلکے بادل",
        "scattered clouds": "چھٹپٹ بادل",
        "broken clouds": "ٹوٹے ہوئے بادل",
        "shower rain": "بوندا باندی",
        "rain": "بارش",
        "moderate rain": "درمیانی بارش",
        "light rain": "ہلکی بارش",
        "thunderstorm": "آندھی اور گرج چمک",
        "snow": "برفباری",
        "mist": "دھند",
        "overcast clouds": "مکمل بادل",
    }
    return mapping.get(safe_str(description).lower(), description)

# ===========================
# 🦠 Plant disease detection
# ===========================
def predict_disease(image_path: str):
    if not image_path:
        return "❌ براہ کرم پتے کی تصویر اپ لوڈ کریں"
    if not rf_client:
        return "❌ Roboflow API key سیٹ نہیں ہے"
    try:
        res = rf_client.infer(image_path, model_id="plant-disease-detection-v2-2nclk/1")
        preds = res.get("predictions", [])
        if not preds:
            return "❌ کوئی بیماری معلوم نہیں ہوئی"
        top = preds[0]
        cls = top.get("class", "نامعلوم")
        conf = float(top.get("confidence", 0.0)) * 100
        return f"🦠 بیماری: {cls}\nاعتماد: {conf:.2f}%"
    except Exception as e:
        return f"❌ خرابی: {e}"

# ===========================
# ☁️ Weather (Urdu)
# ===========================
def get_current_weather(city: str):
    city = (city or "").strip()
    if not city:
        return "❌ براہ کرم شہر کا نام درج کریں"
    if not WEATHER_API_KEY:
        return "❌ Weather API key سیٹ نہیں ہے"
    try:
        params = {"q": city, "appid": WEATHER_API_KEY, "units": "metric", "lang": "en"}
        r = requests.get(WEATHER_URL_NOW, params=params, timeout=20)
        data = r.json()
        if data.get("cod") != 200:
            return "❌ موسم کی معلومات حاصل نہیں ہو سکیں"
        desc_en   = data["weather"][0]["description"]
        desc_ur   = translate_weather(desc_en)
        temp      = data["main"]["temp"]
        feels     = data["main"]["feels_like"]
        humidity  = data["main"]["humidity"]
        wind      = data["wind"].get("speed", 0)
        return (
            f"🌤️ موجودہ موسم: {desc_ur}\n"
            f"🌡️ درجہ حرارت: {temp}°C (محسوس: {feels}°C)\n"
            f"💧 نمی: {humidity}%\n"
            f"💨 ہوا کی رفتار: {wind} m/s"
        )
    except Exception as e:
        return f"❌ موسم کی معلومات میں خرابی: {e}"

def get_tomorrow_forecast(city: str):
    city = (city or "").strip()
    if not city:
        return "❌ براہ کرم شہر کا نام درج کریں"
    if not WEATHER_API_KEY:
        return "❌ Weather API key سیٹ نہیں ہے"
    try:
        params = {"q": city, "appid": WEATHER_API_KEY, "units": "metric", "lang": "en"}
        r = requests.get(WEATHER_URL_3H, params=params, timeout=20)
        data = r.json()
        if data.get("cod") != "200":
            return "❌ شہر کا نام درست نہیں یا موسم کی معلومات دستیاب نہیں"
        tomorrow = (datetime.utcnow() + timedelta(days=1)).date()
        slots = []
        for entry in data.get("list", []):
            ts = datetime.utcfromtimestamp(entry["dt"])
            if ts.date() == tomorrow:
                desc_en = entry["weather"][0]["description"]
                slots.append({
                    "time": ts.strftime("%I:%M %p"),
                    "desc": translate_weather(desc_en),
                    "temp": entry["main"]["temp"],
                    "feels": entry["main"]["feels_like"],
                    "humidity": entry["main"]["humidity"]
                })
        if not slots:
            return "❌ کل کے لیے کوئی پیشگوئی دستیاب نہیں"
        out = [f"📅 کل ({tomorrow}) کے موسم کی پیشگوئی:\n"]
        for s in slots:
            out.append(
                f"🕒 {s['time']} — {s['desc']}\n"
                f"   🌡️ {s['temp']}°C (محسوس: {s['feels']}°C) | 💧 نمی: {s['humidity']}%"
            )
        return "\n".join(out)
    except Exception as e:
        return f"❌ خرابی: {e}"

# ===========================
# 🔗 Combined function
# ===========================
def advisory_app(image, city):
    disease = predict_disease(image)
    now     = get_current_weather(city)
    tomo    = get_tomorrow_forecast(city)
    return disease, now, tomo

# =========================
# 🖼️ OCR config
# =========================
OCR_LANGS = "urd+eng"
OCR_DPI = 300
MAX_OCR_PAGES = None

def ocr_pdf_bytes(pdf_content: bytes, dpi: int = OCR_DPI, langs: str = OCR_LANGS, max_pages=MAX_OCR_PAGES):
    images = convert_from_bytes(pdf_content, dpi=dpi)
    if max_pages is not None:
        images = images[:max_pages]
    out = []
    for i, img in enumerate(images, start=1):
        try:
            txt = pytesseract.image_to_string(img, lang=langs)
            txt = normalize_mixed_text(txt)
            out.append(f"\n--- OCR Page {i} ---\n{txt}\n" if txt else f"\n--- OCR Page {i} (no text) ---\n")
        except Exception as e:
            out.append(f"\n--- OCR Page {i} (error: {e}) ---\n")
    return "".join(out), len(images)

# =========================
# 📥 Google Drive PDF Processor (with OCR fallback)
# =========================
class GoogleDrivePDFProcessor:
    @staticmethod
    def convert_gdrive_link(share_link: str):
        patterns = [r"/file/d/([a-zA-Z0-9\-_]+)", r"id=([a-zA-Z0-9\-_]+)", r"/d/([a-zA-Z0-9\-_]+)"]
        file_id = None
        link = safe_str(share_link)
        for pat in patterns:
            m = re.search(pat, link)
            if m:
                file_id = m.group(1); break
        if not file_id:
            return None
        return f"https://drive.google.com/uc?export=download&id={file_id}"

    @staticmethod
    def download_pdf_from_gdrive(gdrive_link: str):
        try:
            download_link = GoogleDrivePDFProcessor.convert_gdrive_link(gdrive_link)
            if not download_link:
                return None, "Invalid Google Drive link format"
            headers = {"User-Agent": "Mozilla/5.0"}
            resp = requests.get(download_link, headers=headers, stream=True, timeout=60)
            txt = safe_str(resp.text)
            if ("confirm=" in txt) or ("virus scan warning" in txt.lower()):
                token = re.search(r"confirm=([^&]+)", txt)
                if token:
                    confirmed = f"{download_link}&confirm={token.group(1)}"
                    resp = requests.get(confirmed, headers=headers, stream=True, timeout=60)
            if resp.status_code == 200:
                return resp.content, "Success"
            return None, f"Download failed: HTTP {resp.status_code}"
        except Exception as e:
            return None, f"Download error: {e}"

    @staticmethod
    def extract_text_from_pdf(pdf_content: bytes):
        text = ""; pages = 0
        try:
            reader = PyPDF2.PdfReader(io.BytesIO(pdf_content))
            pages = len(reader.pages)
            out = []
            for i in range(pages):
                try:
                    pg = reader.pages[i]
                    t = normalize_mixed_text(pg.extract_text() or "")
                    out.append(f"\n--- Page {i+1} ---\n{t}\n" if t else f"\n--- Page {i+1} (no text) ---\n")
                except Exception:
                    out.append(f"\n--- Page {i+1} (Error extracting) ---\n")
            text = "".join(out)
        except Exception as e:
            text = f"PDF text extraction error: {e}"
            pages = 0

        core = re.sub(r"\s+", "", safe_str(text))
        if (pages == 0) or (len(core) < 100):
            try:
                ocr_text, ocr_pages = ocr_pdf_bytes(pdf_content, dpi=OCR_DPI, langs=OCR_LANGS, max_pages=MAX_OCR_PAGES)
                if len(re.sub(r"\s+", "", ocr_text)) >= 20:
                    return ocr_text, (ocr_pages or pages or 0)
            except Exception:
                pass
        return text, pages

# =========================
# 🧠 Knowledge Base / RAG
# =========================
class AdvancedPakistaniAgriRAG:
    def __init__(self, predefined_links=None):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        self.vector_store = None
        self.gdrive = GoogleDrivePDFProcessor()
        self.processed_documents = []
        self._setup_seed_knowledge()
        if predefined_links:
            self._auto_process_predefined_pdfs(predefined_links)

    def _setup_seed_knowledge(self):
        seed = [
            {
                "content": """Punjab Wheat Varieties for Export:
                اعلیٰ قسم کی گندم کی اقسام:
                - Anmol-91: Yield 45-50 maunds/acre, Export price $320-350/ton
                - Faisalabad-2008: High protein 12-14%, Premium export variety
                - Galaxy-2013: Disease resistant, Suitable for UAE market
                - Punjab-2011: Good for bread making, Export to Afghanistan
                اردو: یہ اقسام برآمد کے لیے بہترین ہیں اور زیادہ قیمت ملتی ہے""",
                "metadata": {"type": "crop_varieties", "region": "Punjab", "crop": "wheat", "language": "mixed"},
            },
            {
                "content": """Rice Export Opportunities - چاول کی برآمدات:
                Basmati Varieties with International Prices:
                - Super Basmati: 1000-1300/ton
                - IRRI-6: 700-850/ton (Middle East markets)
                Export Requirements:
                - Moisture: Maximum 14%
                - Broken grains: Less than 5%
                - Length: Minimum 6.0mm for Basmati
                اردو میں: بسمتی چاول کی برآمد سب سے زیادہ منافع بخش ہے""",
                "metadata": {"type": "export_markets", "crop": "rice", "price_range": "450-1300", "language": "mixed"},
            },
            {
                "content": """Government Support Schemes - حکومتی اسکیمز:
                Kisan Card Program:
                - 25% subsidy on fertilizers
                - 20% discount on certified seeds
                - Easy loan access through banks
                Solar Tube Well Scheme:
                - 60% government subsidy
                - Remaining 40% through easy installments
                - Electricity bill savings: Rs. 50,000+ annually
                Crop Insurance Program:
                - Premium: 5% of sum insured
                - Government pays 75% of premium
                - Coverage: Natural disasters, pest attacks
                کسان ڈویلپمنٹ پروگرام سے مفت تربیت اور مشورے""",
                "metadata": {"type": "government_schemes", "schemes": "kisan_card,solar_tubewell,crop_insurance", "language": "mixed"},
            },
        ]

        docs = []
        for item in seed:
            content = normalize_mixed_text(item["content"])
            meta = dict(item.get("metadata") or {})
            nums = extract_numerical_data(content)
            if nums:
                meta.update(nums)
            docs.append(Document(page_content=content, metadata=meta))

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800, chunk_overlap=100,
            separators=["\n\n", "\n", "۔", ".", ":", ";", " "],
            length_function=len,
        )
        pieces = splitter.split_documents(docs)
        self.vector_store = FAISS.from_documents(pieces, self.embeddings)
        print("✅ Seed agricultural knowledge initialized with", len(pieces), "chunks.")

    def _auto_process_predefined_pdfs(self, links):
        print(f"🚀 Auto-processing {len(links)} Google Drive PDF(s)...")
        ok = 0
        for i, link in enumerate(links, start=1):
            try:
                blob, msg = self.gdrive.download_pdf_from_gdrive(link)
                if blob is None:
                    print(f"❌ Doc {i}: {msg}")
                    self.processed_documents.append({"id": i, "pages": 0, "chunks": 0, "source": link, "status": msg})
                    continue
                text, pages = self.gdrive.extract_text_from_pdf(blob)
                if "pdf text extraction error" in safe_str(text).lower():
                    print(f"❌ Doc {i}: {text}")
                    self.processed_documents.append({"id": i, "pages": 0, "chunks": 0, "source": link, "status": text})
                    continue
                if len(safe_str(text).strip()) < 100:
                    print(f"ℹ️ Doc {i}: low native text — OCR likely used.")
                processed = normalize_mixed_text(text)
                numbers = extract_numerical_data(processed)
                doc = Document(
                    page_content=processed,
                    metadata={
                        "type": "auto_processed_pdf",
                        "source": f"Auto PDF {i}",
                        "pages": pages,
                        "numerical_data": numbers,
                        "processing_date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                        "original_link": link[:50] + "..." if len(link) > 50 else link,
                    },
                )
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=800, chunk_overlap=100,
                    separators=["\n\n", "\n", "۔", ".", ":", ";", " "]
                )
                chunks = splitter.split_documents([doc])
                if self.vector_store:
                    self.vector_store.add_documents(chunks)
                else:
                    self.vector_store = FAISS.from_documents(chunks, self.embeddings)
                self.processed_documents.append(
                    {"id": i, "pages": pages, "chunks": len(chunks), "source": doc.metadata["original_link"], "status": "✅ Success"}
                )
                print(f"✅ Doc {i}: {pages} pages → {len(chunks)} chunks")
                ok += 1
            except Exception as e:
                self.processed_documents.append(
                    {"id": i, "pages": 0, "chunks": 0, "source": link[:50] + "..." if len(link) > 50 else link, "status": f"❌ Error: {e}"}
                )
                print(f"❌ Doc {i}: {e}")
        print(f"🎉 Finished: {ok}/{len(links)} document(s) processed.")

    def get_stats_html(self) -> str:
        if not self.processed_documents:
            return "📊 Knowledge Base: Seed Pakistani agricultural data only (no PDFs yet)"
        total_chunks = sum(d.get("chunks", 0) for d in self.processed_documents)
        total_pages = sum(d.get("pages", 0) for d in self.processed_documents)
        return f"""📊 Knowledge Base Statistics:

🗂️ Auto-processed Documents: {len(self.processed_documents)}
📄 Total Pages Processed: {total_pages}
🧩 Total Text Chunks: {total_chunks}
📚 Seed Knowledge: Pakistani agriculture (Urdu + English)
🔍 Search Capability: Multilingual (English + Urdu)
✅ Status: Ready for queries
"""

    def get_relevant_info(self, query: str, k: int = K_RETRIEVE) -> str:
        if not self.vector_store:
            return "Knowledge base not available"
        try:
            q = normalize_mixed_text(query)
            hits = self.vector_store.similarity_search(q, k=k)
            snippets = []; nums_summary = []
            for i, doc in enumerate(hits, start=1):
                body = _limit_chars(doc.page_content, PER_DOC_CHARS)
                snippets.append(f"معلومات {i}: {body}")
                meta = doc.metadata or {}
                if not isinstance(meta, dict):
                    meta = {}
                nd = meta.get("numerical_data")
                if isinstance(nd, dict):
                    meta = {**meta, **nd}
                if isinstance(meta.get("prices"), list) and meta["prices"]:
                    nums_summary.append(f"💰 قیمتیں: {', '.join(map(safe_str, meta['prices']))}")
                if isinstance(meta.get("percentages"), list) and meta["percentages"]:
                    nums_summary.append(f"📊 فیصد: {', '.join(map(safe_str, meta['percentages']))}%")
                if isinstance(meta.get("yields"), list) and meta["yields"]:
                    y_fmt = []
                    for y in meta["yields"]:
                        try:
                            val, unit, per = y
                            y_fmt.append(f"{val} {unit} per {per}")
                        except Exception:
                            y_fmt.append(safe_str(y))
                    nums_summary.append(f"🌾 پیداوار: {', '.join(y_fmt)}")
            context = "\n\n".join(snippets)
            if nums_summary:
                context = "📈 اہم اعداد و شمار:\n" + "\n".join(nums_summary) + "\n\n" + context
            return _clip_context([context], MAX_CONTEXT_CHARS) or "No relevant information found."
        except Exception as e:
            return f"Error retrieving information: {e}"

# ===========================
# 🤖 Whisper model
# ===========================
print("🤖 Loading Whisper model...")
whisper_model = whisper.load_model("base")
print("✅ Whisper model loaded.")

# ===========================
# 📄 Optional seed PDFs (can be empty)
# ===========================
PREDEFINED_PDF_LINKS = []

# ===========================
# 🚀 Initialize RAG
# ===========================
print("🧠 Initializing Advanced Pakistani Agricultural Knowledge Base...")
pak_agri_rag = AdvancedPakistaniAgriRAG(predefined_links=PREDEFINED_PDF_LINKS)

# ===========================
# 🎙️ Voice, 🌦️ Weather, 🤝 AI
# ===========================
def voice_to_text(audio_file_path):
    if not audio_file_path:
        return ""
    try:
        result = whisper_model.transcribe(audio_file_path, language="ur")
        return normalize_mixed_text(result.get("text", ""))
    except Exception as e:
        return f"آواز سمجھ نہیں آئی: {e}"

def get_weather_with_farming_advice(city="Lahore"):
    try:
        city = safe_str(city).strip() or "Lahore"
        if not WEATHER_API_KEY:
            return "Weather API key سیٹ نہیں ہے"
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city},PK&appid={WEATHER_API_KEY}&units=metric"
        resp = requests.get(url, timeout=20)
        try:
            data = resp.json()
        except Exception:
            return "موسمی JSON درست نہیں۔"
        main = data.get("main") or {}
        wind = data.get("wind") or {}
        weather_l = data.get("weather") or [{}]
        temp = main.get("temp"); humidity = main.get("humidity")
        wind_speed = wind.get("speed"); description = weather_l[0].get("description", "")
        if any(v is None for v in (temp, humidity, wind_speed)):
            return "موسمی معلومات مکمل نہیں مل سکیں۔"
        if temp > 35:
            advice = f"⚠️ زیادہ گرمی ({temp}°C): صبح 6-8 بجے پانی دیں، دوپہر میں نہیں۔ پانی کی مقدار 20% بڑھائیں۔"
        elif humidity > 80:
            advice = f"🌧️ زیادہ نمی ({humidity}%): فنگیسائیڈ سپرے کریں۔ Mancozeb 2g/لیٹر یا Copper Oxychloride 3g/لیٹر۔"
        elif temp < 10:
            advice = f"❄️ سردی ({temp}°C): پودوں کو ڈھانپیں، پانی 50% کم دیں۔ Frost protection ضروری۔"
        elif wind_speed > 5:
            advice = f"💨 تیز ہوا ({wind_speed} m/s): کیڑے مار دوا کا سپرے نہ کریں۔ Wind barriers لگائیں۔"
        else:
            advice = f"✅ موسم اچھا ہے ({temp}°C, {humidity}% نمی): کھیتی کے کام کر سکتے ہیں۔"
        return f"آج {city} میں {temp}°C، نمی {humidity}%، ہوا {wind_speed} m/s، موسم {description}\n\n{advice}"
    except Exception as e:
        return f"موسمی معلومات نہیں مل سکیں: {e}"

def text_to_voice(text):
    try:
        clean = normalize_mixed_text(text)
        if len(clean) > 500:
            clean = clean[:500] + "... مکمل جواب اوپر پڑھیں"
        tts = gTTS(text=clean, lang="ur", slow=False)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            tts.save(tmp.name)
            return tmp.name
    except Exception as e:
        print(f"TTS Error: {e}")
        return None

def get_enhanced_ai_response(user_message: str, location: str = "") -> str:
    relevant_context = pak_agri_rag.get_relevant_info(user_message)
    system_prompt = (
        "You are Zameen Dost, a Pakistani agriculture advisor. "
        "Answer in simple Urdu, start with 'بھائی', use numbers when available, "
        "and keep it concise and actionable. If weather is included, integrate it. "
        "Only use the provided context; do not invent facts."
    )
    prompt_user = (
        f"Context:\n{relevant_context}\n\n"
        f"Location: {safe_str(location)}\n"
        f"Question: {safe_str(user_message)}"
    )
    if not groq_client:
        return "⚠️ GROQ_API_KEY سیٹ نہیں ہے، اس لیے AI جواب محدود ہے۔"
    try:
        chat = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt_user},
            ],
            model="llama-3.1-8b-instant",
            max_tokens=MAX_OUTPUT_TOKENS,
            temperature=0.5,
        )
        return chat.choices[0].message.content
    except Exception as e:
        msg = safe_str(e)
        if ("rate_limit" in msg) or ("tokens per minute" in msg) or ("Request too large" in msg):
            return "معذرت، پیغام بڑا تھا یا رفتار حد سے زیادہ تھی۔ براہِ کرم چھوٹا سوال کریں، یا دوبارہ کوشش کریں۔"
        return f"معذرت، AI سے رابطہ نہیں ہو سکا: {e}"

# ===========================
# 💬 Main chat handler
# ===========================
def zameen_dost_advanced_chat(audio_input, text_input, city_name, focus_area):
    user_message = ""; input_display = ""
    if audio_input:
        user_message = voice_to_text(audio_input)
        input_display = f"💬 آپ نے کہا: {user_message}"
    elif text_input:
        user_message = safe_str(text_input)
        input_display = f"💬 آپ نے لکھا: {user_message}"
    if not isinstance(user_message, str) or not user_message.strip():
        return "کرپیا کوئی سوال پوچھیں", None, "❌ کوئی سوال نہیں ملا"
    enhanced = user_message
    if focus_area and safe_str(focus_area) != "عام سوال":
        enhanced += f" (کسان کی دلچسپی: {focus_area})"
    terms = ["موسم", "بارش", "پانی", "weather", "irrigation", "spray", "سپرے"]
    if isinstance(user_message, str) and any(t in user_message for t in terms):
        weather_info = get_weather_with_farming_advice(city_name or "Lahore")
        enhanced += f"\n\nموسمی حالات: {weather_info}"
    ai_response = get_enhanced_ai_response(enhanced, city_name or "")
    voice_response = text_to_voice(ai_response)
    return input_display, voice_response, ai_response

# =========================
# 🖥️ Professional Gradio UI
# =========================
CUSTOM_CSS = """
:root {
  --brand:#2E8B57; --brand-2:#1f6f45; --bg:#f6faf9; --card:#ffffff; --muted:#6b7280; --ring: rgba(46,139,87,0.2);
}
.gradio-container {
  background: radial-gradient(1200px 600px at 10% -10%, #eaf7f0 0%, transparent 50%), var(--bg);
  font-family: ui-sans-serif, system-ui, -apple-system, "Noto Nastaliq Urdu", "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
  color: #0f172a;
}
a { color: var(--brand); }
.header {
  background: linear-gradient(135deg, #ffffff 0%, #f0faf5 100%);
  border: 1px solid #e6f2ea; border-left: 4px solid var(--brand);
  border-radius: 14px; padding: 18px 20px; box-shadow: 0 2px 10px rgba(16,24,40,0.04);
  margin: 8px 0 18px;
}
.header h1 { display:flex; align-items:center; gap:12px; margin:0; font-size: 26px; }
.header .tag { background: #eaf6f0; color: var(--brand-2); border-radius: 999px; padding: 4px 10px; font-size: 12px; }
.header p { margin: 6px 0 0; color: var(--muted); }
.card {
  background: var(--card); border: 1px solid #eef3ef; border-radius: 14px;
  box-shadow: 0 2px 12px rgba(15,23,42,0.04); padding: 16px; margin-bottom: 14px;
}
.card h3 { margin: 0 0 10px; font-size: 16px; color: #0f172a; }
.card .hint { color: var(--muted); font-size: 13px; }
button, .gr-button-primary { border-radius: 10px !important; box-shadow: 0 1px 2px rgba(0,0,0,0.04); }
button.primary, .gr-button-primary { background: var(--brand) !important; border: 1px solid #1f6f45 !important; }
button.primary:hover, .gr-button-primary:hover { background: var(--brand-2) !important; }
input:focus, textarea:focus, .gr-textbox:focus-within, .gr-textbox textarea:focus { outline: none !important; box-shadow: 0 0 0 4px var(--ring) !important; }
.footer { color: var(--muted); font-size: 12px; text-align: center; margin-top: 8px; }
"""

def _kb_stats_html():
    try:
        return pak_agri_rag.get_stats_html()
    except Exception:
        return "📊 Knowledge Base: (not initialized in this session)"

with gr.Blocks(
    title="Smart Zameen Dost - زمین دوست",
    theme=gr.themes.Soft(primary_hue="green", neutral_hue="gray"),
    css=CUSTOM_CSS
) as app:
    gr.HTML("""
        <div class="header">
          <h1>🌾 زمین دوست <span class="tag">اردو + آر اے جی + او سی آر</span></h1>
          <p>پاکستانی کسانوں کے لیے: ذہین مشورہ، بیماری کی تشخیص، اور موسم کی پیشگوئی</p>
        </div>
    """)

    with gr.Tabs():
        with gr.TabItem("چیٹ اسسٹنٹ"):
            with gr.Row():
                with gr.Column(scale=6):
                    with gr.Group(elem_classes=["card"]):
                        gr.Markdown("### اپنا سوال کریں")
                        audio_input = gr.Audio(sources=["microphone"], type="filepath", label="آواز میں پوچھیں", interactive=True)
                        text_input = gr.Textbox(label="یا یہاں لکھیں (اردو / English)", placeholder="مثال: برآمدی گندم کی اقسام کیا ہیں؟", lines=3)
                        with gr.Row():
                            city_input = gr.Textbox(label="آپ کا شہر", placeholder="Lahore, Karachi, Faisalabad", value="Lahore")
                            focus_area = gr.Dropdown(
                                label="دلچسپی کا شعبہ",
                                choices=["عام سوال","برآمدی فصلیں","گندم کی کاشت","چاول کی کاشت","کپاس کی کاشت","سبزیوں کی کاشت","پھلوں کی کاشت","کھاد اور بیج","بیماریوں کا علاج","حکومتی اسکیمز","منڈی کی قیمتیں"],
                                value="عام سوال",
                            )
                        chat_btn = gr.Button("جواب حاصل کریں", variant="primary", elem_classes=["primary"])
                with gr.Column(scale=6):
                    with gr.Group(elem_classes=["card"]):
                        gr.Markdown("### ذہین جواب")
                        input_display = gr.Textbox(label="آپ کا سوال", lines=2, interactive=False)
                        audio_output = gr.Audio(label="آواز میں جواب")
                        text_output = gr.Textbox(label="تفصیلی جواب", lines=12, interactive=False, show_copy_button=True)

            with gr.Row():
                with gr.Column():
                    gr.HTML(value=_kb_stats_html(), elem_classes=["card"])

            chat_btn.click(zameen_dost_advanced_chat, inputs=[audio_input, text_input, city_input, focus_area], outputs=[input_display, audio_output, text_output])

        with gr.TabItem("بیماری کی شناخت"):
            with gr.Row():
                with gr.Column(scale=6):
                    with gr.Group(elem_classes=["card"]):
                        gr.Markdown("### تصویر اپ لوڈ کریں")
                        image_input = gr.Image(label="پتے/پودے کی تصویر", type="filepath", height=280)
                        run_disease = gr.Button("🔎 بیماری معلوم کریں", variant="primary", elem_classes=["primary"])
                        gr.Markdown('<div class="hint">📌 واضح، قریب سے لی گئی تصویر بہتر نتائج دیتی ہے</div>')
                with gr.Column(scale=6):
                    with gr.Group(elem_classes=["card"]):
                        gr.Markdown("### نتیجہ")
                        disease_out = gr.Textbox(label="بیماری کی پیشگوئی", lines=6, interactive=False)

            run_disease.click(lambda img: predict_disease(img), inputs=[image_input], outputs=[disease_out])

        with gr.TabItem("موسم"):
            with gr.Row():
                with gr.Column(scale=6):
                    with gr.Group(elem_classes=["card"]):
                        gr.Markdown("### شہر منتخب کریں")
                        city_input2 = gr.Textbox(label="آپ کا شہر", placeholder="Lahore, Karachi, Faisalabad", value="Lahore")
                        with gr.Row():
                            run_now = gr.Button("موجودہ موسم", variant="primary", elem_classes=["primary"])
                            run_tomo = gr.Button("کل کی پیشگوئی", variant="secondary")
                with gr.Column(scale=6):
                    with gr.Group(elem_classes=["card"]):
                        gr.Markdown("### موسمی تفصیل")
                        now_out = gr.Textbox(label="موجودہ موسم", lines=6, interactive=False)
                        tomo_out = gr.Textbox(label="کل کی پیشگوئی", lines=10, interactive=False)

            run_now.click(lambda c: get_current_weather(c), inputs=[city_input2], outputs=[now_out])
            run_tomo.click(lambda c: get_tomorrow_forecast(c), inputs=[city_input2], outputs=[tomo_out])

    gr.HTML('<div class="footer">© Zameen Dost • Built for Pakistani farmers • Urdu-first UX</div>')

# Spaces will run the script, but launching locally is fine too.
if __name__ == "__main__":
    app.launch()
