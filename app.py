import streamlit as st
import google.generativeai as genai
from PIL import Image
import io
import base64
import json
import os
import re
from pathlib import Path
import pdfplumber
from pdf2image import convert_from_path
import pickle
import warnings
from typing import Optional, Dict, Any
import sqlite3  # ← ADDED

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ============= CONFIGURATION =============
GEMINI_API_KEY = st.secrets["external_api"]["api_key"]
PDF_FOLDER = "./product_pdfs"
IMAGE_PDF_FOLDER = "./Product Images"
DATA_CACHE = "./data_cache.pkl"
IMAGE_CACHE = "./image_cache.pkl"
CONFIG_USERNAME = st.secrets["app_credentials"]["username"]
CONFIG_PASSWORD = st.secrets["app_credentials"]["password"]
LOGO_PATH = "./LOGO_Aquarius.png"
DB_PATH = "./chat_database.db"  # ← ADDED

GEMINI_MODELS = [
    "models/gemini-2.5-flash",
    "models/gemini-2.0-flash-exp",
    "models/gemini-flash-latest",
    "models/gemini-pro-latest",
]

IMAGE_DPI = 150
# =====================================================

# Page config
st.set_page_config(
    page_title="Aquarius Product Assistant",
    page_icon=LOGO_PATH,
    layout="wide",
    initial_sidebar_state="expanded"  # ← Changed to expanded for chat sessions
)

# Custom CSS (same as before)
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    .stApp {
        max-width: 1400px;
        margin: 0 auto;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        animation: slideIn 0.3s ease-out;
    }
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateY(10px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin-left: 20%;
    }
    .bot-message {
        background: white;
        border-left: 4px solid #667eea;
        margin-right: 20%;
    }
    .header-banner {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 1rem;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    .header {
        display: flex;
        align-items: center;
        gap: 15px;
        text-align: center;
        justify-content: center;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'username' not in st.session_state:
    st.session_state.username = None
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'product_database' not in st.session_state:
    st.session_state.product_database = None
if 'image_database' not in st.session_state:
    st.session_state.image_database = None
if 'models_index' not in st.session_state:
    st.session_state.models_index = {}
if 'chat_session' not in st.session_state:
    st.session_state.chat_session = None
if 'working_model' not in st.session_state:
    st.session_state.working_model = None
if 'current_session_id' not in st.session_state:  # ← ADDED
    st.session_state.current_session_id = None
if 'session_list' not in st.session_state:  # ← ADDED
    st.session_state.session_list = []

genai.configure(api_key=GEMINI_API_KEY)

# --- AUTHENTICATION FUNCTIONS ---
def authenticate_user(username: str, password: str) -> bool:
    if username == CONFIG_USERNAME and password == CONFIG_PASSWORD:
        st.session_state.authenticated = True
        st.session_state.username = username
        return True
    st.session_state.authenticated = False
    st.session_state.username = None
    return False

def login_form():
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.title("🔒 Login to Product Assistant")
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit_button = st.form_submit_button("Login")
            if submit_button:
                if authenticate_user(username, password):
                    st.success("Login successful!")
                    st.rerun()
                else:
                    st.error("Invalid username or password")
    st.stop()

def logout():
    for key in ['authenticated', 'username', 'messages', 'product_database', 'image_database', 'models_index', 'chat_session', 'current_session_id', 'session_list']:
        if key in st.session_state:
            del st.session_state[key]
    st.rerun()

# --- DATABASE FUNCTIONS FOR MULTI-CHAT SESSIONS ---

def init_database():
    """Initialize SQLite database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS chat_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            session_name TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            is_active BOOLEAN DEFAULT 1,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS chat_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            images TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES chat_sessions (id)
        )
    ''')
    
    conn.commit()
    conn.close()

def get_or_create_user(username):
    conn = sqlite3.connect(DB_PATH, timeout=10)
    cursor = conn.cursor()
    cursor.execute('SELECT id FROM users WHERE username = ?', (username,))
    result = cursor.fetchone()
    if result:
        user_id = result[0]
    else:
        cursor.execute('INSERT INTO users (username) VALUES (?)', (username,))
        user_id = cursor.lastrowid
        conn.commit()
    conn.close()
    return user_id

def create_new_chat_session(username, session_name=None):
    try:
        conn = sqlite3.connect(DB_PATH, timeout=10)
        cursor = conn.cursor()
        user_id = get_or_create_user(username)
        
        if not session_name:
            cursor.execute('SELECT COUNT(*) FROM chat_sessions WHERE user_id = ?', (user_id,))
            count = cursor.fetchone()[0]
            session_name = f"Chat {count + 1}"
        
        cursor.execute('''
            INSERT INTO chat_sessions (user_id, session_name)
            VALUES (?, ?)
        ''', (user_id, session_name))
        
        session_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return session_id
    except Exception as e:
        st.error(f"Error creating session: {e}")
        return None

def get_user_sessions(username):
    try:
        conn = sqlite3.connect(DB_PATH, timeout=10)
        cursor = conn.cursor()
        user_id = get_or_create_user(username)
        
        cursor.execute('''
            SELECT 
                s.id,
                s.session_name,
                s.created_at,
                s.last_updated,
                COUNT(m.id) as message_count
            FROM chat_sessions s
            LEFT JOIN chat_messages m ON s.id = m.session_id
            WHERE s.user_id = ? AND s.is_active = 1
            GROUP BY s.id
            ORDER BY s.last_updated DESC
        ''', (user_id,))
        
        sessions = []
        for row in cursor.fetchall():
            sessions.append({
                'id': row[0],
                'name': row[1],
                'created_at': row[2],
                'last_updated': row[3],
                'message_count': row[4]
            })
        conn.close()
        return sessions
    except Exception as e:
        st.warning(f"Could not load sessions: {e}")
        return []

def get_or_create_default_session(username):
    sessions = get_user_sessions(username)
    if sessions:
        return sessions[0]['id']
    else:
        return create_new_chat_session(username, "Chat 1")

def save_message_to_db(session_id, role, content, images=None):
    try:
        conn = sqlite3.connect(DB_PATH, timeout=10)
        cursor = conn.cursor()
        images_json = json.dumps(images) if images else None
        
        cursor.execute('''
            INSERT INTO chat_messages (session_id, role, content, images)
            VALUES (?, ?, ?, ?)
        ''', (session_id, role, content, images_json))
        
        cursor.execute('''
            UPDATE chat_sessions 
            SET last_updated = CURRENT_TIMESTAMP 
            WHERE id = ?
        ''', (session_id,))
        
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        st.error(f"Error saving message: {e}")
        return False

def load_session_messages(session_id, limit=100):
    try:
        conn = sqlite3.connect(DB_PATH, timeout=10)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT role, content, images
            FROM chat_messages
            WHERE session_id = ?
            ORDER BY timestamp ASC
            LIMIT ?
        ''', (session_id, limit))
        
        rows = cursor.fetchall()
        conn.close()
        
        messages = []
        for row in rows:
            message = {'role': row[0], 'content': row[1]}
            if row[2]:
                message['images'] = json.loads(row[2])
            messages.append(message)
        return messages
    except Exception as e:
        st.warning(f"Could not load messages: {e}")
        return []

def rename_session(session_id, new_name):
    try:
        conn = sqlite3.connect(DB_PATH, timeout=10)
        cursor = conn.cursor()
        cursor.execute('UPDATE chat_sessions SET session_name = ? WHERE id = ?', (new_name, session_id))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        st.error(f"Error renaming: {e}")
        return False

def delete_session_permanently(session_id):
    try:
        conn = sqlite3.connect(DB_PATH, timeout=10)
        cursor = conn.cursor()
        cursor.execute('DELETE FROM chat_messages WHERE session_id = ?', (session_id,))
        cursor.execute('DELETE FROM chat_sessions WHERE id = ?', (session_id,))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        st.error(f"Error deleting: {e}")
        return False

def auto_rename_session_from_first_message(session_id):
    try:
        conn = sqlite3.connect(DB_PATH, timeout=10)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT content FROM chat_messages 
            WHERE session_id = ? AND role = 'user'
            ORDER BY timestamp ASC LIMIT 1
        ''', (session_id,))
        result = cursor.fetchone()
        conn.close()
        
        if result:
            content = result[0]
            preview = content[:40] + "..." if len(content) > 40 else content
            rename_session(session_id, preview)
            return preview
    except:
        pass
    return None

# --- GEMINI INIT ---

def initialize_gemini():
    if st.session_state.working_model is not None and st.session_state.chat_session is not None:
        return
        
    WORKING_MODEL = None
    for model_name in GEMINI_MODELS:
        try:
            test_model = genai.GenerativeModel(model_name)
            test_model.generate_content("test")
            WORKING_MODEL = model_name
            break
        except:
            continue

    if WORKING_MODEL is None:
        st.error("❌ Could not find a working Gemini model.")
        st.stop()
        
    st.session_state.working_model = WORKING_MODEL
    
    try:
        model = genai.GenerativeModel(WORKING_MODEL)
        st.session_state.chat_session = model.start_chat(history=[])
    except Exception as e:
        st.error(f"Could not start chat: {e}")
        st.stop()

# --- PDF PROCESSING FUNCTIONS (same as before) ---

def extract_models_from_text(text):
    models = {}
    model_patterns = [
        r'\b([A-Z]+\s*\d+(?:\s*[A-Z]+)*)\b',
        r'\b(\d{3,4}\s*[A-Z]+(?:\s*[A-Z]+)*)\b',
    ]
    for pattern in model_patterns:
        matches = re.finditer(pattern, text)
        for match in matches:
            model_name = match.group(1).strip()
            start = max(0, match.start() - 500)
            end = min(len(text), match.end() + 500)
            context = text[start:end]
            if model_name not in models:
                models[model_name] = []
            models[model_name].append(context)
    return models

def process_pdf_comprehensive(pdf_path):
    pdf_data = {
        'filename': os.path.basename(pdf_path),
        'full_text': '',
        'pages': [],
        'models': {},
        'tables': []
    }
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                page_text = page.extract_text() or ""
                pdf_data['full_text'] += f"\n--- Page {page_num + 1} ---\n{page_text}"
                page_data = {
                    'page_num': page_num + 1,
                    'text': page_text,
                    'tables': []
                }
                tables = page.extract_tables()
                if tables:
                    for table in tables:
                        page_data['tables'].append(table)
                        pdf_data['tables'].append({'page': page_num + 1, 'data': table})
                pdf_data['pages'].append(page_data)
        pdf_data['models'] = extract_models_from_text(pdf_data['full_text'])
    except Exception as e:
        st.error(f"Error processing {pdf_path}: {str(e)}")
    return pdf_data

def load_image_database():
    if os.path.exists(IMAGE_CACHE):
        try:
            with open(IMAGE_CACHE, 'rb') as f:
                return pickle.load(f)
        except:
            pass
    
    if not os.path.exists(IMAGE_PDF_FOLDER):
        st.warning(f"Image folder not found: {IMAGE_PDF_FOLDER}")
        return {'images': [], 'categories': {}}
    
    all_images = []
    categories = {}
    category_folders = [f for f in Path(IMAGE_PDF_FOLDER).iterdir() if f.is_dir()]
    
    if not category_folders:
        st.warning(f"No category folders found")
        return {'images': [], 'categories': {}}
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    total_images = sum(len(list(cf.glob("*.jpg"))) + len(list(cf.glob("*.jpeg"))) + len(list(cf.glob("*.png"))) for cf in category_folders)
    
    if total_images == 0:
        st.warning(f"No images found")
        return {'images': [], 'categories': {}}
    
    processed = 0
    for category_folder in category_folders:
        category_name = category_folder.name
        categories[category_name] = []
        image_files = list(category_folder.glob("*.jpg")) + list(category_folder.glob("*.jpeg")) + list(category_folder.glob("*.png"))
        
        for img_path in image_files:
            status_text.text(f"Processing {category_name}/{img_path.name}...")
            try:
                with open(img_path, "rb") as f:
                    img_data = f.read()
                    img_base64 = base64.b64encode(img_data).decode()
                image_name = img_path.stem
                image_info = {
                    'category': category_name,
                    'name': image_name,
                    'data': img_base64,
                    'source': img_path.name,
                    'page': None
                }
                all_images.append(image_info)
                categories[category_name].append(image_info)
            except Exception as e:
                st.warning(f"Could not process {img_path}: {str(e)}")
            processed += 1
            progress_bar.progress(processed / total_images)
    
    image_data = {'images': all_images, 'categories': categories}
    try:
        with open(IMAGE_CACHE, 'wb') as f:
            pickle.dump(image_data, f)
    except Exception as e:
        st.warning(f"Could not save cache: {str(e)}")
    
    status_text.text(f"✅ {total_images} images loaded!")
    progress_bar.empty()
    return image_data

def load_or_process_pdfs():
    if os.path.exists(DATA_CACHE):
        try:
            with open(DATA_CACHE, 'rb') as f:
                return pickle.load(f)
        except:
            pass
    
    if not os.path.exists(PDF_FOLDER):
        st.error(f"PDF folder not found: {PDF_FOLDER}")
        return None
    
    pdf_files = list(Path(PDF_FOLDER).glob("*.pdf"))
    if not pdf_files:
        st.warning(f"No PDFs found")
        return None
    
    all_data = {'documents': [], 'full_text': '', 'models_index': {}}
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, pdf_path in enumerate(pdf_files):
        status_text.text(f"Processing {pdf_path.name}...")
        pdf_data = process_pdf_comprehensive(str(pdf_path))
        all_data['documents'].append(pdf_data)
        all_data['full_text'] += f"\n\n{'='*50}\nDOCUMENT: {pdf_data['filename']}\n{'='*50}\n{pdf_data['full_text']}"
        for model, contexts in pdf_data['models'].items():
            if model not in all_data['models_index']:
                all_data['models_index'][model] = []
            all_data['models_index'][model].extend([{'source': pdf_data['filename'], 'context': ctx} for ctx in contexts])
        progress_bar.progress((idx + 1) / len(pdf_files))
    
    try:
        with open(DATA_CACHE, 'wb') as f:
            pickle.dump(all_data, f)
    except Exception as e:
        st.warning(f"Could not save cache: {str(e)}")
    
    status_text.text("✅ PDFs processed!")
    progress_bar.empty()
    return all_data

def find_relevant_content(query, database, image_database):
    query_lower = query.lower()
    relevant_data = {'text_sections': [], 'models': [], 'images': [], 'tables': [], 'relevant_categories': set()}
    
    for model in database['models_index'].keys():
        if model.lower() in query_lower or query_lower in model.lower():
            relevant_data['models'].append({'model': model, 'data': database['models_index'][model]})
    
    words = [w for w in query_lower.split() if len(w) > 3]
    
    for doc in database['documents']:
        text = doc['full_text'].lower()
        for word in words:
            if word in text:
                idx = 0
                section_count = 0
                while section_count < 3:
                    idx = text.find(word, idx)
                    if idx == -1:
                        break
                    start = max(0, idx - 300)
                    end = min(len(doc['full_text']), idx + 300)
                    section = doc['full_text'][start:end]
                    relevant_data['text_sections'].append({'source': doc['filename'], 'content': section})
                    idx += 1
                    section_count += 1
                    if len(relevant_data['text_sections']) >= 10:
                        break
        
        for table_data in doc['tables']:
            table_str = str(table_data['data']).lower()
            if any(word in table_str for word in words):
                relevant_data['tables'].append(table_data)

    if image_database and 'images' in image_database:
        image_scores = []
        for img in image_database['images']:
            score = 0
            img_name_lower = img['name'].lower()
            img_category_lower = img['category'].lower()
            for word in words:
                if word in img_name_lower:
                    score += 5
                if word in img_category_lower:
                    score += 2
            for model_info in relevant_data['models']:
                model_lower = model_info['model'].lower()
                if model_lower in img_name_lower:
                    score += 10
                model_parts = model_lower.replace('-', ' ').replace('_', ' ').split()
                if any(part in img_name_lower for part in model_parts if len(part) > 2):
                    score += 7
            if score > 0:
                image_scores.append((score, img))
        
        image_scores.sort(key=lambda x: x[0], reverse=True)
        max_images = min(10, max(3, len(image_scores) // 2)) if image_scores else 0
        for score, img in image_scores[:max_images]:
            relevant_data['images'].append({
                'source': img['source'],
                'data': img['data'],
                'name': img['name'],
                'category': img['category']
            })
    
    return relevant_data

def extract_query_intent(query):
    """Extract key information from the query for better matching"""
    intent_info = {
        'query_type': 'general',
        'key_terms': [],
        'is_comparison': False,
        'is_specification': False,
        'asks_about_height': False,
        'asks_about_reach': False
    }
    
    query_lower = query.lower()
    
    # Detect query type
    if any(word in query_lower for word in ['compare', 'difference', 'vs', 'versus', 'better']):
        intent_info['query_type'] = 'comparison'
        intent_info['is_comparison'] = True
    elif any(word in query_lower for word in ['specification', 'specs', 'details', 'features', 'capacity']):
        intent_info['query_type'] = 'specification'
        intent_info['is_specification'] = True
    elif any(word in query_lower for word in ['recommend', 'suggest', 'best', 'which', 'should i']):
        intent_info['query_type'] = 'recommendation'
    elif any(word in query_lower for word in ['how', 'what is', 'explain', 'tell me about']):
        intent_info['query_type'] = 'explanation'
    
    # Detect specific specification queries
    if any(word in query_lower for word in ['height', 'tall', 'high']):
        intent_info['asks_about_height'] = True
    if any(word in query_lower for word in ['reach', 'vertical reach', 'max reach']):
        intent_info['asks_about_reach'] = True
    
    # Extract key terms
    common_words = {'what', 'which', 'where', 'when', 'about', 'tell', 'give', 'show', 'provide', 'need', 'want', 'like', 'have', 'this', 'that', 'with', 'from', 'they'}
    words = [word.strip('.,!?') for word in query_lower.split() if len(word) > 3 and word not in common_words]
    intent_info['key_terms'] = list(set(words))
    
    return intent_info

def create_comprehensive_prompt(query, relevant_data, database):
    # Extract query intent for better understanding
    intent = extract_query_intent(query)
    
    prompt = f"""You are an expert product assistant for Aquarius Engineers, a construction equipment company.
You have access to complete product catalogs and must provide COMPREHENSIVE, DETAILED answers.

========================================
PHASE 1: UNDERSTAND THE QUERY
========================================
User Question: "{query}"

Query Analysis:
- Query Type: {intent['query_type'].upper()}
- Key Terms: {', '.join(intent['key_terms'])}
- Is Comparison: {'YES' if intent['is_comparison'] else 'NO'}
- Is Specification Request: {'YES' if intent['is_specification'] else 'NO'}

CRITICAL SPECIFICATION GUIDELINES:
"""
    
    # Add specific guidelines based on query intent
    if intent['asks_about_height'] or intent['asks_about_reach']:
        prompt += """
⚠️ IMPORTANT - HEIGHT/REACH SPECIFICATION:
When asked about "height", "maximum height", "how tall", or "reach":
- ALWAYS refer to "VERTICAL REACH" or "MAX VERTICAL REACH" specification
- DO NOT confuse with "Overall Height", "Transport Height", "Folded Height", or "Machine Height"
- If vertical reach is not available, clearly state that and only then mention other height measurements
- Format: "Vertical Reach: [value] meters" or "Maximum Vertical Reach: [value] meters"

"""
    
    prompt += """
SPECIFICATION ACCURACY RULES:
1. **Capacity** - Always refer to "Working Capacity" or "Rated Capacity", not tank capacity or hopper capacity unless specifically asked
2. **Height** - ONLY use "Vertical Reach" for height questions, NOT overall dimensions
3. **Width** - Use "Working Width" for operational width, "Transport Width" for dimensions
4. **Weight** - Use "Operating Weight" for machine weight questions
5. **Power** - Use "Engine Power" or "Motor Power" for power questions
6. **Output** - Use "Output per Cycle" or "Production Capacity" for output questions

RESPONSE INSTRUCTIONS:
1. Match the EXACT specification being asked for
2. Provide COMPLETE information detailed, not summaries
3. Include ALL relevant specifications in clear tables
4. Format numbers with proper units (meters, kg, liters, etc.)
5. DO NOT mention page numbers, document sources, or say "according to the documentation"
6. Be thorough but ACCURATE - don't mix up different types of specifications
7. If unsure about which specification is being asked, provide the most relevant one clearly labeled
8. If asked about a model then give all the features and specification of that model from relevant pdf in detail

========================================
PHASE 2: RELEVANT PRODUCT DOCUMENTATION
========================================
"""
    
    # Add model-specific data
    if relevant_data['models']:
        prompt += f"\n📋 SPECIFIC MODEL DATA ({len(relevant_data['models'])} model(s) found):\n"
        prompt += "=" * 80 + "\n"
        for model_info in relevant_data['models']:
            prompt += f"\n🔹 Model: {model_info['model']}\n"
            prompt += "-" * 80 + "\n"
            for data in model_info['data'][:3]:
                prompt += f"{data['context']}\n\n"
    
    # Add relevant text sections
    if relevant_data['text_sections']:
        prompt += f"\n📄 RELEVANT DOCUMENTATION ({len(relevant_data['text_sections'])} section(s)):\n"
        prompt += "=" * 80 + "\n"
        for idx, section in enumerate(relevant_data['text_sections'][:15], 1):
            prompt += f"\n[Section {idx}]\n{section['content']}\n"
    
    # Add table data
    if relevant_data['tables']:
        prompt += f"\n📊 SPECIFICATION TABLES ({len(relevant_data['tables'])} table(s)):\n"
        prompt += "=" * 80 + "\n"
        for idx, table in enumerate(relevant_data['tables'][:5], 1):
            table_json = json.dumps(table['data'])
            prompt += f"\n[Table {idx}]\n{table_json}\n"
    
    # Add general context if needed
    if not relevant_data['models'] and not relevant_data['text_sections']:
        prompt += f"\n📚 GENERAL PRODUCT CATALOG:\n"
        prompt += "=" * 80 + "\n"
        prompt += f"{database['full_text'][:5000]}\n"
    
    prompt += """

========================================
PHASE 3: PROVIDE YOUR ANSWER
========================================

Now provide a comprehensive, accurate answer that:

✓ DIRECTLY answers what was asked (match the exact specification)
✓ Uses ONLY the correct specifications from documentation
✓ Presents data in clear, organized TABLES for specifications
✓ Includes ALL relevant technical details with proper units
✓ Is formatted professionally and easy to read

✗ DO NOT mention sources, page numbers, or documentation references
✗ DO NOT confuse different types of specifications (e.g., vertical reach vs overall height)
✗ DO NOT provide summaries when full details are available

"""
    
    # Add specific reminders based on query
    if intent['is_comparison']:
        prompt += "\n⚡ COMPARISON QUERY: Provide side-by-side comparison table with all key differences.\n"
    elif intent['is_specification']:
        prompt += "\n⚡ SPECIFICATION QUERY: Provide complete technical specifications in detailed table format.\n"
    if intent['asks_about_height'] or intent['asks_about_reach']:
        prompt += "\n⚡ HEIGHT QUERY: Remember to use VERTICAL REACH specification only!\n"
    
    prompt += "\nYour Professional Answer:\n"
    
    return prompt

def query_gemini_comprehensive(query, database, image_database):
    try:
        relevant_data = find_relevant_content(query, database, image_database)
        prompt = create_comprehensive_prompt(query, relevant_data, database)
        response = st.session_state.chat_session.send_message(prompt)
        return {
            'answer': response.text,
            'relevant_images': relevant_data['images'],
            'models_mentioned': [m['model'] for m in relevant_data['models']]
        }
    except Exception as e:
        return {'answer': f"Error: {str(e)}", 'relevant_images': [], 'models_mentioned': []}

# ==================== MAIN APP ====================

if not st.session_state.authenticated:
    login_form()

# Initialize database
init_database()

# Initialize Gemini
initialize_gemini()

# Initialize current session
if st.session_state.current_session_id is None:
    st.session_state.current_session_id = get_or_create_default_session(st.session_state.username)

# Load messages for current session
if st.session_state.messages == []:
    loaded_messages = load_session_messages(st.session_state.current_session_id)
    if loaded_messages:
        st.session_state.messages = loaded_messages

# Load logo
logo_html = "🏗️"
if os.path.exists(LOGO_PATH):
    try:
        with open(LOGO_PATH, "rb") as f:
            logo_data = base64.b64encode(f.read()).decode()
            logo_html = f'<img src="data:image/png;base64,{logo_data}" style="height: 80px;">'
    except:
        pass

# Header
st.markdown(f"""
<div class="header-banner">
    <span class="header">{logo_html}<h1>Aquarius Product Assistant</h1></span>
    <p>Your comprehensive guide to construction equipment</p>
</div>
""", unsafe_allow_html=True)

# Load databases
if st.session_state.product_database is None:
    if 'chat_session' in st.session_state:
        try:
            model = genai.GenerativeModel(st.session_state.working_model)
            st.session_state.chat_session = model.start_chat(history=[])
        except:
            pass
    with st.spinner("🔄 Loading database..."):
        st.session_state.product_database = load_or_process_pdfs()
        if st.session_state.product_database:
            st.session_state.models_index = st.session_state.product_database['models_index']

if st.session_state.image_database is None:
    with st.spinner("🔄 Loading images..."):
        st.session_state.image_database = load_image_database()

if st.session_state.product_database is None:
    st.error("⚠️ Cannot load database")
    st.stop()

# Display chat
for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(f'<div class="chat-message user-message">👤 <strong>You:</strong><br>{message["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="chat-message bot-message">🤖 <strong>Assistant:</strong><br>{message["content"]}</div>', unsafe_allow_html=True)
        if "images" in message and message["images"]:
            st.markdown("### 📸 Product Images")
            cols = st.columns(4)
            for idx, img_data in enumerate(message["images"]):
                with cols[idx % 4]:
                    img_bytes = base64.b64decode(img_data['data'])
                    caption = img_data.get('name', f"Page {img_data.get('page', '?')}")
                    st.image(img_bytes, caption=caption, use_container_width=True, output_format="PNG")

# Chat input
user_query = st.chat_input("Ask anything...")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    save_message_to_db(st.session_state.current_session_id, "user", user_query)
    
    # Auto-rename session after first message
    if len(st.session_state.messages) == 1:
        auto_rename_session_from_first_message(st.session_state.current_session_id)
    
    with st.spinner("🔍 Searching..."):
        result = query_gemini_comprehensive(user_query, st.session_state.product_database, st.session_state.image_database)
        assistant_message = {"role": "assistant", "content": result['answer']}
        if result['relevant_images']:
            assistant_message["images"] = result['relevant_images']
        st.session_state.messages.append(assistant_message)
        save_message_to_db(st.session_state.current_session_id, "assistant", result['answer'], result.get('relevant_images'))
    st.rerun()

# Sidebar
with st.sidebar:
    st.title("💬 Chat Sessions")
    
    # New Chat Button
    if st.button("➕ New Chat", use_container_width=True, type="primary"):
        new_session_id = create_new_chat_session(st.session_state.username)
        if new_session_id:
            st.session_state.current_session_id = new_session_id
            st.session_state.messages = []
            try:
                model = genai.GenerativeModel(st.session_state.working_model)
                st.session_state.chat_session = model.start_chat(history=[])
            except:
                pass
            st.rerun()
    
    st.markdown("---")
    
    # Load all sessions
    sessions = get_user_sessions(st.session_state.username)
    
    if not sessions:
        st.info("No chats yet. Start a new one!")
    else:
        st.markdown(f"**{len(sessions)} Chat(s)**")
        
        for session in sessions:
            is_current = (session['id'] == st.session_state.current_session_id)
            
            with st.container():
                col1, col2, col3 = st.columns([6, 3, 3])
                
                with col1:
                    button_label = f"{'🟢 ' if is_current else ''} {session['name']}"
                    if st.button(button_label, key=f"sess_{session['id']}", use_container_width=True, type="secondary" if is_current else "tertiary"):
                        st.session_state.current_session_id = session['id']
                        st.session_state.messages = load_session_messages(session['id'])
                        try:
                            model = genai.GenerativeModel(st.session_state.working_model)
                            st.session_state.chat_session = model.start_chat(history=[])
                        except:
                            pass
                        st.rerun()
                
                with col2:
                    if st.button("✏️", key=f"edit_{session['id']}"):
                        st.session_state[f'editing_{session["id"]}'] = True
                        st.rerun()
                
                with col3:
                    if st.button("🗑️", key=f"del_{session['id']}"):
                        if session['id'] == st.session_state.current_session_id:
                            st.warning("Cannot delete current session!")
                        else:
                            delete_session_permanently(session['id'])
                            st.rerun()
                
                # Rename form
                if st.session_state.get(f'editing_{session["id"]}', False):
                    with st.form(key=f"rename_{session['id']}"):
                        new_name = st.text_input("New name:", value=session['name'])
                        col_s, col_c = st.columns(2)
                        with col_s:
                            if st.form_submit_button("Save"):
                                rename_session(session['id'], new_name)
                                st.session_state[f'editing_{session["id"]}'] = False
                                st.rerun()
                        with col_c:
                            if st.form_submit_button("Cancel"):
                                st.session_state[f'editing_{session["id"]}'] = False
                                st.rerun()
                
                # st.caption(f"📊 {session['message_count']} messages")
                st.markdown("<hr style='margin: 5px 0; border: 0.5px solid #ddd;'>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Product Index
    st.title("📚 Product Index")
    
    if st.session_state.models_index:
        st.markdown(f"**{len(st.session_state.models_index)} Models**")
        
        categories = {}
        for model in st.session_state.models_index.keys():
            if 'CW' in model:
                cat = "Concrete Wash"
            elif 'SP' in model or 'Z' in model or 'MP' in model:
                cat = "Batching Plants"
            elif any(x in model for x in ['D', 'E', 'HP']):
                cat = "Pumps"
            else:
                cat = "Other"
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(model)
        
        for category, models in sorted(categories.items()):
            with st.expander(f"{category} ({len(models)})"):
                for model in sorted(models):
                    if st.button(model, key=f"model_{model}"):
                        query = f"Tell me everything about {model} including full specifications"
                        st.session_state.messages.append({"role": "user", "content": query})
                        save_message_to_db(st.session_state.current_session_id, "user", query)
                        st.rerun()
    
    st.markdown("---")
    
    # Admin buttons
    if st.button("🔄 Reload PDFs"):
        if os.path.exists(DATA_CACHE):
            os.remove(DATA_CACHE)
        st.session_state.product_database = None
        st.rerun()
    
    if st.button("🖼️ Reload Images"):
        if os.path.exists(IMAGE_CACHE):
            os.remove(IMAGE_CACHE)
        st.session_state.image_database = None
        st.rerun()
    
    if st.button("🗑️ Clear Current Chat"):
        if len(sessions) > 1:
            delete_session_permanently(st.session_state.current_session_id)
            # Switch to another session
            remaining = [s for s in sessions if s['id'] != st.session_state.current_session_id]
            if remaining:
                st.session_state.current_session_id = remaining[0]['id']
                st.session_state.messages = load_session_messages(remaining[0]['id'])
        else:
            # Last session, just clear messages
            delete_session_permanently(st.session_state.current_session_id)
            new_session_id = create_new_chat_session(st.session_state.username)
            st.session_state.current_session_id = new_session_id
            st.session_state.messages = []
        
        try:
            model = genai.GenerativeModel(st.session_state.working_model)
            st.session_state.chat_session = model.start_chat(history=[])
        except:
            pass
        st.rerun()
    
    st.markdown("---")
    
    if st.button("➡️ Logout"):
        logout()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <small>💡 Ask detailed questions like: "What are the complete specifications for CW 10?"</small>
</div>
""", unsafe_allow_html=True)