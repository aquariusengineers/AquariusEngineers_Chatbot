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

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings

# ============= CONFIGURATION (EDIT THESE) =============
GEMINI_API_KEY = st.secrets["external_api"]["api_key"]  # Your API key
PDF_FOLDER = "./product_pdfs"  # Folder containing your PDFs
DATA_CACHE = "./data_cache.pkl"  # Cached processed data
CONFIG_USERNAME = st.secrets["app_credentials"]["username"]
CONFIG_PASSWORD = st.secrets["app_credentials"]["password"]
LOGO_PATH = "./LOGO_Aquarius.png"  # Path to your company logo (optional)

# Model names to try (in order)
GEMINI_MODELS = [
    "models/gemini-2.5-flash",
    "models/gemini-2.0-flash-exp",
    "models/gemini-flash-latest",
    "models/gemini-pro-latest",
]

# Image extraction settings
SKIP_FIRST_N_PAGES = 4  # Skip first N pages (usually company info, not product images)
IMAGE_DPI = 150  # Image quality (100-200 recommended)
# =====================================================

# Page config (NO CHANGE)
st.set_page_config(
    page_title="Aquarius Product Assistant",
    page_icon=LOGO_PATH,
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS (NO CHANGE)
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
    .model-spec-box {
        background: #f8f9fa;
        border-left: 4px solid #28a745;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0.5rem;
    }
    .product-image {
        border-radius: 0.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    .header {
        display: flex;
        align-items: center;
        gap: 15px;
        text-align: center;
        justify-content: center;
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
    .spec-table {
        background: white;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .stChatInput > div {
        background: white;
        border-radius: 2rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Initialize Streamlit session state for AUTHENTICATION and CHAT
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'username' not in st.session_state:
    st.session_state.username = None
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'product_database' not in st.session_state:
    st.session_state.product_database = None
if 'models_index' not in st.session_state:
    st.session_state.models_index = {}
if 'chat_session' not in st.session_state:
    st.session_state.chat_session = None
if 'working_model' not in st.session_state:
    st.session_state.working_model = None


# Initialize Gemini
genai.configure(api_key=GEMINI_API_KEY)

# --- AUTHENTICATION FUNCTIONS ---

def authenticate_user(username: str, password: str) -> bool:
    """Checks credentials and sets session state if successful."""
    if username == CONFIG_USERNAME and password == CONFIG_PASSWORD:
        st.session_state.authenticated = True
        st.session_state.username = username
        return True
    st.session_state.authenticated = False
    st.session_state.username = None
    return False

def login_form():
    """Renders the login form."""

    # Create columns: an empty column on the left, a column for the form, an empty column on the right
    # The numbers [1, 2, 1] mean the middle column will be 2 units wide,
    # and the side columns 1 unit each, making the form take up 50% of the available width.
    col1, col2, col3 = st.columns([1, 2, 1]) 

    with col2: # Place the form inside the middle column
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
    """Clears authentication and session-specific data."""
    for key in ['authenticated', 'username', 'messages', 'product_database', 'models_index', 'chat_session']:
        if key in st.session_state:
            del st.session_state[key]
    st.rerun() 

# --- GEMINI INIT (Ensures this only runs once) ---

def initialize_gemini():
    """Find working model and initialize chat session."""
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
        st.error("❌ Could not find a working Gemini model. Please check your API key.")
        st.stop()
        
    st.session_state.working_model = WORKING_MODEL
    
    try:
        model = genai.GenerativeModel(WORKING_MODEL)
        st.session_state.chat_session = model.start_chat(history=[])
    except Exception as e:
        st.error(f"Could not start chat session: {e}")
        st.stop()
        
# --- END GEMINI INIT ---


def extract_models_from_text(text):
    models = {}
    
    # Pattern for model names (e.g., CW 10, SP 30, 1004 D SHP)
    model_patterns = [
        r'\b([A-Z]+\s*\d+(?:\s*[A-Z]+)*)\b',  # CW 10, SP 30
        r'\b(\d{3,4}\s*[A-Z]+(?:\s*[A-Z]+)*)\b',  # 1004 D SHP
    ]
    
    for pattern in model_patterns:
        matches = re.finditer(pattern, text)
        for match in matches:
            model_name = match.group(1).strip()
            # Get context around model (500 chars before and after)
            start = max(0, match.start() - 500)
            end = min(len(text), match.end() + 500)
            context = text[start:end]
            
            if model_name not in models:
                models[model_name] = []
            models[model_name].append(context)
    
    return models

def process_pdf_comprehensive(pdf_path):
    """Comprehensive PDF processing - text, tables, images (WITHOUT description analysis)"""
    pdf_data = {
        'filename': os.path.basename(pdf_path),
        'full_text': '',
        'pages': [],
        'models': {},
        'images': [],
        'tables': []
    }
    
    try:
        # Extract text and structure
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                page_text = page.extract_text() or ""
                pdf_data['full_text'] += f"\n--- Page {page_num + 1} ---\n{page_text}"
                
                page_data = {
                    'page_num': page_num + 1,
                    'text': page_text,
                    'tables': []
                }
                
                # Extract tables
                tables = page.extract_tables()
                if tables:
                    for table in tables:
                        page_data['tables'].append(table)
                        pdf_data['tables'].append({
                            'page': page_num + 1,
                            'data': table
                        })
                
                pdf_data['pages'].append(page_data)
        
        # Extract models and their specifications
        pdf_data['models'] = extract_models_from_text(pdf_data['full_text'])
        
        # Extract images (WITHOUT API calls for description)
        try:
            images = convert_from_path(pdf_path, dpi=IMAGE_DPI)
            for idx, img in enumerate(images):
                page_num = idx + 1
                
                # Skip first N pages 
                if page_num <= SKIP_FIRST_N_PAGES:
                    continue
                
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format='PNG')
                img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode()
                
                pdf_data['images'].append({
                    'page': page_num,
                    'data': img_base64
                })
        except Exception as e:
            st.warning(f"Could not extract images from {pdf_path}: {str(e)}")
    
    except Exception as e:
        st.error(f"Error processing {pdf_path}: {str(e)}")
    
    return pdf_data



def load_or_process_pdfs():
    """Load cached data or process PDFs from folder (NO image analysis)."""
    
    # Check if cache exists
    if os.path.exists(DATA_CACHE):
        try:
            with open(DATA_CACHE, 'rb') as f:
                return pickle.load(f)
        except:
            pass
    
    # Process PDFs
    if not os.path.exists(PDF_FOLDER):
        st.error(f"PDF folder not found: {PDF_FOLDER}")
        st.info("Please create a folder called 'product_pdfs' and add your PDF files there.")
        return None
    
    pdf_files = list(Path(PDF_FOLDER).glob("*.pdf"))
    
    if not pdf_files:
        st.warning(f"No PDF files found in {PDF_FOLDER}")
        return None
    
    all_data = {
        'documents': [],
        'full_text': '',
        'models_index': {},
        'all_images': []
    }
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, pdf_path in enumerate(pdf_files):
        status_text.text(f"Processing {pdf_path.name}...")
        pdf_data = process_pdf_comprehensive(str(pdf_path))
        
        # Indexing Models and Text
        all_data['documents'].append(pdf_data)
        all_data['full_text'] += f"\n\n{'='*50}\nDOCUMENT: {pdf_data['filename']}\n{'='*50}\n{pdf_data['full_text']}"
        
        for model, contexts in pdf_data['models'].items():
            if model not in all_data['models_index']:
                all_data['models_index'][model] = []
            all_data['models_index'][model].extend([{
                'source': pdf_data['filename'],
                'context': ctx
            } for ctx in contexts])
        
        # Store images without descriptions
        for img in pdf_data['images']:
            all_data['all_images'].append({
                'source': pdf_data['filename'],
                'page': img['page'],
                'data': img['data']
            })
            
        progress_bar.progress((idx + 1) / len(pdf_files))
    
    # Save cache
    try:
        with open(DATA_CACHE, 'wb') as f:
            pickle.dump(all_data, f)
    except Exception as e:
        st.warning(f"Could not save cache: {str(e)}")
    
    status_text.text("✅ All PDFs processed successfully!")
    progress_bar.empty()
    
    return all_data

def find_relevant_content(query, database):
    """Find all relevant content for the query, including images from ALL relevant PDFs"""
    query_lower = query.lower()
    relevant_data = {
        'text_sections': [],
        'models': [],
        'images': [],
        'tables': [],
        'relevant_docs': {}  # Track which documents are relevant with scores
    }
    
    # Check if query mentions specific models
    for model in database['models_index'].keys():
        if model.lower() in query_lower or query_lower in model.lower():
            relevant_data['models'].append({
                'model': model,
                'data': database['models_index'][model]
            })
            # Track which documents mention this model
            for data in database['models_index'][model]:
                doc_name = data['source']
                relevant_data['relevant_docs'][doc_name] = relevant_data['relevant_docs'].get(doc_name, 0) + 5
    
    # Find relevant text sections and score documents
    words = [w for w in query_lower.split() if len(w) > 3]
    
    for doc in database['documents']:
        text = doc['full_text'].lower()
        doc_score = 0
        
        for word in words:
            if word in text:
                doc_score += text.count(word)
                idx = 0
                section_count = 0
                while section_count < 3:  # Limit sections per word
                    idx = text.find(word, idx)
                    if idx == -1:
                        break
                    
                    start = max(0, idx - 300)
                    end = min(len(doc['full_text']), idx + 300)
                    section = doc['full_text'][start:end]
                    
                    relevant_data['text_sections'].append({
                        'source': doc['filename'],
                        'content': section
                    })
                    
                    idx += 1
                    section_count += 1
                    
                    if len(relevant_data['text_sections']) >= 10:
                        break
        
        # Update document score
        if doc_score > 0:
            relevant_data['relevant_docs'][doc['filename']] = relevant_data['relevant_docs'].get(doc['filename'], 0) + doc_score
        
        # Get relevant tables
        for table_data in doc['tables']:
            table_str = str(table_data['data']).lower()
            if any(word in table_str for word in words):
                relevant_data['tables'].append(table_data)
                relevant_data['relevant_docs'][doc['filename']] = relevant_data['relevant_docs'].get(doc['filename'], 0) + 3

    # Sort documents by relevance score
    sorted_docs = sorted(relevant_data['relevant_docs'].items(), key=lambda x: x[1], reverse=True)
    
    # Get images from ALL relevant documents (distribute evenly)
    # Dynamically determine max images based on number of relevant docs
    if sorted_docs:
        max_images = min(10, len(sorted_docs) * 3)  # Up to 3 images per doc, max 10 total
        images_per_doc = max(1, max_images // len(sorted_docs))
        
        for doc_name, score in sorted_docs:
            # Find the document
            for doc in database['documents']:
                if doc['filename'] == doc_name:
                    # Get up to images_per_doc images from this document
                    doc_images_collected = 0
                    for img in doc['images']:
                        if doc_images_collected >= images_per_doc:
                            break
                        if len(relevant_data['images']) >= max_images:
                            break
                        relevant_data['images'].append({
                            'source': doc['filename'],
                            'page': img['page'],
                            'data': img['data']
                        })
                        doc_images_collected += 1
                    break
            
            # Stop if we've reached max images
            if len(relevant_data['images']) >= max_images:
                break
    
    # Remove the relevant_docs dict before returning (not needed in response)
    del relevant_data['relevant_docs']
            
    return relevant_data

def create_comprehensive_prompt(query, relevant_data, database):
    prompt = f"""You are an expert product assistant for Aquarius Engineers, a construction equipment company.
You have access to complete product catalogs and must provide COMPREHENSIVE, DETAILED answers.

User Question: {query}

IMPORTANT INSTRUCTIONS:
1. Provide COMPLETE information, not summaries
2. Include ALL specifications when discussing models
3. Include ALL relevant details from the documentation
4. Format specifications in clear tables
5. DO NOT mention page numbers or document sources in your response
6. If asked about a model, provide FULL specifications
7. Be thorough and detailed in your response
8. Present information naturally without citing sources

"""
    
    # Add model-specific data
    if relevant_data['models']:
        prompt += "\n\n=== SPECIFIC MODEL INFORMATION ===\n"
        for model_info in relevant_data['models']:
            prompt += f"\n--- Model: {model_info['model']} ---\n"
            for data in model_info['data'][:3]:  # Top 3 contexts
                prompt += f"\n{data['context']}\n\n"
    
    # Add relevant text sections
    if relevant_data['text_sections']:
        prompt += "\n\n=== RELEVANT DOCUMENTATION SECTIONS ===\n"
        for section in relevant_data['text_sections'][:15]:  # Top 15 sections
            prompt += f"\n{section['content']}\n"
    
    # Add table data
    if relevant_data['tables']:
        prompt += "\n\n=== SPECIFICATION TABLES ===\n"
        for table in relevant_data['tables'][:5]:
            # Convert table data to JSON string for better model ingestion
            table_json = json.dumps(table['data'])
            prompt += f"\n{table_json}\n"
    
    # Add general context
    prompt += f"\n\n=== GENERAL PRODUCT INFORMATION ===\n{database['full_text'][:5000]}\n"
    
    prompt += """

Now provide a COMPREHENSIVE answer based on the above information, including:
- Complete specifications if relevant
- All applicable model information
- Features and capabilities
- Technical details

Remember: DO NOT mention sources, page numbers, or document names in your response.

Answer:"""
    
    return prompt

def query_gemini_comprehensive(query, database):
    """Query with comprehensive context using the persistent chat session."""
    try:
        # Find all relevant content (always includes images from relevant PDFs)
        relevant_data = find_relevant_content(query, database)
        
        # Create comprehensive RAG prompt
        prompt = create_comprehensive_prompt(query, relevant_data, database)
        
        # Use the chat session for conversation history
        response = st.session_state.chat_session.send_message(prompt)
        
        return {
            'answer': response.text,
            'relevant_images': relevant_data['images'],
            'models_mentioned': [m['model'] for m in relevant_data['models']]
        }
    
    except Exception as e:
        return {
            'answer': f"Error: {str(e)}",
            'relevant_images': [],
            'models_mentioned': []
        }

# ==================== MAIN APPLICATION LOGIC ====================

if not st.session_state.authenticated:
    login_form()

# --- Everything below this line runs only if authenticated ---
initialize_gemini()

# Load logo if exists
logo_html = ""
if os.path.exists(LOGO_PATH):
    try:
        with open(LOGO_PATH, "rb") as f:
            logo_data = base64.b64encode(f.read()).decode()
            logo_html = f'<img src="data:image/png;base64,{logo_data}" style="height: 80px; margin-bottom: 10px;">'
    except:
        logo_html = "🏗️"
else:
    logo_html = "🏗️"

# Header
st.markdown(f"""
<div class="header-banner">
    <span class="header">
    {logo_html}
    <h1>Aquarius Product Assistant</h1>
    </span>
    <p>Your comprehensive guide to Aquarius construction equipment</p>
</div>
""", unsafe_allow_html=True)

# Load database
if st.session_state.product_database is None:
    # Need to clear chat session history if database is reloaded
    if 'chat_session' in st.session_state:
        try:
            model = genai.GenerativeModel(st.session_state.working_model)
            st.session_state.chat_session = model.start_chat(history=[])
        except Exception as e:
            st.error(f"Could not reset chat session during database load: {e}")
            st.stop()
    
    with st.spinner("🔄 Loading product database..."):
        st.session_state.product_database = load_or_process_pdfs()
        
        if st.session_state.product_database:
            st.session_state.models_index = st.session_state.product_database['models_index']


if st.session_state.product_database is None:
    st.error("⚠️ Cannot load product database. Please check configuration.")
    st.stop()

# Quick access buttons
st.markdown("### 🔍 Quick Search")
cols = st.columns(4)

st.markdown("---")


# Display chat history
for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(f'<div class="chat-message user-message">👤 <strong>You:</strong><br>{message["content"]}</div>', 
                    unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="chat-message bot-message">🤖 <strong>Assistant:</strong><br>{message["content"]}</div>', 
                    unsafe_allow_html=True)
        
        # Show images if available
        if "images" in message and message["images"]:
            st.markdown("### 📸 Relevant Product Images")
            
            # Display all images without grouping by source
            cols = st.columns(4)
            for idx, img_data in enumerate(message["images"]):
                with cols[idx % 4]:
                    img_bytes = base64.b64decode(img_data['data'])
                    page = img_data.get('page', '?')
                    st.image(img_bytes, caption=f"Page {page}", 
                              use_container_width=True, output_format="PNG")

# Chat input
user_query = st.chat_input("Ask anything...")

if user_query:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_query})
    
    # Query database
    with st.spinner("🔍 Searching and synthesizing information..."):
        result = query_gemini_comprehensive(user_query, st.session_state.product_database)
        
        # Add assistant message
        assistant_message = {
            "role": "assistant",
            "content": result['answer']
        }
        
        if result['relevant_images']:
            assistant_message["images"] = result['relevant_images']
        
        st.session_state.messages.append(assistant_message)
    
    st.rerun()

# Sidebar - Model Index
with st.sidebar:
    st.title("📚 Product Index")
    st.markdown("---")
    
    if st.session_state.models_index:
        st.markdown(f"**{len(st.session_state.models_index)} Models Available**")
        
        # Group models by category
        categories = {}
        for model in st.session_state.models_index.keys():
            # Determine category
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
                        st.rerun()
    
    st.markdown("---")
    
    if st.button("🔄 Refresh Database"):
        if os.path.exists(DATA_CACHE):
            os.remove(DATA_CACHE)
        st.session_state.product_database = None
        st.rerun()
    
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        # Reset the chat session to clear history in the model
        if 'chat_session' in st.session_state:
            try:
                model = genai.GenerativeModel(st.session_state.working_model)
                st.session_state.chat_session = model.start_chat(history=[])
            except Exception as e:
                st.error(f"Could not reset chat session: {e}")
        st.rerun()
        
    st.markdown("---")
    
    if st.button("➡️ Logout"):
        logout()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <small>💡 Ask detailed questions like: "What are the complete specifications for CW 10?" or "Compare all batching plant models"</small>
</div>
""", unsafe_allow_html=True)