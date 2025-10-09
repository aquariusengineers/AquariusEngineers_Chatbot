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

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings

# ============= CONFIGURATION (EDIT THESE) =============
GEMINI_API_KEY = "AIzaSyCo2tOwRAqPbQvzTljGOaPAHg9p9uk4zKA"  # Your API key
PDF_FOLDER = "./product_pdfs"  # Folder containing your PDFs
DATA_CACHE = "./data_cache.pkl"  # Cached processed data

# Model names to try (in order) - Updated for Gemini 2.x
GEMINI_MODELS = [
    "models/gemini-2.5-flash",
    "models/gemini-2.0-flash-exp",
    "models/gemini-flash-latest",
    "models/gemini-pro-latest",
]

# Image extraction settings
SKIP_FIRST_N_PAGES = 5  # Skip first N pages (usually company info, not product images)
IMAGE_DPI = 150  # Image quality (100-200 recommended)
# =====================================================

# Page config
st.set_page_config(
    page_title="Aquarius Product Assistant",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
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

# Initialize Gemini
genai.configure(api_key=GEMINI_API_KEY)

# Find working model
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
    st.info("Run check_available_models.py to see available models")
    st.stop()

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'product_database' not in st.session_state:
    st.session_state.product_database = None
if 'models_index' not in st.session_state:
    st.session_state.models_index = {}

def extract_models_from_text(text):
    """Extract model numbers and specifications from text"""
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
    """Comprehensive PDF processing - text, tables, images, structure"""
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
        
        # Extract images
        try:
            images = convert_from_path(pdf_path, dpi=IMAGE_DPI)
            for idx, img in enumerate(images):
                page_num = idx + 1
                
                # Skip first N pages (usually contain company info, not products)
                if page_num <= SKIP_FIRST_N_PAGES:
                    continue
                
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format='PNG')
                img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode()
                
                pdf_data['images'].append({
                    'page': page_num,
                    'data': img_base64,
                    'description': None  # Will be analyzed by Gemini Vision
                })
        except Exception as e:
            st.warning(f"Could not extract images from {pdf_path}: {str(e)}")
    
    except Exception as e:
        st.error(f"Error processing {pdf_path}: {str(e)}")
    
    return pdf_data

def load_or_process_pdfs():
    """Load cached data or process PDFs from folder"""
    
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
        
        all_data['documents'].append(pdf_data)
        all_data['full_text'] += f"\n\n{'='*50}\nDOCUMENT: {pdf_data['filename']}\n{'='*50}\n{pdf_data['full_text']}"
        
        # Index models
        for model, contexts in pdf_data['models'].items():
            if model not in all_data['models_index']:
                all_data['models_index'][model] = []
            all_data['models_index'][model].extend([{
                'source': pdf_data['filename'],
                'context': ctx
            } for ctx in contexts])
        
        # Collect images
        all_data['all_images'].extend([{
            'source': pdf_data['filename'],
            'page': img['page'],
            'data': img['data']
        } for img in pdf_data['images']])
        
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

def analyze_image_with_gemini(image_base64, context=""):
    """Analyze image using Gemini Vision"""
    try:
        # Try vision models - Gemini 2.x supports vision in base models
        vision_models = [
            'models/gemini-2.5-flash',
            'models/gemini-2.0-flash-exp',
            'models/gemini-flash-latest'
        ]
        
        for model_name in vision_models:
            try:
                model = genai.GenerativeModel(model_name)
                
                # Decode base64 to PIL Image
                img_bytes = base64.b64decode(image_base64)
                img = Image.open(io.BytesIO(img_bytes))
                
                prompt = f"""Analyze this image from a product catalog. Describe what you see, including:
- Product models/names visible
- Technical specifications shown
- Diagrams or schematics
- Any tables or data
- Key features highlighted

Context: {context[:500]}

Provide a detailed description that will help answer customer questions."""
                
                response = model.generate_content([prompt, img])
                return response.text
            except:
                continue
        
        return "Image analysis not available with current model"
    except Exception as e:
        return f"Could not analyze image: {str(e)}"

def find_relevant_content(query, database):
    """Find all relevant content for the query"""
    query_lower = query.lower()
    relevant_data = {
        'text_sections': [],
        'models': [],
        'images': [],
        'tables': []
    }
    
    # Check if query mentions specific models
    for model in database['models_index'].keys():
        if model.lower() in query_lower or query_lower in model.lower():
            relevant_data['models'].append({
                'model': model,
                'data': database['models_index'][model]
            })
    
    # Find relevant text sections (500 char chunks)
    words = query_lower.split()
    for doc in database['documents']:
        text = doc['full_text'].lower()
        for word in words:
            if len(word) > 3 and word in text:  # Skip small words
                # Find all occurrences
                idx = 0
                while True:
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
                    if len(relevant_data['text_sections']) >= 10:
                        break
        
        # Get relevant tables
        for table_data in doc['tables']:
            table_str = str(table_data['data']).lower()
            if any(word in table_str for word in words if len(word) > 3):
                relevant_data['tables'].append(table_data)
    
    # Get relevant images (first 3 from relevant documents)
    for doc in database['documents']:
        if any(word in doc['filename'].lower() for word in words):
            for img in doc['images'][:3]:
                relevant_data['images'].append({
                    'source': doc['filename'],
                    'page': img['page'],
                    'data': img['data']
                })
    
    return relevant_data

def create_comprehensive_prompt(query, relevant_data, database):
    """Create detailed prompt with all relevant information"""
    
    prompt = f"""You are an expert product assistant for Aquarius Engineers, a construction equipment company.
You have access to complete product catalogs and must provide COMPREHENSIVE, DETAILED answers.

User Question: {query}

IMPORTANT INSTRUCTIONS:
1. Provide COMPLETE information, not summaries
2. Include ALL specifications when discussing models
3. Include ALL relevant details from the documentation
4. Format specifications in clear tables
5. Cite page numbers and document sources
6. If asked about a model, provide FULL specifications
7. Be thorough and detailed in your response

"""
    
    # Add model-specific data
    if relevant_data['models']:
        prompt += "\n\n=== SPECIFIC MODEL INFORMATION ===\n"
        for model_info in relevant_data['models']:
            prompt += f"\n--- Model: {model_info['model']} ---\n"
            for data in model_info['data'][:3]:  # Top 3 contexts
                prompt += f"Source: {data['source']}\n{data['context']}\n\n"
    
    # Add relevant text sections
    if relevant_data['text_sections']:
        prompt += "\n\n=== RELEVANT DOCUMENTATION SECTIONS ===\n"
        for section in relevant_data['text_sections'][:15]:  # Top 15 sections
            prompt += f"\nFrom: {section['source']}\n{section['content']}\n"
    
    # Add table data
    if relevant_data['tables']:
        prompt += "\n\n=== SPECIFICATION TABLES ===\n"
        for table in relevant_data['tables'][:5]:
            prompt += f"\nPage {table['page']}:\n{table['data']}\n"
    
    # Add general context
    prompt += f"\n\n=== GENERAL PRODUCT INFORMATION ===\n{database['full_text'][:5000]}\n"
    
    prompt += """

Now provide a COMPREHENSIVE answer including:
- Complete specifications if relevant
- All applicable model information
- Features and capabilities
- Technical details
- Any relevant images will be shown separately

Answer:"""
    
    return prompt

def query_gemini_comprehensive(query, database):
    """Query with comprehensive context"""
    try:
        # Find all relevant content
        relevant_data = find_relevant_content(query, database)
        
        # Create comprehensive prompt
        prompt = create_comprehensive_prompt(query, relevant_data, database)
        
        # Query Gemini with working model
        model = genai.GenerativeModel(WORKING_MODEL)
        response = model.generate_content(prompt)
        
        return {
            'answer': response.text,
            'relevant_images': relevant_data['images'][:5],
            'models_mentioned': [m['model'] for m in relevant_data['models']]
        }
    
    except Exception as e:
        return {
            'answer': f"Error: {str(e)}",
            'relevant_images': [],
            'models_mentioned': []
        }

# ==================== UI ====================

# Header
st.markdown(f"""
<div class="header-banner">
    <h1>🏗️ Aquarius Product Assistant</h1>
    <p>Your comprehensive guide to Aquarius construction equipment</p>
</div>
""", unsafe_allow_html=True)

# Load database
if st.session_state.product_database is None:
    with st.spinner("🔄 Loading product database..."):
        st.session_state.product_database = load_or_process_pdfs()
        
        if st.session_state.product_database:
            st.session_state.models_index = st.session_state.product_database['models_index']
            # st.success(f"✅ Loaded {len(st.session_state.product_database['documents'])} product catalogs")
            # st.info(f"📊 Found {len(st.session_state.models_index)} product models in database")

if st.session_state.product_database is None:
    st.error("⚠️ Cannot load product database. Please check configuration.")
    st.stop()

# Quick access buttons
st.markdown("### 🔍 Quick Search")
cols = st.columns(4)
# quick_queries = [
#     "Show all concrete pump models",
#     "CW 10 specifications",
#     "Batching plant options",
#     "Company contact details"
# ]

# for idx, col in enumerate(cols):
#     if col.button(quick_queries[idx], key=f"quick_{idx}"):
#         st.session_state.messages.append({"role": "user", "content": quick_queries[idx]})
#         st.rerun()

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
            cols = st.columns(min(3, len(message["images"])))
            for idx, img_data in enumerate(message["images"]):
                with cols[idx % 3]:
                    img_bytes = base64.b64decode(img_data['data'])
                    # Handle both source formats (from different parts of code)
                    source = img_data.get('source', img_data.get('filename', 'Document'))
                    page = img_data.get('page', '?')
                    st.image(img_bytes, caption=f"{source} - Page {page}", 
                           use_container_width=True, output_format="PNG")

# Chat input
user_query = st.chat_input("Ask anything...")

if user_query:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_query})
    
    # Query database
    with st.spinner("🔍 Searching comprehensive product database..."):
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
        st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <small>💡 Ask detailed questions like: "What are the complete specifications for CW 10?" or "Compare all batching plant models"</small>
</div>
""", unsafe_allow_html=True)