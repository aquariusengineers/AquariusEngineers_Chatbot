import streamlit as st
import google.generativeai as genai
from PIL import Image
import io
import base64
from pathlib import Path
import PyPDF2
import pdfplumber
from pdf2image import convert_from_bytes
import json
import os
from datetime import datetime

# Page config
st.set_page_config(
    page_title="Product Catalog Chatbot",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        background-color: #f5f5f5;
    }
    .stTextInput > div > div > input {
        background-color: white;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 5px solid #2196f3;
    }
    .bot-message {
        background-color: #f1f8e9;
        border-left: 5px solid #4caf50;
    }
    .upload-section {
        background-color: white;
        padding: 2rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'pdf_data' not in st.session_state:
    st.session_state.pdf_data = {}
if 'images' not in st.session_state:
    st.session_state.images = {}

def extract_pdf_content(pdf_file):
    """Extract text and images from PDF"""
    try:
        pdf_content = {
            'text': [],
            'images': [],
            'metadata': {
                'filename': pdf_file.name,
                'upload_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        }
        
        # Read PDF bytes
        pdf_bytes = pdf_file.read()
        
        # Extract text using pdfplumber
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page_num, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text:
                    pdf_content['text'].append({
                        'page': page_num + 1,
                        'content': text
                    })
        
        # Extract images
        try:
            images = convert_from_bytes(pdf_bytes)
            for idx, img in enumerate(images):
                # Convert to bytes
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format='PNG')
                img_byte_arr = img_byte_arr.getvalue()
                
                pdf_content['images'].append({
                    'page': idx + 1,
                    'data': base64.b64encode(img_byte_arr).decode()
                })
        except Exception as e:
            st.warning(f"Could not extract images: {str(e)}")
        
        return pdf_content
    
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        return None

def create_context_from_pdfs():
    """Create a searchable context from all uploaded PDFs"""
    context = ""
    for filename, data in st.session_state.pdf_data.items():
        context += f"\n\n--- Document: {filename} ---\n"
        for page_data in data['text']:
            context += f"\n[Page {page_data['page']}]\n{page_data['content']}\n"
    return context

def get_relevant_images(query, top_k=3):
    """Get relevant images based on query (simple keyword matching)"""
    relevant_images = []
    query_lower = query.lower()
    
    for filename, data in st.session_state.pdf_data.items():
        for img_data in data['images']:
            # Simple relevance: return first few images
            if len(relevant_images) < top_k:
                relevant_images.append({
                    'source': filename,
                    'page': img_data['page'],
                    'data': img_data['data']
                })
    
    return relevant_images

def query_gemini(user_query, context):
    """Query Google Gemini with context"""
    try:
        # Configure Gemini
        genai.configure(api_key=st.session_state.gemini_api_key)
        model = genai.GenerativeModel('gemini-pro')
        
        # Create prompt
        prompt = f"""You are a helpful product catalog assistant for a company. 
You have access to the company's product documentation and should help sales team members find information about products and models.

Context from product documentation:
{context[:8000]}  # Limit context size

User Question: {user_query}

Instructions:
- Provide accurate information based on the documentation
- If you find specific model information, present it clearly
- Include page references when possible
- If information is not in the documentation, say so
- Be concise but thorough
- Format your response in a clear, readable way

Answer:"""
        
        # Generate response
        response = model.generate_content(prompt)
        return response.text
    
    except Exception as e:
        return f"Error querying AI: {str(e)}\n\nPlease check your API key and try again."

# Sidebar - Setup
with st.sidebar:
    st.title("⚙️ Setup")
    
    # API Key input
    api_key = st.text_input(
        "Google Gemini API Key",
        type="password",
        help="Get free API key from https://makersuite.google.com/app/apikey"
    )
    
    if api_key:
        st.session_state.gemini_api_key = api_key
        st.success("✅ API Key configured")
    
    st.markdown("---")
    
    # PDF Upload
    st.title("📄 Upload PDFs")
    uploaded_files = st.file_uploader(
        "Upload product catalogs",
        type=['pdf'],
        accept_multiple_files=True,
        help="Upload your product PDF files"
    )
    
    if uploaded_files:
        if st.button("Process PDFs"):
            with st.spinner("Processing PDFs..."):
                for pdf_file in uploaded_files:
                    if pdf_file.name not in st.session_state.pdf_data:
                        content = extract_pdf_content(pdf_file)
                        if content:
                            st.session_state.pdf_data[pdf_file.name] = content
                            st.success(f"✅ {pdf_file.name}")
    
    # Show uploaded files
    if st.session_state.pdf_data:
        st.markdown("---")
        st.subheader("📚 Loaded Documents")
        for filename in st.session_state.pdf_data.keys():
            st.text(f"• {filename}")
        
        if st.button("Clear All Data"):
            st.session_state.pdf_data = {}
            st.session_state.messages = []
            st.rerun()

# Main content
st.title("🤖 Product Catalog Chatbot")
st.markdown("Ask me anything about your products and models!")

# Check if setup is complete
if not api_key:
    st.warning("⚠️ Please enter your Google Gemini API key in the sidebar to get started.")
    st.info("""
    **How to get started:**
    1. Get a free API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
    2. Enter the API key in the sidebar
    3. Upload your product PDF files
    4. Start chatting!
    """)
    st.stop()

if not st.session_state.pdf_data:
    st.info("📤 Please upload your product PDF files in the sidebar to begin.")
    st.stop()

# Chat interface
st.markdown("---")

# Display chat history
for message in st.session_state.messages:
    role = message["role"]
    content = message["content"]
    
    if role == "user":
        st.markdown(f'<div class="chat-message user-message">👤 <strong>You:</strong><br>{content}</div>', 
                   unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="chat-message bot-message">🤖 <strong>Assistant:</strong><br>{content}</div>', 
                   unsafe_allow_html=True)
        
        # Show images if available
        if "images" in message:
            cols = st.columns(len(message["images"]))
            for idx, img_data in enumerate(message["images"]):
                with cols[idx]:
                    img_bytes = base64.b64decode(img_data['data'])
                    st.image(img_bytes, caption=f"Page {img_data['page']}", use_column_width=True)

# Chat input
user_query = st.chat_input("Ask about any product or model...")

if user_query:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_query})
    
    # Create context and query
    with st.spinner("Searching product catalog..."):
        context = create_context_from_pdfs()
        response = query_gemini(user_query, context)
        
        # Get relevant images
        relevant_images = get_relevant_images(user_query)
        
        # Add assistant message
        assistant_message = {
            "role": "assistant",
            "content": response
        }
        
        if relevant_images:
            assistant_message["images"] = relevant_images
        
        st.session_state.messages.append(assistant_message)
    
    # Rerun to show new messages
    st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <small>💡 Tip: Ask specific questions like "Tell me about model X123" or "What are the specs for product Y?"</small>
</div>
""", unsafe_allow_html=True)