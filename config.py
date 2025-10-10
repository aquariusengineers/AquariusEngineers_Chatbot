# ========================================
# AQUARIUS AI ASSISTANT - CONFIGURATION
# ========================================
# Edit this file to configure your application

# ============= API CONFIGURATION =============
# Get your free API key from: https://makersuite.google.com/app/apikey
import streamlit as st

GEMINI_API_KEY = st.secrets["external_api"]["api_key"] # Replace with your actual key

# ============= FILE PATHS =============
# Folder containing your product PDF files
PDF_FOLDER = "./product_pdfs"

# Cache file for processed data (speeds up loading)
DATA_CACHE = "./data_cache.pkl"

# Query log file (tracks usage)
QUERY_LOG = "./query_log.json"

# ============= PROCESSING SETTINGS =============
# Image quality (DPI) - Lower = faster processing, smaller file
# Recommended: 100-150 for good balance
IMAGE_DPI = 150

# Maximum pages to process per PDF (0 = all pages)
# Use if PDFs are very large and only first pages matter
MAX_PAGES_PER_PDF = 0  # 0 means process all pages

# Context size for AI prompts (characters)
# Larger = more context but slower, more expensive
# Recommended: 5000-8000
CONTEXT_SIZE = 5000

# Number of text sections to retrieve per query
MAX_TEXT_SECTIONS = 15

# Number of images to show per query
MAX_IMAGES_PER_QUERY = 5

# ============= UI CUSTOMIZATION =============
# Company name shown in header
COMPANY_NAME = "Aquarius Engineers"

# Page title
PAGE_TITLE = "Aquarius Product Assistant"

# Page icon (emoji)
PAGE_ICON = "🏗️"

# Theme colors (CSS)
PRIMARY_COLOR = "#667eea"
SECONDARY_COLOR = "#764ba2"

# ============= FEATURES =============
# Enable/disable features
ENABLE_IMAGE_ANALYSIS = True  # Analyze images with Gemini Vision
ENABLE_TABLE_EXTRACTION = True  # Extract specification tables
ENABLE_MODEL_INDEX = True  # Show model index in sidebar
ENABLE_QUERY_LOGGING = False  # Log all queries (for analytics)
ENABLE_CHAT_EXPORT = True  # Allow exporting conversations

# ============= QUICK SEARCH QUERIES =============
# Pre-defined quick search buttons (max 4)
QUICK_QUERIES = [
    "Show all concrete pump models",
    "CW 10 specifications",
    "Batching plant options",
    "Company contact details"
]

# ============= GEMINI API SETTINGS =============
# Model to use (don't change unless you know what you're doing)
GEMINI_MODEL = "models/gemini-2.5-flash"  # Free tier
# Alternative: "gemini-1.5-pro" (paid, better quality)

# Temperature (creativity) - 0 to 1
# 0 = factual, consistent
# 1 = creative, varied
TEMPERATURE = 0.1  # Keep low for technical accuracy

# ============= ADVANCED SETTINGS =============
# Model detection patterns (regex)
# Add patterns if your model numbers don't get detected
MODEL_PATTERNS = [
    r'\b([A-Z]+\s*\d+(?:\s*[A-Z]+)*)\b',  # CW 10, SP 30
    r'\b(\d{3,4}\s*[A-Z]+(?:\s*[A-Z]+)*)\b',  # 1004 D SHP
]

# Cache expiry (hours) - Rebuild cache after this time
# 0 = never expire, use manual refresh only
CACHE_EXPIRY_HOURS = 0

# Maximum tokens per request (safety limit)
MAX_TOKENS = 8000

# ============= VALIDATION =============
def validate_config():
    """Validate configuration settings"""
    errors = []
    
    if GEMINI_API_KEY == "AIzaSyCo2tOwRAqPbQvzTljGOaPAHg9p9uk4zKA":
        errors.append("❌ Please set your GEMINI_API_KEY")
    
    if not PDF_FOLDER:
        errors.append("❌ PDF_FOLDER cannot be empty")
    
    if IMAGE_DPI < 50 or IMAGE_DPI > 300:
        errors.append("⚠️ IMAGE_DPI should be between 50-300")
    
    if CONTEXT_SIZE < 1000 or CONTEXT_SIZE > 20000:
        errors.append("⚠️ CONTEXT_SIZE should be between 1000-20000")
    
    if len(QUICK_QUERIES) > 4:
        errors.append("⚠️ Maximum 4 quick queries recommended")
    
    return errors

# ============= HELPER FUNCTIONS =============
def get_config_summary():
    """Get a summary of current configuration"""
    return f"""
    Configuration Summary:
    =====================
    API Key: {'✅ Set' if GEMINI_API_KEY != 'YOUR_API_KEY_HERE' else '❌ Not Set'}
    PDF Folder: {PDF_FOLDER}
    Cache File: {DATA_CACHE}
    Image Quality: {IMAGE_DPI} DPI
    Context Size: {CONTEXT_SIZE} chars
    Model: {GEMINI_MODEL}
    
    Features:
    - Image Analysis: {'✅' if ENABLE_IMAGE_ANALYSIS else '❌'}
    - Table Extraction: {'✅' if ENABLE_TABLE_EXTRACTION else '❌'}
    - Model Index: {'✅' if ENABLE_MODEL_INDEX else '❌'}
    - Query Logging: {'✅' if ENABLE_QUERY_LOGGING else '❌'}
    - Chat Export: {'✅' if ENABLE_CHAT_EXPORT else '❌'}
    """

# ============= DEPLOYMENT PROFILES =============
# Uncomment one profile to use preset configurations

# DEVELOPMENT PROFILE
# Fast processing, lower quality, more logging
"""
IMAGE_DPI = 100
CONTEXT_SIZE = 3000
ENABLE_QUERY_LOGGING = True
"""

# PRODUCTION PROFILE
# High quality, optimized performance
"""
IMAGE_DPI = 150
CONTEXT_SIZE = 5000
ENABLE_QUERY_LOGGING = True
MAX_PAGES_PER_PDF = 0
"""

# DEMO PROFILE
# Quick setup for demonstrations
"""
IMAGE_DPI = 100
CONTEXT_SIZE = 3000
MAX_PAGES_PER_PDF = 20
ENABLE_IMAGE_ANALYSIS = False
"""

# ============= NOTES =============
"""
GETTING YOUR API KEY:
1. Go to: https://makersuite.google.com/app/apikey
2. Sign in with Google
3. Click "Create API Key"
4. Copy and paste above

PDF SETUP:
1. Create folder: mkdir product_pdfs
2. Copy all PDFs into this folder
3. Run the app - it will process automatically

FIRST RUN:
- Takes 2-5 minutes to process PDFs
- Creates cache file for faster subsequent loads
- Shows progress bar

UPDATES:
- Add new PDFs to product_pdfs/ folder
- Click "Refresh Database" in app sidebar
- Or delete data_cache.pkl and restart

PERFORMANCE:
- Lower IMAGE_DPI = faster but lower quality
- Lower CONTEXT_SIZE = faster but less accurate
- Enable caching for best performance

COSTS:
- Free tier: 1,500 requests/day
- Paid: ~$0.001 per request
- No credit card needed for free tier

SUPPORT:
- Documentation: See SETUP_GUIDE.md
- Issues: Check troubleshooting section
- Updates: pip install --upgrade google-generativeai
"""