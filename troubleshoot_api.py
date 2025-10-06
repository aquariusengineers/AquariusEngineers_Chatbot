"""
Troubleshoot Google Gemini API issues
Run this to diagnose problems with your setup
"""
import sys

print("🔍 Gemini API Troubleshooting Script")
print("=" * 60)

# Step 1: Check google-generativeai installation
print("\n1️⃣ Checking google-generativeai installation...")
try:
    import google.generativeai as genai
    print(f"   ✅ google-generativeai installed (version: {genai.__version__})")
except ImportError as e:
    print(f"   ❌ google-generativeai not installed")
    print(f"   Fix: pip install google-generativeai")
    sys.exit(1)

# Step 2: Get API key
print("\n2️⃣ Checking API key...")
API_KEY = input("   Enter your Gemini API key: ").strip()

if not API_KEY:
    print("   ❌ No API key provided")
    print("   Get one from: https://makersuite.google.com/app/apikey")
    sys.exit(1)

print("   ✅ API key provided")

# Step 3: Configure API
print("\n3️⃣ Configuring API...")
try:
    genai.configure(api_key=API_KEY)
    print("   ✅ API configured")
except Exception as e:
    print(f"   ❌ Configuration failed: {e}")
    sys.exit(1)

# Step 4: List available models
print("\n4️⃣ Fetching available models...")
try:
    models = list(genai.list_models())
    print(f"   ✅ Found {len(models)} total models")
    
    generate_models = [m for m in models if 'generateContent' in m.supported_generation_methods]
    print(f"   ✅ {len(generate_models)} models support generateContent")
    
    if generate_models:
        print("\n   📋 Available models for text generation:")
        for model in generate_models:
            print(f"      • {model.name}")
            print(f"        Display: {model.display_name}")
    else:
        print("   ⚠️  No models support generateContent")
        print("   This might be an API key issue")
        
except Exception as e:
    print(f"   ❌ Failed to list models: {e}")
    print("   Possible issues:")
    print("   - Invalid API key")
    print("   - API key not activated")
    print("   - Network issues")
    sys.exit(1)

# Step 5: Test text generation
print("\n5️⃣ Testing text generation...")
test_models = [
    "gemini-1.5-pro-latest",
    "gemini-1.5-flash-latest",
    "gemini-pro",
    "gemini-1.0-pro-latest",
]

working_model = None
for model_name in test_models:
    try:
        print(f"   Testing {model_name}...", end=" ")
        model = genai.GenerativeModel(model_name)
        response = model.generate_content("Say hello")
        print(f"✅ WORKS! Response: {response.text[:50]}...")
        working_model = model_name
        break
    except Exception as e:
        print(f"❌ Failed: {str(e)[:50]}")

if working_model:
    print(f"\n   🎉 Success! Use this model: {working_model}")
else:
    print("\n   ❌ No working model found")
    print("   Your available models might use different names")

# Step 6: Test vision (if available)
print("\n6️⃣ Testing vision models (optional)...")
vision_models = ["gemini-1.5-pro-latest", "gemini-pro-vision", "gemini-1.5-flash-latest"]

for model_name in vision_models:
    try:
        print(f"   Testing {model_name}...", end=" ")
        model = genai.GenerativeModel(model_name)
        # Simple test without image
        response = model.generate_content("Describe what you see")
        print(f"✅ Available")
        break
    except:
        print(f"❌ Not available")

# Summary
print("\n" + "=" * 60)
print("📊 SUMMARY")
print("=" * 60)

if working_model:
    print(f"✅ Your setup is working!")
    print(f"✅ Recommended model: {working_model}")
    print(f"\n📝 Update app.py line 16-22 with:")
    print(f"   GEMINI_MODELS = ['{working_model}']")
else:
    print("❌ Setup issues detected")
    print("\n🔧 Troubleshooting steps:")
    print("1. Verify API key is correct")
    print("2. Check API key is activated at https://makersuite.google.com")
    print("3. Ensure you have free quota available")
    print("4. Try generating a key again")
    print("5. Check for any service outages")

print("\n💡 If issues persist:")
print("   - Check https://ai.google.dev/gemini-api/docs")
print("   - Verify billing is not required for your region")
print("   - Try from a different network")

print("\n" + "=" * 60)