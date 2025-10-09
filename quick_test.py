"""
Quick test with your API key
"""
import google.generativeai as genai

# Your API key
API_KEY = "AIzaSyBdVTztCMrPiqB01vHZZOrME19Cjk4H9EM"

print("Testing Gemini API...")
print("=" * 60)

# Configure
genai.configure(api_key=API_KEY)

# List available models
print("\n📋 Available Models:")
print("-" * 60)

try:
    for model in genai.list_models():
        if 'generateContent' in model.supported_generation_methods:
            print(f"✅ {model.name}")
except Exception as e:
    print(f"❌ Error listing models: {e}")
    exit(1)

print("\n" + "=" * 60)
print("🧪 Testing Models...")
print("=" * 60)

# Test different model names (with models/ prefix for Gemini 2.x)
test_models = [
    "models/gemini-2.5-flash",
    "models/gemini-2.5-pro",
    "models/gemini-2.0-flash-exp",
    "models/gemini-flash-latest",
    "models/gemini-pro-latest",
    "models/gemini-2.0-flash",
]

working_models = []

for model_name in test_models:
    try:
        print(f"\nTesting: {model_name}...")
        model = genai.GenerativeModel(model_name)
        response = model.generate_content("Say hello in one word")
        print(f"   ✅ WORKS! Response: '{response.text.strip()}'")
        working_models.append(model_name)
    except Exception as e:
        print(f"   ❌ Failed: {str(e)[:80]}")

print("\n" + "=" * 60)
print("📊 RESULTS")
print("=" * 60)

if working_models:
    print(f"\n✅ Found {len(working_models)} working model(s):")
    for m in working_models:
        print(f"   • {m}")
    
    print(f"\n🎯 RECOMMENDED MODEL: {working_models[0]}")
    print(f"\n📝 Update your app.py line 16-22:")
    print(f"   GEMINI_MODELS = ['{working_models[0]}']")
else:
    print("\n❌ No working models found!")
    print("\nPossible issues:")
    print("1. API key might not be activated")
    print("2. Visit: https://aistudio.google.com")
    print("3. Accept terms and conditions")
    print("4. Try generating a new API key")

print("\n" + "=" * 60)