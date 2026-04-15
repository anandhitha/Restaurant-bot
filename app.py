import os, json, logging, tempfile, threading
import httpx
from flask import Flask, jsonify
from openai import OpenAI
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MENU_DATA = {"restaurant":"The Spice Kitchen","categories":[{"name":"Starters","items":[{"name":"Chili Cheese Fries","price":5.99,"vegetarian":False,"spicy":True,"allergens":["dairy"],"calories":450},{"name":"Onion Rings","price":3.49,"vegetarian":True,"spicy":False,"allergens":["gluten"],"calories":320},{"name":"Paneer Tikka","price":6.49,"vegetarian":True,"spicy":True,"allergens":["dairy"],"calories":290},{"name":"Spring Rolls","price":4.99,"vegetarian":True,"spicy":False,"allergens":["gluten"],"calories":260}]},{"name":"Main Course","items":[{"name":"Original Double Steakburger","price":6.49,"vegetarian":False,"spicy":False,"allergens":["gluten","dairy"],"calories":680},{"name":"Chicken Biryani","price":8.99,"vegetarian":False,"spicy":True,"allergens":["none"],"calories":550},{"name":"Butter Chicken","price":9.49,"vegetarian":False,"spicy":False,"allergens":["dairy"],"calories":490},{"name":"Margherita Pizza","price":7.99,"vegetarian":True,"spicy":False,"allergens":["gluten","dairy"],"calories":620},{"name":"Paneer Butter Masala","price":8.49,"vegetarian":True,"spicy":False,"allergens":["dairy"],"calories":420},{"name":"Veg Fried Rice","price":6.99,"vegetarian":True,"spicy":False,"allergens":["none"],"calories":380}]},{"name":"Sides","items":[{"name":"Naan","price":1.99,"vegetarian":True,"spicy":False,"allergens":["gluten","dairy"],"calories":180},{"name":"Roti","price":1.49,"vegetarian":True,"spicy":False,"allergens":["gluten"],"calories":120},{"name":"Garlic Bread","price":2.49,"vegetarian":True,"spicy":False,"allergens":["gluten","dairy"],"calories":220}]},{"name":"Drinks","items":[{"name":"Mango Lassi","price":3.49,"vegetarian":True,"spicy":False,"allergens":["dairy"],"calories":180},{"name":"Masala Chai","price":2.49,"vegetarian":True,"spicy":False,"allergens":["dairy"],"calories":90},{"name":"Fresh Lime Soda","price":1.99,"vegetarian":True,"spicy":False,"allergens":["none"],"calories":60}]},{"name":"Desserts","items":[{"name":"Gulab Jamun","price":3.99,"vegetarian":True,"spicy":False,"allergens":["dairy","gluten"],"calories":350},{"name":"Mango Kulfi","price":3.49,"vegetarian":True,"spicy":False,"allergens":["dairy"],"calories":200}]}]}
MENU_CONTEXT = json.dumps(MENU_DATA, indent=2)

SYSTEM_PROMPT = f"""You are a friendly restaurant recommendation assistant for 'The Spice Kitchen'.
Menu:
{MENU_CONTEXT}
Rules: Recommend only from menu. Include price, calories, allergens. Filter dietary restrictions. 2-3 recommendations max."""

groq_client = OpenAI(api_key=os.environ.get("GROQ_API_KEY",""), base_url="https://api.groq.com/openai/v1")

async def call_clu(text):
    url = f"{os.environ['CLU_ENDPOINT']}language/:analyze-conversations?api-version=2023-04-01"
    headers = {"Ocp-Apim-Subscription-Key": os.environ["CLU_KEY"], "Content-Type": "application/json"}
    body = {"kind":"Conversation","analysisInput":{"conversationItem":{"id":"1","text":text,"participantId":"user"}},"parameters":{"projectName":os.environ["CLU_PROJECT_NAME"],"deploymentName":os.environ["CLU_DEPLOYMENT_NAME"],"stringIndexType":"TextElement_V8"}}
    async with httpx.AsyncClient() as client:
        r = await client.post(url, headers=headers, json=body, timeout=30)
        result = r.json()
    pred = result.get("result",{}).get("prediction",{})
    top = pred.get("topIntent","None")
    conf = 0.0
    for i in pred.get("intents",[]):
        if i["category"]==top: conf=i.get("confidenceScore",0.0); break
    ents = [{"category":e["category"],"text":e["text"],"confidence":e.get("confidenceScore",0.0)} for e in pred.get("entities",[])]
    return {"intent":top,"confidence":conf,"entities":ents,"original_text":text}

def get_recommendation(msg, clu=None, img=None):
    parts = []
    if clu:
        parts.append(f"Customer: {clu['original_text']}")
        parts.append(f"Intent: {clu['intent']} ({clu['confidence']:.2%})")
        if clu["entities"]: parts.append("Entities: " + ", ".join(f"{e['category']}: {e['text']}" for e in clu["entities"]))
    if img:
        parts.append(f"Food image. Description: {img['description']}. Tags: {', '.join(img['tags'])}")
        parts.append("Suggest similar dishes.")
    if not parts: parts.append(f"Customer: {msg}")
    r = groq_client.chat.completions.create(model="llama-3.1-8b-instant",messages=[{"role":"system","content":SYSTEM_PROMPT},{"role":"user","content":"\n".join(parts)}],max_tokens=500,temperature=0.7)
    return r.choices[0].message.content

def analyze_image_api(image_path):
    try:
        url = f"{os.environ['VISION_ENDPOINT']}vision/v3.2/analyze?visualFeatures=Tags,Description"
        headers = {"Ocp-Apim-Subscription-Key": os.environ["VISION_KEY"], "Content-Type": "application/octet-stream"}
        with open(image_path, "rb") as f:
            r = httpx.post(url, headers=headers, content=f.read(), timeout=30)
        result = r.json()
        tags = [t["name"] for t in result.get("tags",[]) if t["confidence"]>0.5]
        desc = result.get("description",{}).get("captions",[{}])[0].get("text","") if result.get("description",{}).get("captions") else ""
        return {"tags":tags,"description":desc}
    except: return {"tags":[],"description":""}

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Welcome to The Spice Kitchen!\n\n1. Type your preference\n2. Send a voice message\n3. Send a food photo\n\nWhat are you in the mood for?")

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    txt = update.message.text
    await update.message.reply_text("Checking our menu...")
    clu = await call_clu(txt)
    rec = get_recommendation(txt, clu=clu)
    await update.message.reply_text(rec + f"\n\n[CLU: {clu['intent']} ({clu['confidence']:.0%})]")

async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Voice processing not available in cloud deployment. Please type your request.")

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Analyzing your food image...")
    photo = update.message.photo[-1]
    pf = await photo.get_file()
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        ip = tmp.name; await pf.download_to_drive(ip)
    info = analyze_image_api(ip)
    if info["description"]: await update.message.reply_text(f"I see: {info['description']}\nFinding similar dishes...")
    r = get_recommendation("", img=info)
    await update.message.reply_text(r + f"\n\n[Vision: {', '.join(info['tags'][:5])}]")
    os.unlink(ip)

async def handle_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    mt = "THE SPICE KITCHEN MENU\n\n"
    for c in MENU_DATA["categories"]:
        mt += f"--- {c['name']} ---\n"
        for i in c["items"]:
            v="V" if i["vegetarian"] else "NV"; s=" (spicy)" if i.get("spicy") else ""
            mt += f"  [{v}] {i['name']} - ${i['price']:.2f} ({i['calories']} cal){s}\n"
        mt += "\n"
    await update.message.reply_text(mt)

flask_app = Flask(__name__)

@flask_app.route("/")
def home():
    return jsonify({"status":"running","bot":"Restaurant Recommendation Bot"})

@flask_app.route("/health")
def health():
    return jsonify({"status":"healthy"})

def run_bot():
    token = os.environ.get("TELEGRAM_BOT_TOKEN","")
    if not token: logger.error("No token"); return
    app = Application.builder().token(token).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", start))
    app.add_handler(CommandHandler("menu", handle_menu))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    logger.info("Bot polling started")
    app.run_polling()

if __name__ == "__main__":
    t = threading.Thread(target=run_bot, daemon=True)
    t.start()
    logger.info("Flask starting on port 8000")
    port = int(os.environ.get("PORT", 8000))
    flask_app.run(host="0.0.0.0", port=port)
