"""NPTech course-guidance chatbot: Flask + optional Amazon Bedrock AI."""

import logging
import os

import boto3
from botocore.exceptions import BotoCoreError, ClientError
from flask import Flask, jsonify, render_template, request

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

application = Flask(__name__)
application.config["MAX_CONTENT_LENGTH"] = 64 * 1024

DEFAULT_REGION = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "us-east-1"
MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "amazon.nova-lite-v1:0")
DEMO_MODE = os.getenv("CHATBOT_DEMO_MODE", "true").strip().lower() in {"1", "true", "yes", "on"}

SYSTEM_PROMPT = """You are tom, the warm, encouraging learning guide for NPTech, whose tagline is
'Empowering Tomorrow's Leaders'. Greet learners naturally and help them explore Data Science
and Full Stack Web Development courses. The learners may be complete beginners, so explain jargon in plain English,
use short headings and practical examples, and suggest manageable hands-on activities.

Data Science learning path: Python for data analysis; NumPy and Pandas; SQL and database fundamentals; data cleaning and preprocessing; exploratory data analysis (EDA); statistics and probability; data visualization using Matplotlib and Seaborn; Power BI or Tableau; and a practical data analysis project.

Full Stack Web Development learning path: HTML and CSS fundamentals; JavaScript programming; responsive web design; React.js frontend development; Python Flask or Django backend development; REST APIs; SQL and database connectivity; authentication and web security; website deployment; and a complete web application project.

Explain how Data Science and Full Stack Web Development complement one another. Do not invent NPTech fees, official course
durations, certificates, schedules, or guarantees. If asked about those, say the learner should
confirm with the NPTech team. Ask one helpful follow-up at a time when needed. Never request
passwords, access keys, payment details, or sensitive personal information."""


def get_bedrock_client():
    """Use the standard AWS credentials chain (IAM role in AWS, profile locally)."""
    return boto3.client("bedrock-runtime", region_name=DEFAULT_REGION)


def demo_reply(text):
    """Beginner-safe local replies so the UI can be practised before AWS is configured."""
    q = text.lower().strip()
    if any(term in q for term in ("data science", "pandas", "numpy", "data analysis", "statistics", "data visualization", "eda", "machine learning", "sql", "dataset")):
        return (
            "Great question! 📊 **Data Science** teaches you how to collect, clean, analyze, and interpret data "
            "to discover useful insights and support decision-making.\n\n"
            "🐍 You will learn Python, NumPy, Pandas, SQL, statistics, data visualization, and machine learning "
            "to work with real-world datasets.\n\n"
            "A good beginner order is: Python basics → NumPy and Pandas → SQL and statistics → Data visualization → "
            "Machine Learning basics → a small data analysis project.\n\n"
            "You can practice with projects such as student performance analysis, sales data analysis, or customer behavior prediction.\n\n"
            "Would you like to start with Python basics or Data Visualization?"
        )
    if any(term in q for term in ("full stack", "web development", "html", "css", "javascript", "react", "frontend", "backend")):
        return (
            "🌐 **Welcome to the Full Stack Web Development learning path!**\n\n"
            "Full Stack Web Development teaches you how to build complete websites and web applications, "
            "including the frontend, backend, and database.\n\n"
            "**Beginner roadmap**\n"
            "1. HTML and CSS for webpage structure and styling\n"
            "2. JavaScript fundamentals for interactive websites\n"
            "3. Responsive web design for mobile and desktop\n"
            "4. React.js for frontend development\n"
            "5. Python Flask or Django for backend development\n"
            "6. SQL and databases for storing application data\n"
            "7. REST APIs, authentication, and web security\n"
            "8. Website deployment and hosting\n\n"
            "**Hands-on project:** build a student management website with a frontend, backend, and database, "
            "then deploy it online.\n\n"
            "Would you like to begin with HTML, CSS, or JavaScript?"
        )
    if any(term in q for term in ("fee", "fees", "price", "cost of course", "duration", "certificate", "schedule", "timing")):
        return (
            "I can explain the technical learning path, but I don't have verified NPTech course fees, official duration, "
            "certificate details, or class schedules. Please confirm those with the NPTech team so you get accurate information.\n\n"
            "Meanwhile, would you like to see what you can learn in Data Science or Full Stack Web Development?"
        )
    if any(term in q for term in ("hello", "hi", "hey", "welcome", "what's up", "whats up", "start")):
        return (
            "Hey there! 🚀 I'm tom, your NPTech learning buddy. I'm ready to help you explore the world of Data Science and automation.\n\n"
            "Pick a path to get started:\n☁️ Data Science — learn how to use AWS and deploy applications.\n"
            "⚙️Full Stack Web Develpment — learn Git, Docker, CI/CD, and automation.\n\nWhich adventure sounds good today?"
        )
    return (
        "I'm tom, your NPTech learning buddy! In this local preview, I can guide you through Data Science, AWS basics, "
        "Full Stack Web Development, Git, Docker, CI/CD, Kubernetes, Terraform, and beginner projects.\n\n"
        "Try asking: **'Show me the Data Science roadmap'** or **'How do I start learning Full Stack Web Developement?'**\n\n"
        "This is a guided demo reply. To enable open-ended AI answers, switch off demo mode and configure Amazon Bedrock as described in the beginner guide."
    )


@application.get("/")
def home():
    return render_template("index.html", demo_mode=DEMO_MODE)


@application.get("/health")
def health():
    return jsonify({"status": "ok", "application": "edutechx-course-chatbot", "mode": "demo" if DEMO_MODE else "bedrock"})


@application.post("/chat")
def chat():
    payload = request.get_json(silent=True) or {}
    raw_messages = payload.get("messages", [])

    if not isinstance(raw_messages, list) or not raw_messages:
        return jsonify({"error": "Send at least one message."}), 400

    # Validate and cap chat history to limit accidental oversized requests.
    messages = []
    for item in raw_messages[-10:]:
        if not isinstance(item, dict):
            continue
        role, content = item.get("role"), item.get("content")
        if role not in {"user", "assistant"} or not isinstance(content, str):
            continue
        content = content.strip()
        if content and len(content) <= 2000:
            messages.append({"role": role, "content": content})

    if not messages or messages[0]["role"] != "user" or messages[-1]["role"] != "user":
        return jsonify({"error": "Conversation history is invalid. Please start a new chat."}), 400
    for previous, current in zip(messages, messages[1:]):
        if previous["role"] == current["role"]:
            return jsonify({"error": "Conversation history is invalid. Please start a new chat."}), 400

    if DEMO_MODE:
        return jsonify({"reply": demo_reply(messages[-1]["content"]), "mode": "demo"})

    bedrock_messages = [
        {"role": item["role"], "content": [{"text": item["content"]}]}
        for item in messages
    ]
    try:
        response = get_bedrock_client().converse(
            modelId=MODEL_ID,
            system=[{"text": SYSTEM_PROMPT}],
            messages=bedrock_messages,
            inferenceConfig={"maxTokens": 700, "temperature": 0.5, "topP": 0.9},
        )
        blocks = response.get("output", {}).get("message", {}).get("content", [])
        reply = "\n".join(block["text"] for block in blocks if isinstance(block, dict) and "text" in block).strip()
        if not reply:
            logger.warning("Bedrock returned an empty response")
            return jsonify({"error": "The AI returned an empty response. Please try again."}), 502
        return jsonify({"reply": reply, "mode": "bedrock"})
    except (ClientError, BotoCoreError) as exc:
        logger.exception("Amazon Bedrock request failed: %s", exc)
        return jsonify({
            "error": "The AI service could not answer. Check the AWS Region, model availability, IAM permission "
                     "bedrock:InvokeModel, and model access, then try again."
        }), 502
    except Exception:
        logger.exception("Unexpected chatbot error")
        return jsonify({"error": "An unexpected server error occurred. Check the server logs."}), 500


if __name__ == "__main__":
    # Development only. Gunicorn runs the app on Elastic Beanstalk.
    application.run(host="127.0.0.1", port=5000, debug=False)
