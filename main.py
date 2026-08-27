import os
import httpx
from fastapi import FastAPI, Request
from engine_interview import MockInterviewEngine
from supabase import create_client

app = FastAPI()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_INTERVIEW_BOT_TOKEN")
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

FIRST_QUESTION_PROMPT = {
    "Linux / Cloud Engineer": "Can you explain how you would troubleshoot a server experiencing high load average while CPU utilization remains below 10%?",
    "Software Engineer": "Walk me through how you design and implement idempotency in a payment processing API.",
    "General": "Tell me about a challenging technical incident you led and resolved recently."
}

@app.get("/")
def health_check():
    return {"status": "AI Mock Interviewer Bot is running!"}

@app.post(f"/webhook/telegram/interview/{TELEGRAM_BOT_TOKEN}")
async def telegram_interview_handler(request: Request):
    data = await request.json()
    if "message" not in data:
        return {"status": "ignored"}

    msg = data["message"]
    chat_id = str(msg["chat"]["id"])

    # 1. Start Command
    if "text" in msg and msg["text"].startswith("/start"):
        welcome_text = (
            "🎯 *Welcome to AI Mock Interviewer!*\n\n"
            "Prepare for your technical rounds with real-time feedback.\n\n"
            "To begin, send your role:\n"
            "`/role Linux / Cloud Engineer`\n"
            "`/role Software Engineer`"
        )
        await send_telegram_text(chat_id, welcome_text)
        return {"status": "ok"}

    # 2. Select Role
    if "text" in msg and msg["text"].startswith("/role"):
        role = msg["text"].replace("/role", "").strip() or "Linux / Cloud Engineer"
        first_q = FIRST_QUESTION_PROMPT.get(role, FIRST_QUESTION_PROMPT["General"])

        supabase.table("interview_sessions").insert({
            "platform": "telegram",
            "platform_user_id": chat_id,
            "target_role": role,
            "current_question_index": 1,
            "current_question_text": first_q
        }).execute()

        prompt_msg = (
            f"🎯 *Interview Started: {role}*\n\n"
            f"*Question 1/5:*\n_{first_q}_\n\n"
            f"👉 *Hold the mic button and send a voice note answering this question.*"
        )
        await send_telegram_text(chat_id, prompt_msg)
        return {"status": "ok"}

    # 3. Handle Voice Answer
    if "voice" in msg:
        session_res = supabase.table("interview_sessions").select("*")\
            .eq("platform_user_id", chat_id)\
            .eq("is_completed", False)\
            .order("created_at", desc=True)\
            .limit(1).execute()

        if not session_res.data:
            await send_telegram_text(chat_id, "Please type `/start` to begin an interview.")
            return {"status": "no_session"}

        session = session_res.data[0]
        session_id = session["id"]
        role = session["target_role"]
        current_q = session["current_question_text"]
        q_idx = session["current_question_index"]

        file_id = msg["voice"]["file_id"]
        async with httpx.AsyncClient() as client:
            file_info = (await client.get(f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getFile?file_id={file_id}")).json()
            file_path = file_info["result"]["file_path"]
            audio_bytes = (await client.get(f"https://api.telegram.org/file/bot{TELEGRAM_BOT_TOKEN}/{file_path}")).content

        candidate_transcript = MockInterviewEngine.transcribe(audio_bytes)
        eval_data = MockInterviewEngine.evaluate_turn(role, current_q, candidate_transcript, q_idx)

        feedback_text = (
            f"📊 *Score:* {eval_data['score']}/10\n\n"
            f"🔍 *Technical Assessment:*\n{eval_data['technical_gaps']}\n\n"
            f"🗣️ *Communication & Grammar:*\n{eval_data['communication_feedback']}\n\n"
            f"💡 *Exemplary Response (STAR):*\n_{eval_data['exemplary_response']}_\n"
            f"───────────────────────────\n"
        )

        if eval_data["next_question"] != "CONCLUDE" and q_idx < 5:
            feedback_text += f"👉 *Question {q_idx + 1}/5:*\n_{eval_data['next_question']}_\n\n_(Send your voice note answer)_"
            supabase.table("interview_sessions").update({
                "current_question_index": q_idx + 1,
                "current_question_text": eval_data["next_question"]
            }).eq("id", session_id).execute()
        else:
            feedback_text += "🎉 *Interview Completed!* Great job practicing today."
            supabase.table("interview_sessions").update({"is_completed": True}).eq("id", session_id).execute()

        await send_telegram_text(chat_id, feedback_text)

        audio_file = await MockInterviewEngine.synthesize_voice(eval_data["spoken_summary"])
        await send_telegram_voice(chat_id, audio_file)

    return {"status": "ok"}

async def send_telegram_text(chat_id: str, text: str):
    async with httpx.AsyncClient() as client:
        await client.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            json={"chat_id": chat_id, "text": text, "parse_mode": "Markdown"}
        )

async def send_telegram_voice(chat_id: str, file_path: str):
    async with httpx.AsyncClient() as client:
        with open(file_path, "rb") as f:
            await client.post(
                f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendVoice",
                data={"chat_id": chat_id},
                files={"voice": f}
            )