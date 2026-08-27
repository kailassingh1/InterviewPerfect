import os
import httpx
import traceback
from fastapi import FastAPI, Request
from engine_interview import MockInterviewEngine
from supabase import create_client

app = FastAPI()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_INTERVIEW_BOT_TOKEN")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

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

    try:
        # 1. Start Command
        if "text" in msg and msg["text"].startswith("/start"):
            welcome_text = (
                "🎯 Welcome to AI Mock Interviewer!\n\n"
                "Prepare for your technical rounds with real-time feedback.\n\n"
                "To begin, send your target role:\n"
                "/role Linux / Cloud Engineer\n"
                "/role Software Engineer"
            )
            await send_telegram_text(chat_id, welcome_text)
            return {"status": "ok"}

        # 2. Select Role (Sends Question 1 in Text + Voice)
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
                f"🎯 Interview Started: {role}\n\n"
                f"Question 1/5:\n{first_q}\n\n"
                f"👉 Listen to the audio below and send a voice note answering this question."
            )
            # Send text question
            await send_telegram_text(chat_id, prompt_msg)

            # Send voice note for Question 1
            spoken_first_q = f"Welcome. Here is your first question: {first_q}"
            audio_file = await MockInterviewEngine.synthesize_voice(spoken_first_q, "q1_voice.mp3")
            await send_telegram_voice(chat_id, audio_file)
            return {"status": "ok"}

        # 3. Handle Voice Answer
        if "voice" in msg:
            print(f"[DEBUG] Received voice note from {chat_id}")
            
            # Fetch active session
            session_res = supabase.table("interview_sessions").select("*")\
                .eq("platform_user_id", chat_id)\
                .eq("is_completed", False)\
                .order("created_at", desc=True)\
                .limit(1).execute()

            if not session_res.data:
                await send_telegram_text(chat_id, "No active session found. Please type /start and select your role first.")
                return {"status": "no_session"}

            session = session_res.data[0]
            session_id = session["id"]
            role = session["target_role"]
            current_q = session["current_question_text"]
            q_idx = session["current_question_index"]

            # Download audio file
            file_id = msg["voice"]["file_id"]
            async with httpx.AsyncClient() as client:
                file_info = (await client.get(f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getFile?file_id={file_id}")).json()
                file_path = file_info["result"]["file_path"]
                audio_bytes = (await client.get(f"https://api.telegram.org/file/bot{TELEGRAM_BOT_TOKEN}/{file_path}")).content

            print("[DEBUG] Transcribing audio with Groq...")
            candidate_transcript = MockInterviewEngine.transcribe(audio_bytes)

            print("[DEBUG] Evaluating with Gemini...")
            eval_data = MockInterviewEngine.evaluate_turn(role, current_q, candidate_transcript, q_idx)

            feedback_text = (
                f"📊 Score: {eval_data.get('score', 0)}/10\n\n"
                f"🔍 Technical Assessment:\n{eval_data.get('technical_gaps', 'N/A')}\n\n"
                f"🗣️ Communication & Grammar:\n{eval_data.get('communication_feedback', 'N/A')}\n\n"
                f"💡 Exemplary Response (STAR):\n{eval_data.get('exemplary_response', 'N/A')}\n"
                f"───────────────────────────\n"
            )

            is_final = (eval_data.get("next_question") == "CONCLUDE" or q_idx >= 5)

            if not is_final:
                next_q = eval_data["next_question"]
                feedback_text += f"👉 Question {q_idx + 1}/5:\n{next_q}\n\n(Listen to the voice note below and send your answer)"
                supabase.table("interview_sessions").update({
                    "current_question_index": q_idx + 1,
                    "current_question_text": next_q
                }).eq("id", session_id).execute()
            else:
                feedback_text += "🎉 Interview Completed! Great job practicing today."
                supabase.table("interview_sessions").update({"is_completed": True}).eq("id", session_id).execute()

            # 1. Send detailed text feedback
            await send_telegram_text(chat_id, feedback_text)

            # 2. Send interviewer voice note (Feedback + Next Question)
            try:
                spoken_text = eval_data.get("spoken_summary", "Good job on this answer.")
                if not is_final:
                    spoken_text += f" Here is question {q_idx + 1}: {eval_data['next_question']}"
                
                audio_file = await MockInterviewEngine.synthesize_voice(spoken_text, f"turn_{q_idx}_voice.mp3")
                await send_telegram_voice(chat_id, audio_file)
            except Exception as voice_err:
                print(f"[WARN] Voice synthesis error (non-fatal): {voice_err}")

    except Exception as e:
        print(f"[ERROR] Exception during processing: {str(e)}")
        traceback.print_exc()
        await send_telegram_text(chat_id, "⚠️ An error occurred processing your answer. Please try again.")

    return {"status": "ok"}

async def send_telegram_text(chat_id: str, text: str):
    async with httpx.AsyncClient() as client:
        await client.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            json={"chat_id": chat_id, "text": text}
        )

async def send_telegram_voice(chat_id: str, file_path: str):
    async with httpx.AsyncClient() as client:
        with open(file_path, "rb") as f:
            await client.post(
                f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendVoice",
                data={"chat_id": chat_id},
                files={"voice": f}
            )