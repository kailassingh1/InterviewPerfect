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

# Role Catalog & Opening Questions
ROLE_CATALOG = {
    # Tech Roles
    "1": ("Linux & Cloud Engineer", "Can you explain how you would troubleshoot a server experiencing high load average while CPU utilization remains below 10%?"),
    "2": ("DevOps / SRE", "How do you design a zero-downtime CI/CD deployment pipeline for a high-traffic microservices architecture?"),
    "3": ("Python Backend Developer", "How do you handle database connection pooling and asynchronous task queues under heavy concurrency?"),
    "4": ("Data Analyst / SQL", "Walk me through how you would optimize a slow-running SQL query joining three multi-million row tables."),
    
    # Visa Interview Roles
    "5": ("US F-1 Student Visa", "Why did you choose this specific university and course instead of studying in your home country?"),
    "6": ("US B1/B2 Tourist/Business Visa", "What is the specific purpose of your visit to the United States, and how long do you plan to stay?"),
    "7": ("US H-1B Work Visa", "Can you explain your job title, day-to-day responsibilities, and how your degree directly aligns with this specialized role?"),
    "8": ("UK / Schengen Student Visa", "How are you financing your studies and living expenses for the entire duration of your stay?")
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
        # 1. Start & Help Menu
        if "text" in msg and (msg["text"].startswith("/start") or msg["text"].startswith("/roles")):
            menu_text = (
                "🎯 *Welcome to AI Mock Interviewer!*\n\n"
                "Select the track you want to practice for by replying with the *number*:\n\n"
                "💼 *Technical & Corporate Roles:*\n"
                "1️⃣ Linux & Cloud Engineer\n"
                "2️⃣ DevOps / SRE\n"
                "3️⃣ Python Backend Developer\n"
                "4️⃣ Data Analyst / SQL\n\n"
                "🛂 *Visa Officer Interviews:*\n"
                "5️⃣ US F-1 Student Visa\n"
                "6️⃣ US B1/B2 Tourist/Business Visa\n"
                "7️⃣ US H-1B Work Visa\n"
                "8️⃣ UK / Schengen Student Visa\n\n"
                "👉 *Reply with 1 to 8* to start your mock round."
            )
            await send_telegram_text(chat_id, menu_text)
            return {"status": "ok"}

        # 2. Number Selection (1 to 8) or Custom /role Command
        if "text" in msg:
            user_input = msg["text"].strip()
            selected_role = None
            first_q = None

            if user_input in ROLE_CATALOG:
                selected_role, first_q = ROLE_CATALOG[user_input]
            elif user_input.startswith("/role"):
                selected_role = user_input.replace("/role", "").strip() or "General Technical"
                first_q = f"Tell me about your background and why you are applying for the {selected_role} position."

            if selected_role:
                # Save session
                supabase.table("interview_sessions").insert({
                    "platform": "telegram",
                    "platform_user_id": chat_id,
                    "target_role": selected_role,
                    "current_question_index": 1,
                    "current_question_text": first_q
                }).execute()

                is_visa = "visa" in selected_role.lower()
                icon = "🛂" if is_visa else "💼"

                prompt_msg = (
                    f"{icon} *Session Started: {selected_role}*\n\n"
                    f"*Question 1/5:*\n{first_q}\n\n"
                    f"👉 *Hold the mic and send a 30-45s voice note answering this question.*"
                )
                await send_telegram_text(chat_id, prompt_msg)

                # Send Question 1 Audio Voice Note
                spoken_prefix = "Visa Officer speaking." if is_visa else "Interviewer speaking."
                audio_file = await MockInterviewEngine.synthesize_voice(
                    f"{spoken_prefix} Here is your first question: {first_q}", 
                    "q1_voice.mp3"
                )
                await send_telegram_voice(chat_id, audio_file)
                return {"status": "ok"}

        # 3. Handle Voice Answer
        if "voice" in msg:
            print(f"[DEBUG] Received voice answer from {chat_id}")

            session_res = supabase.table("interview_sessions").select("*")\
                .eq("platform_user_id", chat_id)\
                .eq("is_completed", False)\
                .order("created_at", desc=True)\
                .limit(1).execute()

            if not session_res.data:
                await send_telegram_text(chat_id, "No active session found. Please type /roles to pick an interview stream.")
                return {"status": "no_session"}

            session = session_res.data[0]
            session_id = session["id"]
            role = session["target_role"]
            current_q = session["current_question_text"]
            q_idx = session["current_question_index"]
            is_visa = "visa" in role.lower()

            # Download audio
            file_id = msg["voice"]["file_id"]
            async with httpx.AsyncClient() as client:
                file_info = (await client.get(f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getFile?file_id={file_id}")).json()
                file_path = file_info["result"]["file_path"]
                audio_bytes = (await client.get(f"https://api.telegram.org/file/bot{TELEGRAM_BOT_TOKEN}/{file_path}")).content

            # Transcribe & Evaluate
            candidate_transcript = MockInterviewEngine.transcribe(audio_bytes)
            eval_data = MockInterviewEngine.evaluate_turn(role, current_q, candidate_transcript, q_idx)

            header_label = "🛂 Visa Officer Assessment:" if is_visa else "🔍 Technical Assessment:"
            ideal_label = "💡 Ideal Visa Response (Concise & Clear):" if is_visa else "💡 Exemplary Response (STAR):"

            feedback_text = (
                f"📊 Score: {eval_data.get('score', 0)}/10\n\n"
                f"{header_label}\n{eval_data.get('technical_gaps', 'N/A')}\n\n"
                f"🗣️ Delivery & Tone:\n{eval_data.get('communication_feedback', 'N/A')}\n\n"
                f"{ideal_label}\n{eval_data.get('exemplary_response', 'N/A')}\n"
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
                feedback_text += "🎉 Interview Completed! You have finished all 5 questions for this round."
                supabase.table("interview_sessions").update({"is_completed": True}).eq("id", session_id).execute()

            # Send Feedback Text
            await send_telegram_text(chat_id, feedback_text)

            # Send Voice Note (Feedback + Next Question)
            try:
                spoken_text = eval_data.get("spoken_summary", "Good answer.")
                if not is_final:
                    spoken_text += f" Next question: {eval_data['next_question']}"

                audio_file = await MockInterviewEngine.synthesize_voice(spoken_text, f"turn_{q_idx}_voice.mp3")
                await send_telegram_voice(chat_id, audio_file)
            except Exception as voice_err:
                print(f"[WARN] Voice synthesis error: {voice_err}")

    except Exception as e:
        print(f"[ERROR] Exception: {str(e)}")
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