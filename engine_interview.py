import os
import json
from google import genai
from groq import Groq
import edge_tts
from gtts import gTTS

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Cache active Groq chat model
CACHED_GROQ_MODEL = None

def get_active_groq_model() -> str:
    global CACHED_GROQ_MODEL
    if CACHED_GROQ_MODEL:
        return CACHED_GROQ_MODEL
    try:
        models = groq_client.models.list()
        available_ids = [m.id for m in models.data]
        # Prefer available high-speed chat models
        for preferred in [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "mixtral-8x7b-32768",
            "qwen/qwen3.6-27b",
            "openai/gpt-oss-20b"
        ]:
            if preferred in available_ids:
                CACHED_GROQ_MODEL = preferred
                print(f"[DEBUG] Selected active Groq model: {CACHED_GROQ_MODEL}")
                return CACHED_GROQ_MODEL
        CACHED_GROQ_MODEL = available_ids[0]
        return CACHED_GROQ_MODEL
    except Exception as e:
        print(f"[WARN] Failed fetching Groq model list ({e}), using default fallback")
        return "llama-3.3-70b-versatile"

class MockInterviewEngine:
    @staticmethod
    def transcribe(audio_bytes: bytes) -> str:
        try:
            res = groq_client.audio.transcriptions.create(
                file=("candidate_voice.ogg", audio_bytes, "audio/ogg"),
                model="whisper-large-v3"
            )
            text = res.text.strip()
            return text if text else "I do not know."
        except Exception as e:
            print(f"[ERROR] STT Transcription failed: {e}")
            return "I do not know."

    @staticmethod
    def evaluate_turn(role: str, question: str, candidate_answer: str, turn_index: int) -> dict:
        is_visa = "visa" in role.lower()

        if is_visa:
            persona = (
                f"You are a strict Consular Visa Officer interviewing an applicant for: '{role}'. "
                f"Focus on: Intent to return, financial stability, brevity, and spotting red flags."
            )
        else:
            persona = (
                f"You are a Senior Technical Lead Interviewer evaluating a candidate for: '{role}'. "
                f"Focus on: Technical accuracy, system architecture, and STAR format."
            )

        system_instruction = f"""
{persona}
Question Asked: "{question}"
Candidate Answer: "{candidate_answer}"
Current Turn: {turn_index} of 5.

Instructions:
1. If the candidate says "I don't know" or gives an incomplete answer, grade 1-3/10 and give actionable guidance.
2. If strong, grade 7-10/10.
3. If Turn is 5 (turn_index == 5), set "next_question" to "CONCLUDE". Otherwise, generate a completely unique, context-aware question.
4. "spoken_summary": Exactly 1-2 spoken sentences evaluating the candidate's answer directly.

Return ONLY a JSON object matching this schema:
{{
    "score": 7,
    "technical_gaps": "Bullet points of missing details or concerns.",
    "communication_feedback": "Critique on confidence, tone, and delivery.",
    "exemplary_response": "The model answer the applicant should have stated.",
    "next_question": "Next dynamic question or CONCLUDE",
    "spoken_summary": "Your previous answer was clear on funding, but lacked details on your intent to return."
}}
"""

        # 1. Primary: Groq (High daily rate-limit)
        try:
            model_id = get_active_groq_model()
            chat_completion = groq_client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": "You are a professional mock interviewer that always outputs raw JSON."},
                    {"role": "user", "content": system_instruction}
                ],
                response_format={"type": "json_object"},
                temperature=0.6
            )
            raw = chat_completion.choices[0].message.content
            parsed = json.loads(raw)
            print(f"[DEBUG] Groq LLM succeeded for turn {turn_index}")
            return parsed
        except Exception as groq_err:
            print(f"[WARN] Groq LLM failed ({groq_err}), switching to Gemini...")

        # 2. Secondary: Gemini Fallback
        try:
            response = gemini_client.models.generate_content(
                model="gemini-2.0-flash",
                contents=system_instruction,
                config={"response_mime_type": "application/json"}
            )
            parsed = json.loads(response.text)
            print(f"[DEBUG] Gemini succeeded for turn {turn_index}")
            return parsed
        except Exception as gemini_err:
            print(f"[ERROR] Both LLMs failed: {gemini_err}")

        # 3. Dynamic Fallback
        visa_bank = [
            "Who is funding your trip, and what is their annual income source?",
            "What assets or family ties ensure you will return to your home country?",
            "What will you do if your visa application is denied today?"
        ]
        tech_bank = [
            "How do you handle zero-downtime database migrations in production?",
            "Can you explain a situation where you had to debug a memory leak?",
            "How do you design an API to prevent race conditions during high load?"
        ]
        bank = visa_bank if is_visa else tech_bank
        fallback_q = bank[(turn_index - 1) % len(bank)] if turn_index < 5 else "CONCLUDE"

        return {
            "score": 3 if "know" in candidate_answer.lower() else 6,
            "technical_gaps": f"The answer did not provide enough concrete proof for: '{question}'",
            "communication_feedback": "Be direct, concise, and avoid filler hesitations.",
            "exemplary_response": f"When addressing '{question}', state your primary point directly in 2 clear sentences.",
            "next_question": fallback_q,
            "spoken_summary": "Your last response needed more concrete details and confidence."
        }

    @staticmethod
    async def synthesize_voice(text: str, output_path: str = "interviewer_voice.mp3") -> str:
        try:
            comm = edge_tts.Communicate(text=text, voice="en-US-GuyNeural")
            await comm.save(output_path)
            return output_path
        except Exception as e:
            print(f"[WARN] Edge-TTS fallback: {e}")
            tts = gTTS(text=text, lang="en", tld="com")
            tts.save(output_path)
            return output_path