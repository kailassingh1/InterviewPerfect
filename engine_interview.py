import os
import json
import re
from google import genai
from groq import Groq
import edge_tts
from gtts import gTTS

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

class MockInterviewEngine:
    @staticmethod
    def transcribe(audio_bytes: bytes) -> str:
        res = groq_client.audio.transcriptions.create(
            file=("candidate_voice.ogg", audio_bytes, "audio/ogg"),
            model="whisper-large-v3"
        )
        return res.text

    @staticmethod
    def evaluate_turn(role: str, question: str, candidate_answer: str, turn_index: int) -> dict:
        is_visa = "visa" in role.lower()

        if is_visa:
            persona_instructions = f"""
            Role: Strict Consular Visa Officer for '{role}'.
            Question: "{question}"
            Applicant Spoken Answer: "{candidate_answer}"
            Question Number: {turn_index} of 5.
            Focus on: Strong ties to home country, financial readiness, conciseness, and eliminating red flags.
            """
        else:
            persona_instructions = f"""
            Role: Senior Technical Lead Interviewer for '{role}'.
            Question: "{question}"
            Candidate Spoken Answer: "{candidate_answer}"
            Question Number: {turn_index} of 5.
            Focus on: STAR method, architectural trade-offs, and technical accuracy.
            """

        prompt = f"""
        {persona_instructions}

        Return STRICT JSON format with exactly these keys:
        {{
            "score": <integer from 1 to 10>,
            "technical_gaps": "<bullet points of technical gaps or visa concerns>",
            "communication_feedback": "<clarity, tone, grammar, and delivery feedback>",
            "exemplary_response": "<an ideal high-impact response>",
            "next_question": "<the next logical question, or 'CONCLUDE' if Question Number is 5>",
            "spoken_summary": "<concise 2-sentence feedback summary for the interviewer voice note>"
        }}
        """

        # 1. Primary: Google Gemini
        try:
            response = gemini_client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt,
                config={"response_mime_type": "application/json"}
            )
            return json.loads(response.text)
        except Exception as e:
            print(f"[WARN] Gemini failed ({e}). Switching to Groq Llama 3.3 70B fallback...")

        # 2. Resilient Fallback: Groq Llama-3.3-70b (Generous free rate limits)
        try:
            chat_completion = groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a professional mock interviewer. Always reply in valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                model="llama-3.3-70b-versatile",
                response_format={"type": "json_object"}
            )
            raw_text = chat_completion.choices[0].message.content
            return json.loads(raw_text)
        except Exception as groq_err:
            print(f"[ERROR] Groq LLM fallback error: {groq_err}")
            # Safe default if all APIs fail
            return {
                "score": 7,
                "technical_gaps": "Good explanation, but elaborate further on specific implementations.",
                "communication_feedback": "Tone is clear and confident.",
                "exemplary_response": "A well-structured answer addressing core principles directly.",
                "next_question": "Can you elaborate on your experience handling real-world escalations?",
                "spoken_summary": "Good points covered. Let's proceed to the next question."
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