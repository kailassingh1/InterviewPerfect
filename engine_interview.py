import os
import json
from google import genai
from groq import Groq
import edge_tts
from gtts import gTTS

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

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
                f"You are a strict, formal Consular Visa Officer interviewing an applicant for: '{role}'. "
                f"Evaluate intent to return, financial stability, clarity, and conciseness."
            )
        else:
            persona = (
                f"You are an exacting Senior Technical Interviewer assessing a candidate for: '{role}'. "
                f"Evaluate technical depth, architectural trade-offs, and communication clarity."
            )

        system_instruction = f"""
{persona}
Current Question {turn_index} of 5: "{question}"
Candidate's Spoken Answer: "{candidate_answer}"

Rules for Evaluation:
1. If the candidate says "I don't know", gives an irrelevant answer, or passes, score it 1-3/10, explain why constructively, and supply the ideal answer.
2. If the candidate gives a good answer, score accurately from 7-10/10.
3. If Question Number is 5 (turn_index == 5), set "next_question" to "CONCLUDE". Otherwise, generate a fresh, dynamic follow-up question.
4. "spoken_summary" must be a concise, realistic 2-sentence feedback snippet for the audio voice note.

You MUST respond strictly with a valid JSON object matching this schema:
{{
    "score": 7,
    "technical_gaps": "Specific missing points or feedback.",
    "communication_feedback": "Tone, confidence, and language critique.",
    "exemplary_response": "The model answer the candidate should give.",
    "next_question": "Next dynamic question or CONCLUDE",
    "spoken_summary": "Spoken feedback to read aloud."
}}
"""

        # Strategy 1: Google Gemini 3.6 Flash
        try:
            response = gemini_client.models.generate_content(
                model="gemini-3.6-flash",
                contents=system_instruction,
                config={"response_mime_type": "application/json"}
            )
            parsed = json.loads(response.text)
            print(f"[DEBUG] Gemini evaluation succeeded for turn {turn_index}")
            return parsed
        except Exception as gemini_err:
            print(f"[WARN] Gemini failed ({gemini_err}), switching to Groq Llama 3.1 8B fallback...")

        # Strategy 2: Groq Llama 3.1 8B Instant (Ultra-fast & active model)
        try:
            chat_completion = groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": "You are a professional mock interviewer that always outputs raw JSON."},
                    {"role": "user", "content": system_instruction}
                ],
                response_format={"type": "json_object"},
                temperature=0.6
            )
            raw_content = chat_completion.choices[0].message.content
            parsed = json.loads(raw_content)
            print(f"[DEBUG] Groq evaluation succeeded for turn {turn_index}")
            return parsed
        except Exception as groq_err:
            print(f"[ERROR] Both LLMs failed: {groq_err}")

        # Strategy 3: Dynamic fallback
        next_q = "CONCLUDE" if turn_index >= 5 else f"Let's move to the next area. Can you describe how you handle production incidents in {role}?"
        return {
            "score": 3 if "know" in candidate_answer.lower() else 6,
            "technical_gaps": f"The answer did not adequately address: {question}",
            "communication_feedback": "Structure your answers with the STAR method.",
            "exemplary_response": f"When answering {question}, clearly explain your strategy and measurable outcomes.",
            "next_question": next_q,
            "spoken_summary": f"Let's keep moving. {next_q}"
        }

    @staticmethod
    async def synthesize_voice(text: str, output_path: str = "interviewer_voice.mp3") -> str:
        try:
            comm = edge_tts.Communicate(text=text, voice="en-US-GuyNeural")
            await comm.save(output_path)
            return output_path
        except Exception as e:
            print(f"[WARN] Edge-TTS failed ({e}), falling back to gTTS...")
            tts = gTTS(text=text, lang="en", tld="com")
            tts.save(output_path)
            return output_path