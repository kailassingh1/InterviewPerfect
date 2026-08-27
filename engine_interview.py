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
        res = groq_client.audio.transcriptions.create(
            file=("candidate_voice.ogg", audio_bytes, "audio/ogg"),
            model="whisper-large-v3"
        )
        return res.text

    @staticmethod
    def evaluate_turn(role: str, question: str, candidate_answer: str, turn_index: int) -> dict:
        prompt = f"""
        Role: You are an expert Lead Technical Interviewer conducting a mock interview for the position of: '{role}'.
        Question Asked: "{question}"
        Candidate's Spoken Answer: "{candidate_answer}"
        Current Question Number: {turn_index} of 5.

        Evaluate strictly and return JSON with keys:
        1. "score": Numeric rating out of 10.
        2. "technical_gaps": Brief bullet points of missing concepts.
        3. "communication_feedback": Grammar corrections or filler words.
        4. "exemplary_response": A concise, high-impact STAR answer.
        5. "next_question": Next question for this role (or 'CONCLUDE' if turn_index == 5).
        6. "spoken_summary": 2-3 spoken sentences combining quick feedback and the next question.
        """
        response = gemini_client.models.generate_content(
            model="gemini-3.6-flash",
            contents=prompt,
            config={"response_mime_type": "application/json"}
        )
        return json.loads(response.text)

    @staticmethod
    async def synthesize_voice(text: str, output_path: str = "interviewer_voice.mp3") -> str:
        """Attempts edge-tts first; seamlessly falls back to gTTS if cloud IPs are blocked."""
        try:
            comm = edge_tts.Communicate(text=text, voice="en-US-GuyNeural")
            await comm.save(output_path)
            return output_path
        except Exception as e:
            print(f"[WARN] Edge-TTS failed ({e}), falling back to gTTS...")
            tts = gTTS(text=text, lang="en", tld="com")
            tts.save(output_path)
            return output_path