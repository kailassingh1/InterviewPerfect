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
        is_visa = "visa" in role.lower()

        if is_visa:
            persona_instructions = f"""
            Role: You are a strict, formal Consular Visa Officer interviewing an applicant for: '{role}'.
            Question Asked: "{question}"
            Applicant's Spoken Answer: "{candidate_answer}"
            Current Question Number: {turn_index} of 5.

            Visa Grading Criteria:
            - Strong ties to home country & intent to return (crucial).
            - Financial clarity (funding, sponsor credibility).
            - Conciseness (short, confident 2-3 sentence answers without nervous rambling).
            - No contradictory statements.
            """
        else:
            persona_instructions = f"""
            Role: You are a Lead Technical Interviewer evaluating a candidate for: '{role}'.
            Question Asked: "{question}"
            Candidate's Spoken Answer: "{candidate_answer}"
            Current Question Number: {turn_index} of 5.

            Tech Grading Criteria:
            - STAR framework (Situation, Task, Action, Result).
            - Technical accuracy, architecture, and trade-offs.
            - Professional communication and terminology.
            """

        prompt = f"""
        {persona_instructions}

        Evaluate strictly and return JSON with keys:
        1. "score": Numeric rating out of 10.
        2. "technical_gaps": Specific gaps, risks, or red flags spotted in the answer.
        3. "communication_feedback": Tone, confidence markers, fillers, or grammatical fixes.
        4. "exemplary_response": The ideal, confident response the user should have given.
        5. "next_question": The logical next question for this role/visa officer (or 'CONCLUDE' if turn_index == 5).
        6. "spoken_summary": 2-sentence feedback and transition for voice synthesis.
        """

        response = gemini_client.models.generate_content(
            model="gemini-3.6-flash",
            contents=prompt,
            config={"response_mime_type": "application/json"}
        )
        return json.loads(response.text)

    @staticmethod
    async def synthesize_voice(text: str, output_path: str = "interviewer_voice.mp3") -> str:
        try:
            comm = edge_tts.Communicate(text=text, voice="en-US-GuyNeural")
            await comm.save(output_path)
            return output_path
        except Exception as e:
            print(f"[WARN] Edge-TTS fallback triggered: {e}")
            tts = gTTS(text=text, lang="en", tld="com")
            tts.save(output_path)
            return output_path