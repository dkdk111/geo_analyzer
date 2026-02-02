# geo_engine.py
import os
import json
import sqlite3
import datetime
import re
from dotenv import load_dotenv
from openai import OpenAI
import google.generativeai as genai
from concurrent.futures import ThreadPoolExecutor, as_completed

load_dotenv()

class GEOAnalyzer:
    def __init__(self):
        # API 키 로드
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.gpt_model = "gpt-4o-mini"
        
        # Gemini 설정 (gemini-3-flash-preview 모델 강제 고정)
        google_key = os.getenv("GOOGLE_API_KEY")
        if google_key:
            genai.configure(api_key=google_key)
            self.gemini_model = genai.GenerativeModel("gemini-3-flash-preview")
            self.gemini_available = True
        else:
            self.gemini_available = False
        
        # DB 초기화
        self._init_db()

    def _init_db(self):
        """상담 신청 저장을 위한 SQLite 데이터베이스 초기화"""
        with sqlite3.connect("inquiries.db", check_same_thread=False) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS inquiries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    brand TEXT,
                    keyword TEXT,
                    name TEXT,
                    contact TEXT,
                    message TEXT
                )
            """)
            conn.commit()

    def save_inquiry(self, brand, keyword, name, contact, message):
        """컨설팅 신청 데이터를 실제 DB에 저장"""
        with sqlite3.connect("inquiries.db", check_same_thread=False) as conn:
            cursor = conn.cursor()
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cursor.execute("""
                INSERT INTO inquiries (timestamp, brand, keyword, name, contact, message) 
                VALUES (?, ?, ?, ?, ?, ?)
            """, (timestamp, brand, keyword, name, contact, message))
            conn.commit()

    def check_visibility_logic(self, text, brand_name):
        """브랜드명 노출 여부 물리적 매칭 검증"""
        if not text or not brand_name: return False
        def normalize(s):
            return re.sub(r'[^a-zA-Z0-9가-힣]', '', s).lower()
        norm_text = normalize(text)
        norm_brand = normalize(brand_name)
        return norm_brand in norm_text

    def generate_scenarios(self, brand_name, service_keyword):
        """단계 1: 전략적 GEO 쿼리 생성"""
        prompt = f"""
        당신은 전문 GEO(Generative Engine Optimization) 전략가입니다.
        브랜드: {brand_name}, 서비스: {service_keyword}
        
        이 브랜드가 AI 답변에서 노출될 수밖에 없는 'GEO 테스트 질문' 3개를 한국어로 생성하세요.
        - 질문자는 자신의 문제를 해결해줄 곳을 찾는 관점이어야 합니다.
        - 브랜드명은 질문에 절대 포함하지 마십시오.
        JSON 형식: {{"strategy": "타겟팅 전략 설명", "queries": ["질문1", "질문2", "질문3"]}}
        """
        response = self.openai_client.chat.completions.create(
            model=self.gpt_model, messages=[{"role": "system", "content": prompt}], response_format={"type": "json_object"}
        )
        data = json.loads(response.choices[0].message.content)
        return data["queries"], data["strategy"]

    def analyze_task(self, model_type, brand_name, question, q_idx):
        """단계 2: 개별 AI 답변 수집 및 채점"""
        try:
            if model_type == "GPT":
                res = self.openai_client.chat.completions.create(
                    model=self.gpt_model, messages=[{"role": "user", "content": question}]
                )
                raw_text = res.choices[0].message.content
            else:
                if not self.gemini_available: return None
                res = self.gemini_model.generate_content(question)
                raw_text = res.text

            actual_visibility = self.check_visibility_logic(raw_text, brand_name)

            analysis_prompt = f"""
            AI 답변을 한국어로 진단하십시오. 브랜드: {brand_name}. 답변: {raw_text}.
            물리적 노출 여부: {actual_visibility}
            JSON 형식: {{
                "mentioned": bool, 
                "solution_fit": 1~10, 
                "fit_reason": "한국어 근거", 
                "summary": "한국어 요약"
            }}
            """
            analysis_res = self.openai_client.chat.completions.create(
                model=self.gpt_model, messages=[{"role": "system", "content": analysis_prompt}], response_format={"type": "json_object"}
            )
            analysis_data = json.loads(analysis_res.choices[0].message.content)
            analysis_data["mentioned"] = actual_visibility
            return {"q_idx": q_idx, "model_type": model_type, "raw": raw_text, "analysis": analysis_data}
        except Exception as e:
            return {"q_idx": q_idx, "model_type": model_type, "error": str(e)}

    def generate_final_report(self, brand_name, results_json):
        """단계 3: 최종 마케팅 분석 보고서 생성"""
        prompt = f"""
        당신은 전문 GEO 컨설턴트입니다. 진단 데이터({results_json})를 바탕으로 브랜드 '{brand_name}'을 위한 심층 진단 보고서를 한국어로 작성하십시오.
        내용에는 GEO의 정의, 현재 노출 실태, 경쟁력 분석, 최적화 제언이 포함되어야 합니다.
        """
        response = self.openai_client.chat.completions.create(
            model=self.gpt_model, messages=[{"role": "system", "content": prompt}]
        )
        return response.choices[0].message.content