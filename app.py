# app.py
import streamlit as st
import re
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from geo_engine import GEOAnalyzer

# 페이지 기본 설정
st.set_page_config(page_title="GEO 실시간 통합 진단 시스템", layout="wide")

# 모든 아이콘/SVG 제거 및 폰트/디자인 CSS
st.markdown("""
    <link rel="stylesheet" as="style" crossorigin href="https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/static/pretendard.css" />
    <style>
        * { font-family: 'Pretendard', sans-serif !important; }
        
        /* 모든 아이콘 및 아이콘 텍스트 완전 제거 */
        span[data-testid="stIconMaterial"], 
        button[data-testid="stSidebarCollapseButton"],
        .st-emotion-cache-1v0vdom, 
        .material-icons,
        svg {
            display: none !important;
            visibility: hidden !important;
            width: 0 !important;
            height: 0 !important;
        }

        /* 섹션 구분 헤더 */
        .section-header {
            font-size: 1.6rem;
            font-weight: 800;
            margin-top: 50px;
            margin-bottom: 20px;
            border-bottom: 2px solid #333;
            padding-bottom: 10px;
        }

        /* 쿼리 박스: 검정 배경 / 노란 글씨 */
        .query-box {
            background-color: #000000;
            color: #FFFF00;
            padding: 25px;
            border-radius: 5px;
            margin: 20px 0;
            border: 1px solid #FFFF00;
            font-size: 1.2rem;
            font-weight: 700;
        }
        
        /* 답변 박스 미리 확보 */
        .answer-box {
            height: 400px;
            overflow-y: auto;
            border: 1px solid #ddd;
            padding: 20px;
            margin-top: 10px;
            font-size: 0.95rem;
            line-height: 1.7;
        }

        /* 결과 색상 텍스트 */
        .found { color: #28a745; font-weight: 800; }
        .not-found { color: #dc3545; font-weight: 800; }
    </style>
""", unsafe_allow_html=True)

# 메인 타이틀
st.markdown('<h1 style="text-align: center;">GEO 실시간 통합 진단 시스템</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #666; margin-bottom: 40px;">생성형 엔진 최적화(Generative Engine Optimization) 가시성 및 성과 분석</p>', unsafe_allow_html=True)

if "results" not in st.session_state: st.session_state.results = None
if "report" not in st.session_state: st.session_state.report = None

def highlight(text, brand):
    if not brand: return text
    pattern = re.compile(f"({re.escape(brand)})", re.IGNORECASE)
    return pattern.sub(r'<mark style="background-color: #FFFF00; color: #000; font-weight: bold;">\1</mark>', text)

# 사이드바 입력
with st.sidebar:
    st.markdown("### 진단 정보 입력")
    brand_in = st.text_input("진단 브랜드명", placeholder="예: 토스")
    kw_in = st.text_input("서비스/카테고리", placeholder="예: 간편 송금 서비스")
    st.divider()
    analyze_btn = st.button("실시간 진단 시작", type="primary", use_container_width=True)

# 진단 실행
if analyze_btn and brand_in and kw_in:
    engine = GEOAnalyzer()
    
    # --- 단계 1: 전략 설계 ---
    st.markdown('<div class="section-header">단계 1. 전략적 진단 시나리오 설계</div>', unsafe_allow_html=True)
    with st.spinner("최적화 쿼리를 설계 중입니다..."):
        queries, strategy = engine.generate_scenarios(brand_in, kw_in)
        st.session_state.queries = queries
        st.session_state.target_brand = brand_in
        st.session_state.target_keyword = kw_in
    st.text(f"분석 배경: {strategy}")
    st.divider()

    # --- 단계 2: 교차 검증 (박스 선확보) ---
    st.markdown('<div class="section-header">단계 2. AI 검색 엔진 실시간 교차 검증</div>', unsafe_allow_html=True)
    
    slots = []
    for i, q in enumerate(queries):
        st.markdown(f'<div class="query-box">테스트 질문 {i+1}: {q}</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        
        with col1:
            st.text("분석 엔진: ChatGPT (GPT-4o)")
            gpt_slot = st.empty()
            gpt_slot.markdown('<div class="answer-box">분석 진행 중...</div>', unsafe_allow_html=True)
        with col2:
            st.text("분석 엔진: Gemini (flash-preview)")
            gem_slot = st.empty()
            gem_slot.markdown('<div class="answer-box">분석 진행 중...</div>', unsafe_allow_html=True)
        
        slots.append({"GPT": gpt_slot, "Gemini": gem_slot})

    # 병렬 처리
    all_res = []
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = [executor.submit(engine.analyze_task, m, brand_in, q, i) 
                   for i in range(len(queries)) for m in ["GPT", "Gemini"]]
        
        for future in as_completed(futures):
            res = future.result()
            if res:
                all_res.append(res)
                q_idx, m_type = res["q_idx"], res["model_type"]
                slot = slots[q_idx][m_type]
                
                with slot.container():
                    ana = res["analysis"]
                    # 발견/미발견 색상 적용
                    visibility_html = '<span class="found">발견됨</span>' if ana['mentioned'] else '<span class="not-found">미발견</span>'
                    st.markdown(f"결과: {visibility_html} | 적합도: {ana['solution_fit']}/10", unsafe_allow_html=True)
                    st.markdown(f'<div class="answer-box">{highlight(res["raw"], brand_in)}</div>', unsafe_allow_html=True)
                    st.caption(f"근거: {ana.get('fit_reason', '')}")
    
    st.session_state.results = all_res
    st.divider()

    # --- 단계 3: 진단 보고서 ---
    st.markdown('<div class="section-header">단계 3. 데이터 기반 심층 진단 보고서</div>', unsafe_allow_html=True)
    with st.spinner("최종 분석 리포트를 작성 중입니다..."):
        report = engine.generate_final_report(brand_in, json.dumps(all_res))
        st.session_state.report = report
    
    st.markdown(st.session_state.report)
    st.divider()

# 결과가 세션에 있을 때만 신청 폼 표시
if st.session_state.report:
    # --- 단계 4: 상담 신청 ---
    st.markdown('<div class="section-header">단계 4. GEO 전략 컨설팅 신청</div>', unsafe_allow_html=True)
    st.text("본 진단 결과는 귀사의 브랜드가 AI 검색 시장에서 마주한 실제 가시성입니다.")
    
    with st.form("inquiry_form"):
        name = st.text_input("성함 / 기업명")
        contact = st.text_input("연락처 (이메일 혹은 휴대전화)")
        msg = st.text_area("상담 희망 내용")
        if st.form_submit_button("상담 신청하기", type="primary", use_container_width=True):
            if name and contact:
                # DB 저장 로직 호출
                engine = GEOAnalyzer()
                engine.save_inquiry(
                    st.session_state.target_brand, 
                    st.session_state.target_keyword, 
                    name, contact, msg
                )
                st.success("상담 신청이 완료되었습니다. 담당 전문가가 곧 연락드리겠습니다.")
            else:
                st.error("필수 정보를 입력해주세요.")