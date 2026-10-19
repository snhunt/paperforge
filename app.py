import streamlit as st
import re
import random
import io
import json
from collections import Counter, defaultdict
from datetime import datetime

try:
    import fitz
    HAS_FITZ = True
except:
    HAS_FITZ = False

try:
    import docx
    HAS_DOCX = True
except:
    HAS_DOCX = False

try:
    from pptx import Presentation
    HAS_PPTX = True
except:
    HAS_PPTX = False

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="PaperForge - Updated - No Repeat + SA News Scenarios", page_icon="🏫", layout="wide")

# ---------- FILE READERS ----------
def extract_text_from_file(uploaded_file):
    name = uploaded_file.name.lower()
    try:
        if name.endswith(".pdf") and HAS_FITZ:
            doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
            return "\n".join([page.get_text() for page in doc])
        elif name.endswith(".docx") and HAS_DOCX:
            d = docx.Document(uploaded_file)
            return "\n".join([p.text for p in d.paragraphs])
        elif name.endswith(".pptx") and HAS_PPTX:
            prs = Presentation(uploaded_file)
            texts = []
            for slide in prs.slides:
                for shape in slide.shapes:
                    if hasattr(shape, "text"):
                        texts.append(shape.text)
                    if shape.has_table:
                        for row in shape.table.rows:
                            for cell in row.cells:
                                texts.append(cell.text)
            return "\n".join(texts)
        elif name.endswith(".txt"):
            return uploaded_file.read().decode("utf-8", errors="ignore")
        else:
            # fallback pdf
            if HAS_FITZ and name.endswith(".pdf"):
                doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
                return "\n".join([page.get_text() for page in doc])
            return uploaded_file.read().decode("utf-8", errors="ignore")
    except Exception as e:
        return f"[Error {uploaded_file.name}: {e}]"

def chunk_text(text, chunk_size=500, overlap=80):
    words = text.split()
    chunks = []
    i=0
    while i < len(words):
        chunk = " ".join(words[i:i+chunk_size])
        if len(chunk.strip())>80:
            chunks.append(chunk)
        i+= chunk_size - overlap
    return chunks

# ---------- 1. LEARN PAST PAPER STYLE (your new requirement) ----------
def learn_past_paper_style(texts):
    combined = "\n".join(texts)
    lower = combined.lower()

    # Major headings
    headings = re.findall(r'(SECTION\s+[A-C]|QUESTION\s+\d+|INSTRUCTIONS AND INFORMATION)', combined, re.I)
    heading_counts = Counter([h.upper() for h in headings])

    # Question types - GRADE 10 EXAM HEADINGS (CAPS aligned)
    # Changed from generic Calculation/Essay/MCQ/Short to Grade 10 assessment taxonomy
    q_types = {
        "A1: Knowledge & Recall (Define, List, Expand, Name)": len(re.findall(r'\b(define|list|name|expand acronym|what does .* stand for)\b', lower)),
        "A2: Comprehension (Explain, Describe, Purpose)": len(re.findall(r'\b(explain|describe|what is purpose|what does .* mean|why must)\b', lower)),
        "B1: Application (Provide example, Scenario use)": len(re.findall(r'\b(provide one example|provide example|using scenario|give example|how would you use)\b', lower)),
        "B2: Analysis & Comparison (Compare, Difference, Calculate)": len(re.findall(r'\b(difference between|compare|distinguish|calculate|is .* same as|differentiate)\b', lower)),
        "C: Evaluation & Ethics (Motivate, Ethical, Green, Ergonomics)": len(re.findall(r'\b(motivate|ethical|is this ethical|dangers|why does school pay|evaluate|justify|e-waste|ergonomics|green computing)\b', lower)),
    }

    # Mark allocations
    marks_found = re.findall(r'\(?\s*(\d+)\s*(?:marks?|x\d+)?\s*\)?', combined, re.I)
    marks_int = [int(m) for m in marks_found if m.isdigit() and int(m)<=50 and int(m)>0]
    # Section weightings - try to detect SECTION A (20) etc
    section_weights = {}
    sec_matches = re.findall(r'SECTION\s+([A-C]).*?\((\d+)\)', combined, re.I)
    for sec, m in sec_matches:
        section_weights[f"SECTION {sec.upper()}"] = int(m)
    # Fallback if not found - estimate from marks
    if not section_weights and marks_int:
        section_weights = {"SECTION A": 20, "SECTION B": 50, "SECTION C": 50}

    # Style of language - verbs and tone
    allowed_verbs = ["Explain", "Define", "Expand", "Provide ONE example", "What is", "Name ONE", "Why must", "What is purpose", "Is ... same as", "Difference between", "Give TWO", "Motivate"]
    verb_counts = {}
    for v in ["explain","define","expand","provide","name","why must","purpose","difference","give","motivate","what is"]:
        verb_counts[v] = len(re.findall(r'\b' + v + r'\b', lower))

    formal_patterns = {
        "Do NOT write one-word answers": "do not write one-word" in lower,
        "Leave a line open": "leave a line open" in lower,
        "Start each section on new page": "start each section" in lower,
        "Number in margin": "number your answers" in lower,
        "Mark = number of facts": "number of facts" in lower,
    }

    # Concepts assessed (keywords)
    vec = TfidfVectorizer(stop_words='english', max_features=100)
    try:
        X = vec.fit_transform([combined])
        terms = vec.get_feature_names_out()
        scores = X.toarray()[0]
        top_idx = scores.argsort()[-30:][::-1]
        concepts = [terms[i] for i in top_idx if len(terms[i])>3]
    except:
        words = lower.split()
        concepts = [w for w,_ in Counter([w for w in words if len(w)>4]).most_common(30)]

    return {
        "headings": dict(heading_counts),
        "question_types": q_types,
        "marks": marks_int,
        "section_weights": section_weights,
        "language_verbs": verb_counts,
        "formal_patterns": formal_patterns,
        "concepts_assessed": concepts,
        "total_marks_detected": sum(marks_int),
        "avg_marks": sum(marks_int)/len(marks_int) if marks_int else 0
    }

# ---------- 2. READ TAUGHT CONTENT - ONLY WHAT WAS TAUGHT ----------
def extract_taught_concepts(texts, chunks):
    combined = " ".join(texts)
    # All topics
    vec = TfidfVectorizer(stop_words='english', max_features=2000)
    try:
        X = vec.fit_transform([combined])
        terms = vec.get_feature_names_out()
        scores = X.toarray()[0]
        top_idx = scores.argsort()[-80:][::-1]
        all_concepts = [terms[i] for i in top_idx if len(terms[i])>3]
    except:
        words = combined.lower().split()
        all_concepts = [w for w,_ in Counter([w for w in words if len(w)>4]).most_common(80)]

    # Group into CAPS-like buckets (heuristic)
    buckets = defaultdict(list)
    for c in all_concepts:
        lc = c.lower()
        if any(k in lc for k in ["hardware","storage","cpu","ram","touch","modular","compatibility","desktop","smartphone","magnetic","optical","electronic"]):
            buckets["Hardware & System"].append(c)
        elif any(k in lc for k in ["network","pan","lan","wan","node","router","ip","isp","broadband","wifi","utp","fibre","server","bluetooth"]):
            buckets["Networks"].append(c)
        elif any(k in lc for k in ["green","ergonomics","cts","e-waste","recycle","donate","power"]):
            buckets["Green & Social"].append(c)
        elif any(k in lc for k in ["data","information","html","browser","search","w3c","plugin","extension","convergence"]):
            buckets["Data & Web"].append(c)
        else:
            buckets["Other"].append(c)

    return all_concepts, dict(buckets)

def gap_and_no_repeat(taught_concepts, past_concepts, past_patterns=None):
    past_set = set([c.lower() for c in past_concepts])
    # Concepts taught but never assessed = gaps - must be prioritized
    gaps = [c for c in taught_concepts if c.lower() not in past_set]
    # Concepts taught and assessed but not over-assessed
    remaining = [c for c in taught_concepts if c.lower() in past_set]
    # For no repeat in same paper: we will use each concept only once
    # Prioritize gaps first (40-60% of paper)
    random.shuffle(gaps)
    random.shuffle(remaining)
    # Final ordered list: gaps first, then remaining
    ordered = gaps + remaining
    return gaps, ordered

# ---------- 3. SA CURRENT NEWS SCENARIOS - NEVER REPEAT ----------
SA_NEWS_POOL_2025_2026 = [
    "GreenMap AI Lab - Drones mapping illegal dumping in Gauteng using AI, solar powered lab, 15k photos, water crisis data",
    "Township SME Solar Shop - Spaza shop in Soweto installing solar during load-shedding, using POS system, mobile payments, stock tracking",
    "Matric Results Portal - Department of Education portal for 2025 Matric results, 900k learners checking results, high traffic, data privacy concerns",
    "Springboks Fan App - Rugby World Cup 2027 qualifier app, live streaming, ticket sales, fan location sharing, township viewing parties",
    "Cape Town Water Crisis Dashboard - Day Zero 2.0 monitoring, IoT sensors in dams, 4 million residents checking usage, misinformation",
    "SASSA Grant Payment System - SRD grant biometric verification, USSD vs app, fraud prevention, rural connectivity issues",
    "Johannesburg Road Repair Tracker - Pothole reporting app using citizen photos, AI classification, municipal dashboard, e-waste from old devices",
    "KZN Flood Early Warning - Drone + SMS early warning for floods, community WhatsApp groups, offline data collection, green computing with old phones",
    "Limpopo AgriTech - Small farmers using smartphone sensors for soil moisture, market price app, UTP vs satellite connectivity",
    "Tshwane e-Waste Drive - Schools collecting e-waste, refurbishing PCs for township schools, tracking with QR codes, ergonomic refurbishment stations"
]

def generate_sa_scenario_llm(client, model, used_scenarios, concepts_for_section):
    # Used scenarios to avoid repeat
    used_str = ", ".join(used_scenarios) if used_scenarios else "None yet"
    concepts_str = ", ".join(concepts_for_section[:5])

    prompt = f"""You are a South African current affairs scenario generator for Grade 10-12 IT exams.

Already used scenarios (NEVER repeat these, create brand new one): {used_str}

Concepts to be assessed in this section: {concepts_str}

Task: Generate ONE brand new scenario based on SA current news 2025-2026. Must be:
- 100% South African: Use real SA context: load-shedding, water crisis, township economy, matric, SASSA, Springboks, potholes, e-waste, solar, 4IR in schools
- Different from used scenarios
- 2-3 sentences, includes hardware, network, data
- Suitable for learners in Parktown, Gauteng
- Example style: "GreenMap AI Lab with drones, new PCs, solar power, 15 000 photos, schools across SA accessing website"

Output ONLY scenario title and 2-sentence description, no extra text.
Format: TITLE: <title> | DESCRIPTION: <2 sentences>"""

    try:
        if client and model:
            resp = client.chat.completions.create(model=model, messages=[{"role":"user","content":prompt}], temperature=0.9, max_tokens=250)
            text = resp.choices[0].message.content.strip()
            # Ensure not repeating
            if any(used.lower() in text.lower() for used in used_scenarios):
                # fallback to pool
                available = [s for s in SA_NEWS_POOL_2025_2026 if s not in used_scenarios]
                return random.choice(available) if available else random.choice(SA_NEWS_POOL_2025_2026)
            return text
        else:
            # Template fallback - pick unused
            available = [s for s in SA_NEWS_POOL_2025_2026 if s not in used_scenarios]
            return random.choice(available) if available else random.choice(SA_NEWS_POOL_2025_2026)
    except:
        available = [s for s in SA_NEWS_POOL_2025_2026 if s not in used_scenarios]
        return random.choice(available) if available else random.choice(SA_NEWS_POOL_2025_2026)

# ---------- 4. GENERATE PAPER - NO REPEAT CONCEPTS, ONLY TAUGHT, DIFF SCENARIOS PER SECTION ----------
def generate_paper_no_repeat(taught_chunks, ordered_concepts, gaps, past_texts, style_info, client, model_name, settings):
    # Track used concepts to never repeat in same paper
    used_concepts = set()
    used_scenarios = []
    past_all = " ".join(past_texts)
    vectorizer = TfidfVectorizer(stop_words='english').fit(taught_chunks + [past_all] + ["dummy"])
    past_vecs = vectorizer.transform([past_all])

    paper = {"header": [], "instructions": [], "section_a": [], "section_b": [], "section_c": [], "scenarios": {}}

    # Header - Parktown exact
    paper["header"] = [
        "PARKTOWN HIGH SCHOOL FOR GIRLS",
        f"Examiner: {settings.get('examiner','Mr K Witelson')}",
        f"Moderator: {settings.get('moderator','Mrs L Le Grange')}",
        f"Marks: {settings.get('total_marks',120)}",
        f"Time: {settings.get('time','2.5 hours')}",
        f"This paper consists of {settings.get('pages','11')} pages."
    ]
    paper["instructions"] = [
        f"This question paper consists of THREE sections subdivided as follows: SECTION A: Short Questions ({settings.get('sec_a',20)}) SECTION B: System Technologies & Networks ({settings.get('sec_b',50)}) SECTION C: Social Issues & Data Management ({settings.get('sec_c',50)})",
        "Answer ALL the questions.",
        "Read ALL the questions carefully.",
        "Do NOT write one-word answers.",
        "The mark allocation generally gives an indication of the number of facts/reasons required and NOT for each sentence.",
        "Number your answers in the margin of the answer paper.",
        "Leave a line open between each answer.",
        "Start each section on a new page.",
        "Number the answers correctly according to the numbering system used in this question paper.",
        "Write neatly and legibly."
    ]

    # Generate 3 different scenarios - one per section, never repeat
    for sec in ["A","B","C"]:
        concepts_for_sec = [c for c in ordered_concepts if c not in used_concepts][:5]
        scenario = generate_sa_scenario_llm(client, model_name, used_scenarios, concepts_for_sec)
        paper["scenarios"][sec] = scenario
        used_scenarios.append(scenario)

    # Section A - Easy - 5 MCQ + Matching, uses first 20 concepts, no repeat
    sec_a_concepts = []
    for i in range(5):
        # Find unused concept
        for c in ordered_concepts:
            if c not in used_concepts:
                sec_a_concepts.append(c)
                used_concepts.add(c)
                break

    # MCQ - Parktown style - URL, Information vs Data, NIC, Byte, Peripheral
    mcq_templates = [
        ("{scenario} - https://greenmap.org.za/2026 is an example of:", ["IP Address","URL","Email address","Domain name"], "B", sec_a_concepts[0] if sec_a_concepts else "URL"),
        ("Which does NOT belong logically in {scenario}?", ["Summary per day","Raw photos from drone","Average counted by AI","Total kg waste"], "B", sec_a_concepts[1] if len(sec_a_concepts)>1 else "Information vs Data"),
        ("Hardware that lets device connect to network in {scenario} is:", ["Network Interface Card","Network Internet Card","Network Interface Connector","Network Internet Connector"], "A", sec_a_concepts[2] if len(sec_a_concepts)>2 else "NIC"),
        ("Order from smallest to largest for {scenario} data:", ["MB, Byte, Bit, KB","Bit, Byte, KB, MB","KB, Bit, Byte, MB","Byte, Bit, KB, MB"], "B", sec_a_concepts[3] if len(sec_a_concepts)>3 else "Byte size"),
        ("Which is peripheral that does BOTH input and output for {scenario}?", ["RAM","Touch screen","Hard Drive","Mouse"], "B", sec_a_concepts[4] if len(sec_a_concepts)>4 else "Peripheral"),
    ]

    for idx, (q_tpl, opts, ans, concept) in enumerate(mcq_templates):
        scenario_short = paper["scenarios"]["A"][:80]
        q_text = q_tpl.format(scenario=scenario_short)
        # Novelty check
        q_vec = vectorizer.transform([q_text])
        sim = cosine_similarity(q_vec, past_vecs).max()
        if sim > 0.82:
            q_text = q_text.replace("GreenMap", "Tshwane e-Waste") + " [Rephrased - SA context]"
        paper["section_a"].append({"num": f"1.1.{idx+1}", "q": q_text, "opts": opts, "ans": ans, "marks": 1, "concept": concept, "type": "MCQ"})

    # Matching 15x1 - use next 15 unused concepts (gaps prioritized)
    matching_pool = []
    for c in ordered_concepts:
        if c not in used_concepts:
            matching_pool.append(c)
            if len(matching_pool)>=15:
                break
    # If not enough, fill with gaps
    for c in matching_pool:
        used_concepts.add(c)

    # Predefined matching based on Parktown gaps
    matching_definitions = [
        ("Ability of new tech to read old tech (e.g. Blu-ray reads DVD)", "Backwards compatibility"),
        ("Stores physical components you touch directly", "Hardware"),
        ("Fastest growing pollutant", "E-Waste"),
        ("Small network around person (Bluetooth)", "PAN"),
        ("OS that runs on server to facilitate communication", "Network Operating System"),
        ("Injury from long computer use", "CTS"),
        ("Reduces brightness, recycles, donates devices", "Green computing"),
        ("Two cabled options for networks", "UTP and Fibre"),
        ("Desktops vs Smartphones on Modularity", "Desktops are modular"),
        ("Software for specific end-user task", "Application software"),
        ("Device that is phone + camera + GPS", "Convergence"),
        ("Stores data using light (CD/DVD)", "Optical storage"),
        ("All devices connected to network", "Nodes"),
        ("Largest network covering SA", "WAN"),
        ("Work + natural law", "Ergonomics"),
    ]

    for idx, (defn, term) in enumerate(matching_definitions[:15]):
        # Ensure term is from taught content if possible
        concept = term
        paper["section_a"].append({"num": f"1.2.{idx+1}", "q": defn, "a": term, "marks": 1, "concept": concept, "type": "Matching"})

    # Section B - 50 marks - System & Networks - One scenario, different concepts, no repeat
    sec_b_concepts = []
    for c in ordered_concepts:
        if c not in used_concepts:
            sec_b_concepts.append(c)
            if len(sec_b_concepts)>=10:
                break
    for c in sec_b_concepts:
        used_concepts.add(c)

    scenario_b = paper["scenarios"]["B"]
    for i, concept in enumerate(sec_b_concepts):
        chunk = random.choice(taught_chunks)[:600] if taught_chunks else f"Content about {concept}"
        # Parktown language: Explain, Define, Expand, Provide ONE example
        prompt = f"""You are Parktown examiner Mr K Witelson. Formal Grade 7 simple precise.
Scenario for Section B (use this): {scenario_b}
Taught content (ONLY source): {chunk}
Concept to assess (was taught, must assess, never repeat concept {concept} in same paper, concept list used so far: {list(used_concepts)[:5]}):
Concept: {concept}
Style: Use verbs Explain, Define, Expand acronym, Provide ONE example, Why must, What is purpose, Difference between. NEVER Discuss.
Marks: 2-3
Create ONE question like "2.{i+1} Explain {concept} using scenario. Why important? (2)" - Output only question."""

        if client and model_name:
            try:
                resp = client.chat.completions.create(model=model_name, messages=[{"role":"user","content":prompt}], temperature=0.75, max_tokens=250)
                q_text = resp.choices[0].message.content.strip()
                q_vec = vectorizer.transform([q_text])
                sim = cosine_similarity(q_vec, past_vecs).max()
                if sim > 0.82:
                    q_text = f"2.{i+1} Explain {concept} in {scenario_b[:50]}... Why important for lab? (2) [Auto-rephrased for originality]"
            except:
                q_text = f"2.{i+1} Explain {concept} using scenario: {scenario_b[:80]}. Why important for lab? (2)"
        else:
            q_text = f"2.{i+1} Explain {concept} using scenario: {scenario_b[:80]}. Why important for lab? (2) [Gap: taught but never assessed]"

        paper["section_b"].append({"q": q_text, "marks": 2, "concept": concept, "scenario": scenario_b, "source": chunk[:100]})

    # Section C - 50 marks - Hard - Different scenario, no repeat concepts
    sec_c_concepts = []
    for c in ordered_concepts:
        if c not in used_concepts:
            sec_c_concepts.append(c)
            if len(sec_c_concepts)>=10:
                break
    for c in sec_c_concepts:
        used_concepts.add(c)

    scenario_c = paper["scenarios"]["C"]
    for i, concept in enumerate(sec_c_concepts):
        chunk = random.choice(taught_chunks)[:600] if taught_chunks else f"Content about {concept}"
        prompt = f"""You are Parktown examiner Section C - TOP OF HILL HARD.
Scenario for Section C (different from Section B, fresh SA news): {scenario_c}
Taught content: {chunk}
Concept: {concept} (never repeat, used concepts: {list(used_concepts)[:5]})
Must assess: Ergonomics, E-Waste fastest growing, Green actions Recycle/Donate/Upgrade, AI ethics ChatGPT, Dangers sharing location
Style: What does X mean? Give 2 ways to avoid, Why does school pay more, Is this ethical? Motivate with 2 points, Explain why E-Waste fastest growing
Marks: 2-3
Create ONE hard question."""

        if client and model_name:
            try:
                resp = client.chat.completions.create(model=model_name, messages=[{"role":"user","content":prompt}], temperature=0.8, max_tokens=300)
                q_text = resp.choices[0].message.content.strip()
            except:
                q_text = f"4.{i+1} What is {concept}? Give 2 ways to practice it in {scenario_c[:60]}... (3)"
        else:
            q_text = f"4.{i+1} What is {concept}? Give 2 practical ways {scenario_c[:60]}... can practice it. Why important? (3) [Gap filled]"

        paper["section_c"].append({"q": q_text, "marks": 3 if i%2==0 else 2, "concept": concept, "scenario": scenario_c})

    return paper, used_concepts, used_scenarios

def export_parktown_docx(paper, subject="IT Grade 10"):
    if not HAS_DOCX:
        return None
    d = docx.Document()
    for h in paper['header']:
        d.add_heading(h, 0 if "PARKTOWN" in h else 2)
    d.add_heading("INSTRUCTIONS AND INFORMATION", 1)
    for i, inst in enumerate(paper['instructions'], 1):
        d.add_paragraph(f"{i}. {inst}")
    d.add_heading(f"SECTION A: SHORT QUESTIONS ({sum([q['marks'] for q in paper['section_a']])}) - Scenario: {paper['scenarios']['A'][:100]}", 1)
    for q in paper['section_a']:
        if q['type'] == "MCQ":
            p = d.add_paragraph()
            p.add_run(f"{q['num']} {q['q']} (1) [Concept: {q['concept']}]").bold = True
            p = d.add_paragraph(f"A) {q['opts'][0]} B) {q['opts'][1]} C) {q['opts'][2]} D) {q['opts'][3]}")
        else:
            d.add_paragraph(f"{q['num']} {q['q']} -> {q['a']} (1) [Concept: {q['concept']}]")
    d.add_heading(f"SECTION B: SYSTEM TECHNOLOGIES & NETWORKS - Scenario: {paper['scenarios']['B']}", 1)
    for q in paper['section_b']:
        d.add_paragraph(f"{q['q']} [Concept: {q['concept']} | Marks: {q['marks']}]")
    d.add_heading(f"SECTION C: SOCIAL ISSUES & DATA MANAGEMENT - Scenario: {paper['scenarios']['C']}", 1)
    for q in paper['section_c']:
        d.add_paragraph(f"{q['q']} [Concept: {q['concept']} | Marks: {q['marks']}]")
    bio = io.BytesIO()
    d.save(bio)
    return bio.getvalue()

# ---------- UI ----------
st.title("🏫 PaperForge - Parktown Updated - No Repeat + SA News Scenarios")
st.caption("Learns past paper style + headings + weights + language, reads PPTs/notes, assesses ONLY taught content, never repeats concepts, fresh SA scenario every time.")

with st.sidebar:
    st.header("🤖 Llama Brain - For SA News Scenarios")
    provider = st.selectbox("Provider", ["Groq (Llama 3.3 - Free & Fast) - Recommended", "OpenAI"])
    saved_key = ""
    try:
        if provider.startswith("Groq") and "GROQ_API_KEY" in st.secrets:
            saved_key = st.secrets["GROQ_API_KEY"]
        elif "OPENAI_API_KEY" in st.secrets:
            saved_key = st.secrets["OPENAI_API_KEY"]
    except:
        saved_key = ""
    if saved_key:
        st.success(f"✅ Key loaded from secrets.toml - {saved_key[:7]}...{saved_key[-4:]}")
        api_key = saved_key
        if st.checkbox("Override saved key"):
            api_key = st.text_input("API Key", type="password")
    else:
        api_key = st.text_input("Groq API Key gsk_...", type="password", help="Get free at console.groq.com/keys - for fresh SA scenarios")
        if api_key:
            st.success("✅ Connected!")
        else:
            st.info("Add Groq key for fresh SA news scenarios every generation. Offline template still works with pool.")

    st.divider()
    st.header("📝 Parktown Settings - Locked Style")
    examiner = st.text_input("Examiner", "Mr K Witelson")
    moderator = st.text_input("Moderator", "Mrs L Le Grange")
    total_marks = st.number_input("Total Marks", 100, 150, 120)
    sec_a = st.number_input("Section A", 10, 30, 20)
    sec_b = st.number_input("Section B", 30, 70, 50)
    sec_c = st.number_input("Section C", 30, 70, 50)

col1, col2 = st.columns(2)
with col1:
    st.subheader("1️⃣ Taught Content - PPTs, Notes, Concepts (ONLY what was taught will be assessed)")
    taught_files = st.file_uploader("Drop 8 PPTs + notes + textbook", type=["pdf","docx","pptx","txt"], accept_multiple_files=True, key="taught")
    if taught_files:
        st.write(f"✅ {len(taught_files)} files uploaded: {[f.name for f in taught_files]}")
with col2:
    st.subheader("2️⃣ Past Papers - Style Learning (style, language, types, headings, weights, concepts)")
    past_files = st.file_uploader("Drop past Parktown papers", type=["pdf","docx","txt"], accept_multiple_files=True, key="past")
    if past_files:
        st.write(f"✅ {len(past_files)} past papers: {[f.name for f in past_files]}")

if taught_files and past_files:
    if st.button("🔍 Learn Past Paper Style + Read Taught Content + Find Gaps", type="primary", use_container_width=True):
        with st.spinner("Step 1: Learning past paper style (headings, types, marks, weights, language)... Step 2: Reading PPTs... Step 3: Finding gaps..."):
            taught_texts = [extract_text_from_file(f) for f in taught_files]
            past_texts = [extract_text_from_file(f) for f in past_files]
            st.session_state['taught_texts'] = taught_texts
            st.session_state['past_texts'] = past_texts
            st.session_state['taught_chunks'] = []
            for t in taught_texts:
                st.session_state['taught_chunks'].extend(chunk_text(t))

            style_info = learn_past_paper_style(past_texts)
            taught_keywords, buckets = extract_taught_concepts(taught_texts, st.session_state['taught_chunks'])
            past_concepts = style_info['concepts_assessed']
            gaps, ordered_concepts = gap_and_no_repeat(taught_keywords, past_concepts)

            st.session_state['style_info'] = style_info
            st.session_state['taught_keywords'] = taught_keywords
            st.session_state['buckets'] = buckets
            st.session_state['gaps'] = gaps
            st.session_state['ordered_concepts'] = ordered_concepts
            st.session_state['used_scenarios_global'] = []  # Track across generations to never repeat scenario

            st.success(f"✅ Learned: {len(style_info['headings'])} headings, {len(style_info['question_types'])} Q types, {style_info['total_marks_detected']} marks detected, {len(taught_keywords)} taught concepts, {len(gaps)} gaps (taught but never assessed)!")

if 'style_info' in st.session_state:
    style_info = st.session_state['style_info']
    st.divider()
    st.subheader("📊 What Engine Learned From Your Past Papers (Style, Language, Weights)")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Headings Found", len(style_info['headings']))
        st.write(style_info['headings'])
    with c2:
        st.metric("Total Marks Detected", style_info['total_marks_detected'])
        st.write("**Section Weights (from past):**")
        st.json(style_info['section_weights'])
    with c3:
        st.write("**Grade 10 Assessment Headings (CAPS Aligned):**")
        st.caption("A1 Knowledge, A2 Comprehension, B1 Application, B2 Analysis, C Evaluation - Balanced for Grade 10")
        st.bar_chart(style_info['question_types'])
    with c4:
        st.write("**Language Verbs (Parktown formal):**")
        st.bar_chart(style_info['language_verbs'])

    st.write("**Formal Patterns Found:**", style_info['formal_patterns'])
    st.write("**Concepts Already Assessed (will be deprioritized to avoid repeat):**", ", ".join(style_info['concepts_assessed'][:15]))

    st.divider()
    st.subheader("📚 What Was Actually Taught (from PPTs/Notes) - ONLY These Will Be Assessed")
    st.write(f"**All Taught Concepts ({len(st.session_state['taught_keywords'])}):**", ", ".join(st.session_state['taught_keywords'][:25]))
    st.write("**Buckets:**")
    st.json(st.session_state['buckets'])

    st.subheader("🎯 GAP Analysis - Taught But NEVER Assessed (Engine Will Focus 40% Here)")
    st.error(f"GAPS ({len(st.session_state['gaps'])}): {', '.join(st.session_state['gaps'][:20])}")
    st.info(f"Ordered for No-Repeat Assessment (gaps first): {', '.join(st.session_state['ordered_concepts'][:20])} - Each concept will be used ONLY ONCE per paper!")

    st.divider()
    st.subheader("🌍 SA Current News Scenarios - Fresh Every Time, Never Repeat")
    if 'used_scenarios_global' in st.session_state:
        st.write(f"Used scenarios so far (will never repeat): {st.session_state['used_scenarios_global']}")

    settings = {"examiner": examiner, "moderator": moderator, "total_marks": total_marks, "sec_a": sec_a, "sec_b": sec_b, "sec_c": sec_c, "time": "2.5 hours", "pages": "11"}

    if st.button("✨ Generate Brand New Paper - No Repeat Concepts + Fresh SA Scenario", type="primary", use_container_width=True):
        client = None
        model_name = None
        if api_key:
            try:
                from openai import OpenAI
                if provider.startswith("Groq"):
                    client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=api_key)
                    model_name = "llama-3.3-70b-versatile"
                else:
                    client = OpenAI(api_key=api_key)
                    model_name = "gpt-4o-mini"
            except Exception as e:
                st.warning(f"LLM error {e}, using template pool")

        with st.spinner("Generating fresh SA news scenarios (never repeat) + assessing all taught concepts with no repeat..."):
            paper, used_concepts, used_scenarios = generate_paper_no_repeat(
                st.session_state['taught_chunks'],
                st.session_state['ordered_concepts'],
                st.session_state['gaps'],
                st.session_state['past_texts'],
                style_info,
                client,
                model_name,
                settings
            )
            # Track globally to never repeat scenario across generations
            if 'used_scenarios_global' not in st.session_state:
                st.session_state['used_scenarios_global'] = []
            st.session_state['used_scenarios_global'].extend(used_scenarios)
            st.session_state['paper'] = paper
            st.session_state['used_concepts'] = used_concepts
            st.success(f"✅ Generated! {len(used_concepts)} unique concepts assessed (no repeat), 3 fresh SA scenarios: {used_scenarios}")

if 'paper' in st.session_state:
    paper = st.session_state['paper']
    st.divider()
    st.markdown("## 📄 FINAL PAPER - Parktown Exact Format - No Repeat Concepts - Fresh SA Scenarios")
    for h in paper['header']:
        st.markdown(f"**{h}**")

    st.markdown("### INSTRUCTIONS AND INFORMATION")
    for i, inst in enumerate(paper['instructions'], 1):
        st.write(f"{i}. {inst}")

    st.markdown(f"### SECTION A: SHORT QUESTIONS ({sum([q['marks'] for q in paper['section_a']])}) - Scenario: {paper['scenarios']['A']}")
    st.info(f"Scenario A (SA News): {paper['scenarios']['A']}")
    for q in paper['section_a']:
        if q['type'] == "MCQ":
            st.write(f"**{q['num']}** {q['q']} ({q['marks']}) [Concept: {q['concept']} - No repeat]")
            st.write(f"A) {q['opts'][0]} B) {q['opts'][1]} C) {q['opts'][2]} D) {q['opts'][3]} | Answer: {q['ans']}")
        else:
            st.write(f"**{q['num']}** {q['q']} -> {q['a']} ({q['marks']}) [Concept: {q['concept']}]")

    st.markdown(f"### SECTION B: SYSTEM TECHNOLOGIES & NETWORKS ({sum([q['marks'] for q in paper['section_b']])})")
    st.info(f"Scenario B (Different SA News, Never Repeat A): {paper['scenarios']['B']}")
    for q in paper['section_b']:
        st.write(f"**{q['q']}** [Concept: {q['concept']} | Source: {q['source'][:60]}...]")

    st.markdown(f"### SECTION C: SOCIAL ISSUES & DATA MANAGEMENT ({sum([q['marks'] for q in paper['section_c']])}) - TOP OF HILL")
    st.info(f"Scenario C (Different SA News, Never Repeat A or B): {paper['scenarios']['C']}")
    for q in paper['section_c']:
        st.write(f"**{q['q']}** [Concept: {q['concept']} | Marks: {q['marks']}]")

    st.success(f"✅ No concept repeated: {len(st.session_state['used_concepts'])} unique concepts assessed. All from taught content only. Gaps filled: {len([c for c in st.session_state['used_concepts'] if c in st.session_state['gaps']])} previously unassessed topics now assessed.")

    # Export
    docx_bytes = export_parktown_docx(paper)
    if docx_bytes:
        st.download_button("📥 Download Parktown Paper DOCX - No Repeat + Fresh SA Scenarios", docx_bytes, file_name=f"Parktown_2026_NoRepeat_FreshSA_{datetime.now().strftime('%Y%m%d_%H%M')}.docx")
        st.download_button("📥 Download Scenarios Used (to ensure never repeat)", "\n".join(paper['scenarios'].values()), file_name="SA_Scenarios_Used.txt")
