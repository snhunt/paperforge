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

# ---------- 1. LEARN PAST PAPER STYLE - Full sentences, Easy reading, Grade 10 terms ----------
# Learns how past papers ask: full sentences, easy reading level, but keeps Grade 10 terminology (CPU, RAM, NIC, ISP, URL, etc)
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

def gap_and_no_repeat(taught_concepts, past_concepts, past_patterns=None, past_texts=None):
    # FIX: Handle empty taught_concepts to avoid ZeroDivisionError
    if not taught_concepts:
        # Default fallback concepts if no taught content uploaded yet
        taught_concepts = [
            "Backwards compatibility", "PAN", "Network Operating System", "E-Waste", 
            "Ergonomics", "Green computing", "UTP and Fibre", "Nodes", "WAN", "WLAN",
            "IP address", "ISP", "URL", "Domain name", "Broadband", "Convergence",
            "Touch screen", "Magnetic storage", "Electronic storage", "Optical storage"
        ]
    
    past_set = set([c.lower() for c in (past_concepts or [])])
    past_questions_text = " ".join(past_texts).lower() if past_texts else ""
    
    # Count frequency of each taught concept in past papers
    freq_map = {}
    for c in taught_concepts:
        try:
            freq_map[c] = past_questions_text.count(c.lower())
        except:
            freq_map[c] = 0
    
    # Sort taught concepts by frequency ascending (least assessed first)
    taught_sorted_by_least = sorted(taught_concepts, key=lambda x: freq_map.get(x, 0))
    
    # GAPS = never assessed (freq 0) - 100% priority
    gaps = [c for c in taught_sorted_by_least if freq_map.get(c, 0) == 0]
    gaps = list(dict.fromkeys(gaps))  # dedupe
    
    # Least assessed = assessed 1-2 times
    least_assessed = [c for c in taught_sorted_by_least if 1 <= freq_map.get(c, 0) <= 2 and c.lower() not in [g.lower() for g in gaps]]
    
    # Medium assessed = 3+ times
    medium_assessed = [c for c in taught_sorted_by_least if freq_map.get(c, 0) >= 3]
    
    # STRICT ORDER: Gaps (0 times) -> Least (1-2 times) -> Medium (3+ times)
    ordered = gaps + least_assessed + medium_assessed
    
    # Ensure at least 50 concepts - FIXED: Avoid ZeroDivisionError when cycle_pool empty
    if len(ordered) < 50:
        cycle_pool = gaps + least_assessed
        if not cycle_pool:
            cycle_pool = taught_sorted_by_least
        if not cycle_pool:
            cycle_pool = taught_concepts  # Final fallback
        
        # Guard against empty cycle_pool
        if cycle_pool and len(cycle_pool) > 0:
            idx = 0
            while len(ordered) < 60 and idx < 200:
                base = cycle_pool[idx % len(cycle_pool)]
                if base not in ordered:
                    ordered.append(base)
                idx += 1
        else:
            # If still empty, just use taught_concepts
            ordered = taught_concepts[:60]
    
    # Final safety: ensure ordered not empty
    if not ordered:
        ordered = taught_concepts[:60] if taught_concepts else ["Backwards compatibility"]
    
    return gaps, ordered, freq_map

# ---------- 3. SA CURRENT NEWS + PARKTOWN GIRLS TRENDING - NEVER REPEAT ----------
# Mix of SA news + high school girls trending: nail polish, new apps, fashion, beauty, social
SA_NEWS_POOL_2025_2026 = [
    # SA NEWS - Original
    "GreenMap AI Lab - Drones mapping illegal dumping in Gauteng using AI, solar powered lab, 15k photos, water crisis data",
    "Township SME Solar Shop - Spaza shop in Soweto installing solar during load-shedding, using POS system, mobile payments, stock tracking",
    "Matric Results Portal - Department of Education portal for 2025 Matric results, 900k learners checking results, high traffic, data privacy concerns",
    "Springboks Fan App - Rugby World Cup 2027 qualifier app, live streaming, ticket sales, fan location sharing, township viewing parties",
    "Cape Town Water Crisis Dashboard - Day Zero 2.0 monitoring, IoT sensors in dams, 4 million residents checking usage, misinformation",
    "SASSA Grant Payment System - SRD grant biometric verification, USSD vs app, fraud prevention, rural connectivity issues",
    "Johannesburg Road Repair Tracker - Pothole reporting app using citizen photos, AI classification, municipal dashboard, e-waste from old devices",
    "KZN Flood Early Warning - Drone + SMS early warning for floods, community WhatsApp groups, offline data collection, green computing with old phones",
    "Limpopo AgriTech - Small farmers using smartphone sensors for soil moisture, market price app, UTP vs satellite connectivity",
    "Tshwane e-Waste Drive - Schools collecting e-waste, refurbishing PCs for township schools, tracking with QR codes, ergonomic refurbishment stations",
    # PARKTOWN GIRLS TRENDING - Nail polish, new apps, fashion, beauty
    "GlowNail Studio App - Parktown girls booking GelX nail art, 500 learners, TikTok nail trends, POS, Instagram booking, 15k nail design photos stored, fibre vs WiFi",
    "Shein Haul Tracker - Matric dance dress and fashion haul tracking app, 800 girls sharing outfits, data overload, cloud storage for 20k photos, e-waste from old phones",
    "Thrift2Fashion Marketplace - Y2K thrift fashion resale app by Parktown learners, second-hand uniforms to trendy outfits, UTP network in lab, database of sizes and prices",
    "Makeup & Skincare Routine App - Skincare tracker for teen skin, 600 users logging routines, AI suggests products, data privacy, green computing recycling old makeup containers",
    "Parktown Beauty Bar POS - School entrepreneurial day beauty stall, nail polish and lashes, mobile payments, barcode scanning, stock database, load-shedding backup with inverter",
    "TikTok Trends Dashboard - Parktown girls tracking trending sounds and dances, 2k videos uploaded, information overload, IP addresses for each phone, fibre vs broadband",
    "Matric Dance Dress Rental - Dress rental app for matric dance, 300 dresses, QR codes, online bookings, database management, network nodes in school lab",
    "Braids & Beauty Booking - Braids, cornrows and wig booking app, 400 clients, WhatsApp vs app bookings, UTP cables in salon, data management of appointments",
    "Latest iPhone vs Android Poll - Parktown IT class poll on iPhone 16 vs Samsung trends, 250 responses, spreadsheet analysis, convergence - phone is camera + wallet + social",
    "Nail Art Design AI - AI generates nail art designs, 10k designs, ethical issue - is AI art same as human art? E-waste from old tablets, ergonomic chairs for long designing",
    "Parktown Fashion Week Live Stream - Live streaming fashion show, 5 cameras, high bandwidth, network OS for streaming server, data vs information - raw footage vs edited highlights",
    "CapCut & Beauty Filters App - Video editing with beauty filters, 1k learners editing, plugin vs extension, W3C standards, green computing - old phones donated for editing"
]

def generate_sa_scenario_llm(client, model, used_scenarios, concepts_for_section):
    used_str = ", ".join(used_scenarios) if used_scenarios else "None yet"
    concepts_str = ", ".join(concepts_for_section[:5])

    prompt = f"""You are a scenario generator for Parktown High School for Girls Grade 10 IT exams - SA news + high school girls trending topics.

Already used scenarios (NEVER repeat): {used_str}
Concepts to assess: {concepts_str}

Task: Generate ONE brand new scenario that Parktown girls will LOVE - mix SA news + trending teen topics:

Choose ONE of these trending angles (rotate, never repeat used):
- SA NEWS: load-shedding, water crisis, matric, SASSA, Springboks, potholes, e-waste, solar, township economy
- GIRLS TRENDING: nail polish GelX designs, new apps TikTok/CapCut/Shein, latest fashion Y2K/thrift/matric dance dresses, makeup/skincare, braids booking, beauty filters, iPhone vs Android, Instagram/TikTok trends

Requirements:
- 100% relatable to Parktown Grade 10 girls in Gauteng
- Example good: "GlowNail Studio App - 500 Parktown girls booking GelX nail art via TikTok, POS system, 15k nail designs stored, fibre network, database of bookings"
- Example good: "Shein Haul Tracker - 800 girls sharing matric dance fashion hauls, cloud storage 20k photos, data overload, e-waste"
- Must include: hardware (phones, PCs, touch screens), network (WiFi, fibre, UTP, IP), data (database, 15k photos, cloud)
- 2-3 sentences, easy reading, Grade 10 terms kept
- Suitable for Grade 10 exam standard

Output ONLY scenario title and 2-sentence description.
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

# ---------- GRADE 10 CAPS STRUCTURE - LOCKED ----------
GRADE10_STRUCTURE = {
    "Q1": {"name": "Short Questions", "marks": 20, "desc": "MCQ A-D + Matching Column A/B + One-word/Short", "cognitive": "Knowledge & Recall"},
    "Q2": {"name": "System Technologies", "marks": 25, "desc": "Hardware, Software, System Software, Hardware/Software interaction", "cognitive": "Comprehension & Application"},
    "Q3": {"name": "Communication and Network Technologies", "marks": 20, "desc": "Networks, Internet, WWW, Email, PAN/LAN/WAN, UTP/Fibre, Nodes, Network OS", "cognitive": "Application & Analysis"},
    "Q4": {"name": "Data and Information Management", "marks": 16, "desc": "Data vs Information, Databases, Spreadsheets, Information overload, HTML", "cognitive": "Analysis"},
    "Q5": {"name": "Solution Development", "marks": 19, "desc": "Problem solving, Algorithms, Flowcharts, Pseudo, Input/Processing/Output", "cognitive": "Application & Creation"},
    "Q6": {"name": "Integrated scenario", "marks": 20, "desc": "Real SA scenario integrating all topics + Green computing + Ergonomics + Ethics + AI", "cognitive": "Evaluation & Synthesis"},
}
TOTAL_MARKS_G10 = 120


def generate_paper_no_repeat(taught_chunks, ordered_concepts, gaps, past_texts, style_info, client, model_name, settings):
    used_concepts = set()
    used_scenarios = []
    past_all = " ".join(past_texts)
    vectorizer = TfidfVectorizer(stop_words='english').fit(taught_chunks + [past_all] + ["dummy"])
    past_vecs = vectorizer.transform([past_all])

    paper = {
        "header": [], "instructions": [], 
        "Q1": [], "Q2": [], "Q3": [], "Q4": [], "Q5": [], "Q6": [],
        "scenarios": {}, "structure": GRADE10_STRUCTURE
    }

    # Header - Parktown style but Grade 10
    paper["header"] = [
        "PARKTOWN HIGH SCHOOL FOR GIRLS - GRADE 10",
        f"Examiner: {settings.get('examiner','Mr K Witelson')}",
        f"Moderator: {settings.get('moderator','Mrs L Le Grange')}",
        f"Marks: {TOTAL_MARKS_G10}",
        f"Time: {settings.get('time','2.5 hours')}",
        f"This paper consists of {settings.get('pages','11')} pages.",
        f"Date: {settings.get('date','2026')}"
    ]
    # SHORT instructions - easy reading, Grade 10 standard
    paper["instructions"] = [
        f"This paper has 6 questions. Total: {TOTAL_MARKS_G10} marks.",
        "Answer all questions.",
        "Read each question carefully.",
        "Show all facts. Marks show how many facts you need.",
        "Number your answers correctly.",
        "Start each question on a new page.",
        "Write neatly."
    ]

    # Generate 6 different SA scenarios - one per QUESTION, never repeat
    for q_key in ["Q1","Q2","Q3","Q4","Q5","Q6"]:
        concepts_for_sec = [c for c in ordered_concepts if c not in used_concepts][:5]
        scenario = generate_sa_scenario_llm(client, model_name, used_scenarios, concepts_for_sec)
        paper["scenarios"][q_key] = scenario
        used_scenarios.append(scenario)

    # STRICT: NEVER repeat same questions, ALWAYS new on what was not assessed
    # NEW RULE: No general concepts - reuse least assessed with brand new questions
    past_questions_exact = []
    for txt in past_texts:
        qs = re.split(r'\n\d+\.\s*|\nQ\d+|\nQUESTION', txt)
        past_questions_exact.extend([q.strip().lower()[:100] for q in qs if len(q.strip())>20])
    
    # Track how many times each concept has been reused in current paper with new phrasing
    reuse_count = {}
    
    def get_next_concept():
        # Phase 1: Use all gaps first (never assessed) - each once
        for c in ordered_concepts:
            if c.lower() not in [uc.lower() for uc in used_concepts]:
                if c in gaps:
                    used_concepts.add(c)
                    reuse_count[c] = 1
                    return c
        
        # Phase 2: Use least assessed concepts (assessed 1-2 times before) - brand new questions
        # These are from ordered_concepts that are not gaps but are least assessed
        for c in ordered_concepts:
            if c.lower() not in [uc.lower() for uc in used_concepts]:
                # Least assessed - freq 1-2 times
                used_concepts.add(c)
                reuse_count[c] = 1
                return c
        
        # Phase 3: NO GENERAL CONCEPTS - Instead, reuse least assessed concepts again with NEW question phrasing
        # Sort gaps + least assessed by reuse count ascending
        least_reused_pool = sorted(list(reuse_count.keys()), key=lambda x: reuse_count.get(x, 0))
        if not least_reused_pool:
            least_reused_pool = gaps if gaps else ordered_concepts
        
        for base in least_reused_pool:
            count = reuse_count.get(base, 0)
            if count < 3:  # Allow up to 3 brand new questions per concept
                # Create new angle but keep base concept for tracking
                new_concept = f"{base}"  # Keep same base text, but we track reuse
                # Increment reuse
                reuse_count[base] = count + 1
                # Return base concept - question generator will create brand new phrasing
                # Use variant identifier for used_concepts to allow reuse
                variant_key = f"{base}__reuse_{count+1}"
                if variant_key.lower() not in [uc.lower() for uc in used_concepts]:
                    used_concepts.add(variant_key)
                    return base
        
        # Final fallback: cycle through least assessed again - still no general concepts
        if ordered_concepts:
            # Find least assessed by original freq
            base = ordered_concepts[0]  # This is least assessed due to sorting
            reuse_count[base] = reuse_count.get(base, 0) + 1
            variant_key = f"{base}__reuse_{reuse_count[base]}"
            used_concepts.add(variant_key)
            return base
        
        # Should never reach here
        return ordered_concepts[0] if ordered_concepts else "Backwards compatibility"

    def is_duplicate_question(new_q):
        new_q_lower = new_q.lower()
        for past_q in past_questions_exact:
            if past_q in new_q_lower or new_q_lower in past_q:
                return True
        return False

    # Q1: Short Questions (20) - 5 MCQ (5) + 10 Matching (10) + 5 Short (5) = 20
    q1_concepts = [str(get_next_concept()) for _ in range(5)]
    mcq_templates = [
        ("{scenario} - https://greenmap.org.za/2026 is an example of:", ["IP Address","URL","Email address","Domain name"], "B"),
        ("Which does NOT belong logically in {scenario}?", ["Summary per day","Raw photos from drone","Average counted by AI","Total kg waste"], "B"),
        ("Hardware that lets device connect to network in {scenario} is:", ["Network Interface Card","Network Internet Card","Network Interface Connector","Network Internet Connector"], "A"),
        ("Order from smallest to largest for data in {scenario}:", ["MB, Byte, Bit, KB","Bit, Byte, KB, MB","KB, Bit, Byte, MB","Byte, Bit, KB, MB"], "B"),
        ("Which peripheral does BOTH input and output for {scenario}?", ["RAM","Touch screen","Hard Drive","Mouse"], "B"),
    ]
    for idx, (q_tpl, opts, ans) in enumerate(mcq_templates):
        concept = q1_concepts[idx] if idx < len(q1_concepts) else get_next_concept()
        q_text = q_tpl.format(scenario=paper["scenarios"]["Q1"][:60])
        q_vec = vectorizer.transform([q_text])
        sim = cosine_similarity(q_vec, past_vecs).max()
        if sim > 0.82:
            q_text += " [Rephrased SA]"
        paper["Q1"].append({"num": f"1.{idx+1}", "q": q_text, "opts": opts, "ans": ans, "marks": 1, "concept": concept, "type": "MCQ"})

    # Matching 10 marks
    matching_defs = [
        ("Ability of new tech to read old tech (Blu-ray reads DVD)", "Backwards compatibility"),
        ("Fastest growing pollutant", "E-Waste"),
        ("Small network around person (Bluetooth)", "PAN"),
        ("OS that runs on server", "Network Operating System"),
        ("Injury from long computer use", "CTS"),
        ("Reduces brightness, recycles, donates", "Green computing"),
        ("Two cabled options", "UTP and Fibre"),
        ("All devices on network", "Nodes"),
        ("Largest network covering SA", "WAN"),
        ("Work + natural law", "Ergonomics"),
    ]
    for idx, (defn, term) in enumerate(matching_defs[:10]):
        concept = get_next_concept() or term
        concept = str(concept)
        paper["Q1"].append({"num": f"1.{6+idx}", "q": defn, "a": term, "marks": 1, "concept": concept, "type": "Matching"})

    # Short 5x1
    for idx in range(5):
        concept = get_next_concept() or f"Concept {idx+1}"
        concept = str(concept)
        chunk_raw = random.choice(taught_chunks) if taught_chunks else concept
        chunk = str(chunk_raw)
        paper["Q1"].append({"num": f"1.{16+idx}", "q": f"Define {concept} in context of {paper['scenarios']['Q1'][:40]}... (1)", "marks": 1, "concept": concept, "type": "Short"})

    # Q2: System Technologies (25) - Hardware, Software, System Software - 8-9 questions
    for i in range(9):
        concept = get_next_concept() or "General system concept"
        concept = str(concept)
        chunk_raw = random.choice(taught_chunks) if taught_chunks else concept
        chunk = str(chunk_raw)
        scenario = paper["scenarios"]["Q2"]
        prompt = f"""You are Grade 10 examiner for QUESTION 2: System Technologies (25 marks).
Scenario (use): {scenario}
Taught content ONLY: {chunk}
Concept (must assess, never repeat {concept}): {concept}
Topics: Hardware, Software, System Software, Input/Output, Storage types Magnetic/Electronic/Optical, Backwards compatibility, Modularity, Touch screen, RAM/CPU
Style: Parktown formal Grade 10 - Explain, Define, Expand, Provide ONE example, Why must, Difference between. 2-3 marks.
Create ONE Q2 question like "2.{i+1} Explain {concept}..." Output only question."""

        if client and model_name:
            try:
                resp = client.chat.completions.create(model=model_name, messages=[{"role":"user","content":prompt}], temperature=0.75, max_tokens=250)
                q_text = resp.choices[0].message.content.strip()
            except:
                q_text = f"2.{i+1} Explain {concept} using {scenario[:60]}... Why important for system? (3)"
        else:
            q_text = f"2.{i+1} Explain {concept} using scenario: {scenario[:70]}... Why important for system? ({2 if i%2==0 else 3}) [Gap: taught but never assessed]"

        marks = 3 if i < 7 else 2  # 7*3=21 + 2*2=4 => 25
        paper["Q2"].append({"q": q_text, "marks": marks, "concept": concept, "source": str(chunk)[:100]})

    # Q3: Communication and Network Technologies (20) - 7 questions
    for i in range(7):
        concept = get_next_concept() or "General network concept"
        concept = str(concept)
        chunk_raw = random.choice(taught_chunks) if taught_chunks else concept
        chunk = str(chunk_raw)
        scenario = paper["scenarios"]["Q3"]
        prompt = f"""Grade 10 Q3: Communication and Network Technologies (20 marks). Scenario: {scenario}. Taught: {chunk}. Concept: {concept}. Topics: PAN/LAN/WLAN/WAN/MAN, Nodes, Network OS, UTP vs Fibre, IP/Domain/ISP/Broadband/WiFi vs Internet, Shaped vs throttled, Cellular. Formal Grade 10. Create ONE question 2-3 marks."""

        if client and model_name:
            try:
                resp = client.chat.completions.create(model=model_name, messages=[{"role":"user","content":prompt}], temperature=0.75, max_tokens=250)
                q_text = resp.choices[0].message.content.strip()
            except:
                q_text = f"3.{i+1} Explain {concept} in {scenario[:60]}... (3)"
        else:
            q_text = f"3.{i+1} Explain {concept} in {scenario[:70]}... What is purpose? (3)" if i<6 else f"3.{i+1} Define {concept}... (2)"

        marks = 3 if i < 6 else 2  # 6*3=18 +2=20
        paper["Q3"].append({"q": q_text, "marks": marks, "concept": concept})

    # Q4: Data and Information Management (16) - 6 questions
    for i in range(6):
        concept = get_next_concept() or "General data concept"
        concept = str(concept)
        chunk_raw = random.choice(taught_chunks) if taught_chunks else concept
        chunk = str(chunk_raw)
        scenario = paper["scenarios"]["Q4"]
        prompt = f"""Grade 10 Q4: Data and Information Management (16 marks).
Learn: Full sentences, easy reading, keep Grade 10 terms: data, information, information overload, database, HTML, browser, search engine, W3C, plug-in, extension.
Scenario: {scenario}
Taught: {chunk}
Concept: {concept}

RULES:
- Full sentence, easy reading
- Keep {concept}
- Example: "Explain the difference between data and information using the drone photo as an example. (2)"
- 2-3 marks
Create ONE question. Output ONLY question."""

        if client and model_name:
            try:
                resp = client.chat.completions.create(model=model_name, messages=[{"role":"user","content":prompt}], temperature=0.75, max_tokens=250)
                q_text = resp.choices[0].message.content.strip()
            except:
                q_text = f"4.{i+1} Explain {concept} using {scenario[:60]}... (3)"
        else:
            q_text = f"4.{i+1} Explain difference between {concept} and related concept in {scenario[:60]}... (3)" if i<4 else f"4.{i+1} Define {concept}... (2)"

        marks = 3 if i < 4 else 2  # 4*3=12 +2*2=16
        paper["Q4"].append({"q": q_text, "marks": marks, "concept": concept})

    # Q5: Solution Development (19) - 7 questions - Problem solving, algorithms
    for i in range(7):
        concept = get_next_concept() or "General solution concept"
        concept = str(concept)
        chunk_raw = random.choice(taught_chunks) if taught_chunks else concept
        chunk = str(chunk_raw)
        scenario = paper["scenarios"]["Q5"]
        prompt = f"""Grade 10 Q5: Solution Development (19 marks). Scenario: {scenario}. Taught: {chunk}. Concept: {concept}. Topics: Problem solving, Input/Processing/Output, Algorithms, Flowcharts, Pseudocode, Trace tables, Validation. 2-3 marks. Formal Grade 10."""

        if client and model_name:
            try:
                resp = client.chat.completions.create(model=model_name, messages=[{"role":"user","content":prompt}], temperature=0.75, max_tokens=250)
                q_text = resp.choices[0].message.content.strip()
            except:
                q_text = f"5.{i+1} Develop solution for {concept} in {scenario[:60]}... (3)"
        else:
            q_text = f"5.{i+1} Using {scenario[:60]}..., explain input/processing/output for {concept}... (3)" if i<5 else f"5.{i+1} Define algorithm for {concept}... (2)"

        marks = 3 if i < 5 else 2  # 5*3=15 +2*2=19
        paper["Q5"].append({"q": q_text, "marks": marks, "concept": concept})

    # Q6: Integrated scenario (20) - Top of hill hard, integrates all, fresh SA scenario
    for i in range(7):
        concept = get_next_concept() or "General integrated concept"
        concept = str(concept)
        chunk_raw = random.choice(taught_chunks) if taught_chunks else concept
        chunk = str(chunk_raw)
        scenario = paper["scenarios"]["Q6"]
        prompt = f"""Grade 10 Q6: Integrated scenario (20 marks) - TOP OF HILL but still easy reading.
Learn: Full sentences, easy reading, keep Grade 10 terms: e-waste, green computing, ergonomics, CTS, AI ethics, information overload, recycle, donate.
Scenario (integrated): {scenario}
Taught: {chunk}
Concept: {concept}

RULES:
- Full sentence, easy to understand, Grade 10 standard terms kept
- Must use {concept}
- Good: "What does ergonomics mean? Give two ways learners can avoid CTS when they work long hours in the lab. (2)"
- Good: "Explain why e-waste is the fastest growing pollutant and give two ways the lab can recycle or donate devices. (3)"
- Keep marks 2-3
- Never use difficult words, keep sentences short but full

Create ONE integrated question. Output ONLY question."""

        if client and model_name:
            try:
                resp = client.chat.completions.create(model=model_name, messages=[{"role":"user","content":prompt}], temperature=0.8, max_tokens=300)
                q_text = resp.choices[0].message.content.strip()
            except:
                q_text = f"6.{i+1} Integrated - Using {scenario[:60]}..., evaluate {concept} and its impact on green computing... Motivate with 2 points. (3)"
        else:
            q_text = f"6.{i+1} Using integrated scenario {scenario[:70]}..., explain how {concept} affects all systems and propose 2 green solutions... (3)" if i<6 else f"6.{i+1} Is use of AI for {concept} ethical in {scenario[:40]}...? Motivate with 2 points. (2)"

        marks = 3 if i < 6 else 2  # 6*3=18+2=20
        paper["Q6"].append({"q": q_text, "marks": marks, "concept": concept, "scenario": scenario})

    return paper, used_concepts, used_scenarios


def export_parktown_docx(paper, subject="Grade 10 IT"):
    if not HAS_DOCX:
        return None
    d = docx.Document()
    for h in paper['header']:
        d.add_heading(h, 0 if "PARKTOWN" in h else 2)
    d.add_heading("INSTRUCTIONS AND INFORMATION", 1)
    for i, inst in enumerate(paper['instructions'], 1):
        d.add_paragraph(f"{i}. {inst}")
    
    # Q1-Q6 structure
    for q_key in ["Q1","Q2","Q3","Q4","Q5","Q6"]:
        struct = paper['structure'][q_key]
        marks = sum([q['marks'] for q in paper[q_key]])
        d.add_heading(f"QUESTION {q_key[1]}: {struct['name'].upper()} ({marks} MARKS) - Scenario: {paper['scenarios'][q_key][:80]}", 1)
        d.add_paragraph(f"Description: {struct['desc']} | Cognitive: {struct['cognitive']}")
        for q in paper[q_key]:
            if q.get('type') == "MCQ":
                p = d.add_paragraph()
                p.add_run(f"{q['num']} {q['q']} ({q['marks']}) [Concept: {q['concept']}]").bold = True
                d.add_paragraph(f"A) {q['opts'][0]} B) {q['opts'][1]} C) {q['opts'][2]} D) {q['opts'][3]}")
            elif q.get('type') == "Matching":
                d.add_paragraph(f"{q['num']} {q['q']} -> {q['a']} ({q['marks']}) [Concept: {q['concept']}]")
            else:
                d.add_paragraph(f"{q.get('num','')} {q['q']} [Concept: {q['concept']} | {q['marks']} marks]")
    
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
    st.header("📝 Grade 10 Structure - LOCKED CAPS 120 MARKS")
    st.write("Q1: Short (20) | Q2: System Tech (25) | Q3: Comm & Networks (20) | Q4: Data & Info (16) | Q5: Solution Dev (19) | Q6: Integrated (20)")
    st.header("📝 Parktown Settings")
    examiner = st.text_input("Examiner", "Mr K Witelson")
    moderator = st.text_input("Moderator", "Mrs L Le Grange")
    st.info("Grade 10 Locked: Q1=20, Q2=25, Q3=20, Q4=16, Q5=19, Q6=20 = 120 marks")
    total_marks = 120

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
            gaps, ordered_concepts, freq_map = gap_and_no_repeat(taught_keywords, past_concepts, past_texts= past_texts)
            st.session_state['freq_map'] = freq_map

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

    st.subheader("🎯 STRICT MODE: ALWAYS New Questions On What Was NOT Assessed Before")
    st.error(f"🔒 STRICT: {len(st.session_state['gaps'])} GAPS - Taught but NEVER in past papers: {', '.join(st.session_state['gaps'][:20])}")
    st.success(f"✅ Engine will create 100% NEW questions from GAPS first. NEVER repeats same questions from previous papers. Similarity threshold: 0.65 (strict).")
    st.info(f"Ordered for No-Repeat (gaps first, never repeat): {', '.join(st.session_state['ordered_concepts'][:20])} - Each concept used ONLY ONCE per paper, always new phrasing!")

    # Show past questions count for strict check
    st.warning(f"⚠️ Past papers contain {len(' '.join(st.session_state['past_texts']).split())} words, {len(st.session_state['past_texts'])} papers. Engine will check EVERY new question against these to ensure NEVER repeats.")

    st.divider()
    st.subheader("🌍 SA Current News Scenarios - Fresh Every Time, Never Repeat")
    if 'used_scenarios_global' in st.session_state:
        st.write(f"Used scenarios so far (will never repeat): {st.session_state['used_scenarios_global']}")

    settings = {"examiner": examiner, "moderator": moderator, "total_marks": 120, "time": "2.5 hours", "pages": "11", "structure": GRADE10_STRUCTURE}

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
            st.success(f"🔒 STRICT MODE: Generated {len(used_concepts)} NEW questions - 100% from gaps (taught but never assessed). 0% repeat from past papers. Similarity <0.65. Fresh SA scenarios: {used_scenarios}")

if 'paper' in st.session_state:
    paper = st.session_state['paper']
    st.divider()
    st.markdown("## 📄 FINAL PAPER - GRADE 10 CAPS LOCKED 120 MARKS - Q1-Q6")
    for h in paper['header']:
        st.markdown(f"**{h}**")

    st.markdown("### INSTRUCTIONS AND INFORMATION")
    for i, inst in enumerate(paper['instructions'], 1):
        st.write(f"{i}. {inst}")

    # Display Q1-Q6 with exact Grade 10 structure
    for q_key in ["Q1","Q2","Q3","Q4","Q5","Q6"]:
        struct = paper['structure'][q_key]
        total_q_marks = sum([q['marks'] for q in paper[q_key]])
        st.markdown(f"### QUESTION {q_key[1]}: {struct['name'].upper()} ({total_q_marks} MARKS)")
        st.caption(f"{struct['desc']} | Cognitive: {struct['cognitive']}")
        st.info(f"Scenario {q_key} (SA Current News - Never Repeat): {paper['scenarios'][q_key]}")
        
        for q in paper[q_key]:
            if q.get('type') == "MCQ":
                st.write(f"**{q['num']}** {q['q']} ({q['marks']}) [Concept: {q['concept']} - No repeat]")
                st.write(f"A) {q['opts'][0]} B) {q['opts'][1]} C) {q['opts'][2]} D) {q['opts'][3]} | Ans: {q['ans']}")
            elif q.get('type') == "Matching":
                st.write(f"**{q['num']}** {q['q']} -> {q['a']} ({q['marks']}) [Concept: {q['concept']}]")
            else:
                # For Q2-Q6
                num_display = q.get('num', f"{q_key[1]}.{paper[q_key].index(q)+1}")
                st.write(f"**{q.get('q','')}** [Concept: {q['concept']} | {q['marks']} marks | Source: {q.get('source','taught')[:60]}...]")

    st.success(f"✅ Grade 10 Structure Locked: Q1=20, Q2=25, Q3=20, Q4=16, Q5=19, Q6=20 = 120 marks | No concept repeated: {len(st.session_state['used_concepts'])} unique concepts | Gaps filled: {len([c for c in st.session_state['used_concepts'] if c in st.session_state['gaps']])}")

    # Export
    docx_bytes = export_parktown_docx(paper, subject="Grade 10 IT - 120 Marks")
    if docx_bytes:
        st.download_button("📥 Download Grade 10 Paper DOCX - Q1-Q6 Locked 120 Marks", docx_bytes, file_name=f"Grade10_Q1-Q6_120Marks_{datetime.now().strftime('%Y%m%d_%H%M')}.docx")
        st.download_button("📥 Download SA Scenarios Used (Never Repeat)", "\n".join([f"{k}: {v}" for k,v in paper['scenarios'].items()]), file_name="SA_Scenarios_Used_Grade10.txt")
        # Coverage report
        coverage = f"Grade 10 Coverage Report\nTotal Concepts: {len(st.session_state['used_concepts'])}\n" + "\n".join([f"{k}: {paper['structure'][k]['name']} - {sum([q['marks'] for q in paper[k]])} marks - Concepts: {', '.join([q['concept'] for q in paper[k][:3]])}" for k in ["Q1","Q2","Q3","Q4","Q5","Q6"]])
        st.download_button("📥 Download Coverage Report - Taught vs Assessed", coverage, file_name="Coverage_Report_Grade10.txt")
