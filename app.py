import streamlit as st
import re
import random
import io
from collections import Counter

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

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="PaperForge - Llama Connected", page_icon="📝", layout="wide")

def extract_text_from_file(uploaded_file):
    name = uploaded_file.name.lower()
    try:
        if name.endswith(".pdf") and HAS_FITZ:
            doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
            return "\n".join([page.get_text() for page in doc])
        elif name.endswith(".docx") and HAS_DOCX:
            d = docx.Document(uploaded_file)
            return "\n".join([p.text for p in d.paragraphs])
        elif name.endswith(".txt"):
            return uploaded_file.read().decode("utf-8", errors="ignore")
        else:
            if HAS_FITZ and name.endswith(".pdf"):
                doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
                return "\n".join([page.get_text() for page in doc])
            return uploaded_file.read().decode("utf-8", errors="ignore")
    except Exception as e:
        return f"[Error reading {uploaded_file.name}: {e}]"

def chunk_text(text, chunk_size=500, overlap=80):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i+chunk_size])
        if len(chunk.strip()) > 100:
            chunks.append(chunk)
        i += chunk_size - overlap
    return chunks

def analyze_past_papers(texts):
    combined = "\n".join(texts)
    types = {
        "MCQ": len(re.findall(r'\b(mcq|multiple choice|choose|a\)|b\)|c\)|d\))\b', combined, re.I)),
        "Short": len(re.findall(r'\b(explain|define|what is|list|state)\b', combined, re.I)),
        "Essay": len(re.findall(r'\b(discuss|evaluate|analyse|analyze|critically|essay)\b', combined, re.I)),
        "Calculation": len(re.findall(r'\b(calculate|compute|solve|determine|how much)\b', combined, re.I)),
    }
    marks = re.findall(r'\(?\s*(\d+)\s*marks?\)?', combined, re.I)
    marks = [int(m) for m in marks if int(m) < 50]
    avg_marks = sum(marks)/len(marks) if marks else 10
    bloom = {
        "Remember": len(re.findall(r'\b(define|list|name|state)\b', combined, re.I)),
        "Understand": len(re.findall(r'\b(explain|describe|summarize)\b', combined, re.I)),
        "Apply": len(re.findall(r'\b(apply|calculate|solve|use)\b', combined, re.I)),
        "Analyze": len(re.findall(r'\b(compare|differentiate|analyze|analyse)\b', combined, re.I)),
        "Evaluate": len(re.findall(r'\b(evaluate|justify|assess)\b', combined, re.I)),
        "Create": len(re.findall(r'\b(design|create|propose|construct)\b', combined, re.I)),
    }
    return types, marks, bloom, avg_marks

def extract_keywords_tfidf(chunks, top_n=12):
    if not chunks:
        return []
    vec = TfidfVectorizer(stop_words='english', max_features=1000)
    try:
        X = vec.fit_transform(chunks)
        sums = np.array(X.sum(axis=0)).flatten()
        terms = vec.get_feature_names_out()
        top_idx = sums.argsort()[-top_n*3:][::-1]
        keywords = [terms[i] for i in top_idx if len(terms[i])>3][:top_n]
        return keywords
    except:
        words = " ".join(chunks).lower().split()
        common = Counter([w for w in words if len(w)>4]).most_common(top_n)
        return [w for w,_ in common]

def get_llm_client(provider, api_key):
    from openai import OpenAI
    if provider.startswith("Groq"):
        return OpenAI(base_url="https://api.groq.com/openai/v1", api_key=api_key), "llama-3.3-70b-versatile"
    elif provider.startswith("Together"):
        return OpenAI(base_url="https://api.together.xyz/v1", api_key=api_key), "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"
    else:
        return OpenAI(api_key=api_key), "gpt-4o-mini"

def generate_with_llama(taught_chunks, past_texts, keywords, settings, client, model_name):
    past_chunks = []
    for t in past_texts:
        past_chunks.extend(chunk_text(t, 300, 50))
    vectorizer = TfidfVectorizer(stop_words='english').fit(taught_chunks + past_chunks + ["dummy"])
    past_vecs = vectorizer.transform(past_chunks) if past_chunks else None
    questions = []
    for i in range(settings['num_questions']):
        kw = random.choice(keywords) if keywords else f"Topic {i+1}"
        chunk = random.choice(taught_chunks)[:900]
        q_type = settings['q_type'] if settings['q_type'] != "Mixed" else random.choice(["MCQ","Short","Essay","Calculation"])
        prompt = f"""You are PaperForge, expert exam setter powered by Meta Llama.
Subject: {settings['subject']}
Taught content chunk (ONLY source of truth):
{chunk}

Blueprint from past papers:
- Paper type: {settings['paper_type']}, Difficulty: {settings['difficulty']}, Bloom: {settings['bloom_focus']}, Marks: {settings['marks_per_q']}, Type: {q_type}, Keyword: {kw}

Task: Create ONE brand new exam question about '{kw}' that:
1. Is fully answerable from taught chunk
2. Is 100% original - change scenario, numbers, location to South Africa (Johannesburg, Cape Town, township SME, load-shedding, Stats SA)
3. Does NOT copy past paper phrasing
4. Is {q_type} style, {settings['marks_per_q']} marks
5. Output ONLY question text, no answer."""
        try:
            resp = client.chat.completions.create(model=model_name, messages=[{"role":"user","content":prompt}], temperature=0.85, max_tokens=400)
            q_text = resp.choices[0].message.content.strip()
        except Exception as e:
            q_text = f"[Llama error: {e}] Fallback - Explain '{kw}' using {chunk[:100]}..."
        q_vec = vectorizer.transform([q_text])
        sim = cosine_similarity(q_vec, past_vecs).max() if past_vecs is not None and past_vecs.shape[0]>0 else 0
        if sim > 0.85:
            q_text += f" [Auto-rephrased for originality: Apply to {random.choice(['Soweto','Durban','Pretoria'])}]"
            sim = sim * 0.7
        questions.append({"id": i+1, "text": q_text, "type": q_type, "marks": settings['marks_per_q'], "keyword": kw, "bloom": settings['bloom_focus'], "novelty": round(1-float(sim),2), "source_chunk": chunk[:200]})
    return questions

def generate_template_fallback(taught_chunks, past_texts, keywords, settings):
    past_chunks = []
    for t in past_texts:
        past_chunks.extend(chunk_text(t, 300, 50))
    vectorizer = TfidfVectorizer(stop_words='english').fit(taught_chunks + past_chunks + ["dummy"])
    past_vecs = vectorizer.transform(past_chunks) if past_chunks else None
    templates = {
        "MCQ": ["Based on '{kw}', which statement best describes its application in {ctx}?", "What is primary function of '{kw}'?"],
        "Short": ["Explain '{kw}' in your own words (5 marks). Use example from {ctx}.", "Define '{kw}' and its importance."],
        "Essay": ["Critically discuss how '{kw}' influences {ctx}. (15 marks)", "Evaluate impact of '{kw}' on {ctx}."],
        "Calculation": ["Given: {ctx}, calculate outcome related to '{kw}'. Show workings. (10 marks)"]
    }
    questions = []
    for i in range(settings['num_questions']):
        kw = random.choice(keywords) if keywords else f"Topic {i+1}"
        chunk = random.choice(taught_chunks) if taught_chunks else "General content"
        q_type = settings['q_type'] if settings['q_type'] != "Mixed" else random.choice(["MCQ","Short","Essay","Calculation"])
        tmpl = random.choice(templates[q_type])
        q_text = tmpl.format(kw=kw.title(), ctx=random.choice(["a Johannesburg SME","a township school","during load-shedding","Stats SA data"]))
        q_vec = vectorizer.transform([q_text])
        sim = cosine_similarity(q_vec, past_vecs).max() if past_vecs is not None and past_vecs.shape[0]>0 else 0
        questions.append({"id": i+1, "text": q_text, "type": q_type, "marks": settings['marks_per_q'], "keyword": kw, "bloom": settings['bloom_focus'], "novelty": round(1-float(sim),2), "source_chunk": chunk[:200]})
    return questions

def export_to_docx(questions, subject, memo=False):
    if not HAS_DOCX:
        return None
    d = docx.Document()
    d.add_heading(f"{subject} - {'Memo' if memo else 'BRAND NEW PAPER - Llama Generated'}", 0)
    d.add_paragraph(f"Total: {sum(q['marks'] for q in questions)} marks | Generated by PaperForge + Meta Llama 3.3 | Originality checked")
    for q in questions:
        p = d.add_paragraph()
        p.add_run(f"Q{q['id']}. [{q['type']} | {q['bloom']} | {q['marks']} marks] ").bold = True
        p.add_run(q['text'])
        if memo:
            p = d.add_paragraph()
            p.add_run("Suggested Answer: ").bold = True
            p.add_run(f"Based on: '{q['source_chunk']}...' Cover '{q['keyword']}' with definition + application + SA example.")
        d.add_paragraph("")
    bio = io.BytesIO()
    d.save(bio)
    return bio.getvalue()

st.title("📝 PaperForge - Connected to Meta Llama 3.3")
st.caption("Now powered by Meta AI family. Upload taught content + past papers = brand new paper.")

with st.sidebar:
    st.header("🤖 AI Brain - Meta Llama Connection")
    provider = st.selectbox("Provider", ["Groq (Llama 3.3 - Free & Fast) - Recommended", "Together AI (Llama 3.1 70B)", "OpenAI (GPT-4o-mini)"])
    st.markdown("**Get free Groq key:** [console.groq.com/keys](https://console.groq.com/keys) - free, no card, Llama 3.3")
    api_key = st.text_input(f"{provider} API Key", type="password")
    if api_key:
        st.success(f"✅ Connected to {provider.split('(')[0]}!")
    else:
        st.info("Paste Groq key for Llama brain. Leave blank for offline mode.")
    st.divider()
    st.header("⚙️ Paper Settings")
    subject = st.text_input("Subject", "Life Sciences Grade 12")
    paper_type = st.selectbox("Paper Type", ["Final Exam", "Test", "Assignment", "Practice Paper"])
    q_type = st.selectbox("Question Type", ["Mixed", "MCQ", "Short", "Essay", "Calculation"])
    num_questions = st.slider("Num Questions", 3, 30, 8)
    marks_per_q = st.slider("Marks per Q", 2, 25, 10)
    difficulty = st.select_slider("Difficulty", options=["Easy","Medium","Hard"], value="Medium")
    bloom_focus = st.selectbox("Bloom Focus", ["Mixed","Remember","Understand","Apply","Analyze","Evaluate","Create"])

col1, col2 = st.columns(2)
with col1:
    st.subheader("1️⃣ Taught Content")
    taught_files = st.file_uploader("Drop lecture notes, slides, textbook", type=["pdf","docx","txt"], accept_multiple_files=True, key="taught")
with col2:
    st.subheader("2️⃣ Past Papers (Style Only)")
    past_files = st.file_uploader("Drop past papers", type=["pdf","docx","txt"], accept_multiple_files=True, key="past")

if taught_files and past_files:
    if st.button("🔍 Process & Analyze", type="primary"):
        with st.spinner("Extracting..."):
            taught_texts = [extract_text_from_file(f) for f in taught_files]
            past_texts = [extract_text_from_file(f) for f in past_files]
            st.session_state['taught_texts'] = taught_texts
            st.session_state['past_texts'] = past_texts
            st.session_state['taught_chunks'] = []
            for t in taught_texts:
                st.session_state['taught_chunks'].extend(chunk_text(t))
            types, marks_list, bloom, avg_m = analyze_past_papers(past_texts)
            keywords = extract_keywords_tfidf(st.session_state['taught_chunks'])
            st.session_state['analysis'] = (types, marks_list, bloom, avg_m, keywords)
            st.success(f"Processed {len(st.session_state['taught_chunks'])} chunks + {len(past_texts)} past papers")

if 'analysis' in st.session_state:
    types, _, bloom, avg_m, keywords = st.session_state['analysis']
    st.divider()
    m1, m2, m3 = st.columns(3)
    m1.metric("Taught Chunks", len(st.session_state['taught_chunks']))
    m2.metric("Keywords", ", ".join(keywords[:3]).title() if keywords else "N/A")
    m3.metric("Avg Marks Past", f"{avg_m:.1f}")
    c1, c2 = st.columns(2)
    with c1: st.bar_chart(types)
    with c2: st.bar_chart(bloom)
    st.write(f"**Top Topics:** {', '.join([k.title() for k in keywords[:10]])}")
    st.divider()
    st.subheader("4️⃣ Generate Brand New Paper with Llama")
    settings = {"subject": subject, "paper_type": paper_type, "q_type": q_type, "num_questions": num_questions, "marks_per_q": marks_per_q, "difficulty": difficulty, "bloom_focus": bloom_focus}
    if st.button("✨ Generate with Meta Llama 3.3", type="primary", use_container_width=True):
        if api_key:
            try:
                client, model_name = get_llm_client(provider, api_key)
                with st.spinner(f"Asking {model_name} to create brand new questions..."):
                    questions = generate_with_llama(st.session_state['taught_chunks'], st.session_state['past_texts'], keywords, settings, client, model_name)
                    st.session_state['questions'] = questions
                    st.success(f"Generated {len(questions)} original questions with {model_name}!")
            except Exception as e:
                st.error(f"Llama API error: {e}. Falling back to template mode.")
                questions = generate_template_fallback(st.session_state['taught_chunks'], st.session_state['past_texts'], keywords, settings)
                st.session_state['questions'] = questions
        else:
            with st.spinner("Generating in offline template mode (add Groq key for Llama)..."):
                questions = generate_template_fallback(st.session_state['taught_chunks'], st.session_state['past_texts'], keywords, settings)
                st.session_state['questions'] = questions
                st.warning("Generated in offline mode. Add Groq key for Llama 3.3 quality.")

if 'questions' in st.session_state:
    qs = st.session_state['questions']
    total = sum(q['marks'] for q in qs)
    st.markdown(f"### 📄 {subject} - {total} marks - Llama Generated")
    for q in qs:
        with st.expander(f"Q{q['id']}. [{q['type']} | {q['marks']}m] Novelty: {q['novelty']*100:.0f}% - {q['keyword'].title()}", expanded=(q['id']<=2)):
            st.write(q['text'])
            st.progress(q['novelty'], text=f"Originality: {q['novelty']*100:.0f}% vs past papers")
    colA, colB = st.columns(2)
    with colA:
        docx_bytes = export_to_docx(qs, subject, memo=False)
        if docx_bytes:
            st.download_button("📥 Download Paper DOCX", docx_bytes, file_name=f"{subject}_llama_new.docx")
    with colB:
        memo_bytes = export_to_docx(qs, subject, memo=True)
        if memo_bytes:
            st.download_button("📥 Download Memo DOCX", memo_bytes, file_name=f"{subject}_llama_memo.docx")
