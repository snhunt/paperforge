# PaperForge - Working Streamlit App

### How to run

1. Install:
```
pip install -r requirements.txt
```

2. Run:
```
streamlit run app.py
```

3. Open http://localhost:8501

### Modes

**Offline Mode (no API key):** 100% working, uses TF-IDF + template engine + novelty check. Guarantees originality, good for testing.

**LLM Mode (recommended):** Add OpenAI API key in sidebar. Uses gpt-4o-mini to generate natural questions grounded in your taught chunks, then runs cosine similarity check vs past papers (<0.85 threshold).

### Workflow
1. Upload taught content (PDF/DOCX/TXT) - lecture notes, syllabus
2. Upload past papers - engine extracts style, bloom, marks, question types
3. Click Process & Analyze
4. Set paper settings (marks, difficulty, Bloom)
5. Generate New Paper -> download DOCX + Memo

### Features implemented
- OCR via PyMuPDF
- Chunking with overlap
- TF-IDF keyword extraction
- Past paper pattern analysis (regex)
- Novelty check via cosine similarity
- Brand new generation with context randomization (SA contexts)
- Export to DOCX + TXT + Memo with marking guide
- Originality score per question

### Deploy to Streamlit Cloud / HuggingFace Spaces
Just push app.py + requirements.txt

### Extending
- Swap TF-IDF for sentence-transformers/all-MiniLM-L6-v2 for better semantic search
- Add ChromaDB persistent storage
- Add image/diagram questions via vision model
- Add coverage validator to ensure all topics covered
