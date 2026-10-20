import re
from dataclasses import dataclass
from pathlib import Path

MAX_DOCUMENT_BYTES = 5 * 1024 * 1024
WORD_RE = re.compile(r"[\w'-]+", re.UNICODE)

@dataclass
class SearchResult:
    source: str
    excerpt: str
    score: int

class DocumentIndex:
    def __init__(self, chunk_words: int = 120, overlap_words: int = 20):
        if chunk_words < 1 or overlap_words < 0 or overlap_words >= chunk_words: raise ValueError("Require chunk_words > overlap_words >= 0.")
        self.chunk_words = chunk_words; self.overlap_words = overlap_words; self.chunks = []
    def add_files(self, paths):
        added = 0
        for item in paths:
            path = Path(item)
            if path.suffix.lower() not in {".txt", ".md"} or not path.is_file() or path.stat().st_size > MAX_DOCUMENT_BYTES: continue
            added += self.add_text(path.name, path.read_text(encoding="utf-8-sig"))
        return added
    def add_text(self, source: str, text: str) -> int:
        words = text.split(); stride = self.chunk_words - self.overlap_words; before = len(self.chunks)
        for start in range(0, len(words), stride):
            part = words[start:start + self.chunk_words]
            if not part: break
            self.chunks.append((Path(source).name, " ".join(part)))
            if start + self.chunk_words >= len(words): break
        return len(self.chunks) - before
    def search(self, query: str, top_k: int = 3):
        if top_k < 1: raise ValueError("top_k must be positive.")
        terms = {term.casefold() for term in WORD_RE.findall(query) if len(term) > 1}
        if not terms: return []
        results = []
        for source, excerpt in self.chunks:
            words = [word.casefold() for word in WORD_RE.findall(excerpt)]
            score = sum(words.count(term) for term in terms)
            if score: results.append(SearchResult(source, excerpt, score))
        return sorted(results, key=lambda result: (-result.score, result.source))[:top_k]
