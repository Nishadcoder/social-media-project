#!/usr/bin/env python3
"""
Duplicate Review Detection (Python port)

Detects duplicate / near-duplicate reviews using two classic text-similarity
techniques:
  * TF-IDF Cosine Similarity
  * Jaccard Similarity

GUI :  python duplicate_review_detection.py
CLI :  python duplicate_review_detection.py --cli --sample --method tfidf --threshold 70
       python duplicate_review_detection.py --cli --file reviews.txt --method jaccard

Only the Python standard library is used (tkinter for the GUI).
"""

from __future__ import annotations

import argparse
import math
import re
from collections import Counter
from typing import Dict, List, Tuple

# =====================================================================
#  NLP / SIMILARITY LOGIC (pure Python, no GUI)
# =====================================================================

STOPWORDS = {
    "the", "is", "a", "an", "and", "or", "to", "of", "in", "on", "for", "with",
    "this", "that", "it", "was", "were", "i", "my", "we", "our", "you", "your",
    "are", "be", "as", "at", "but", "so", "very", "just", "had", "have", "has",
    "its", "im", "it's",
}

SAMPLE = [
    "This app is amazing, I use it every day and love the interface.",
    "Amazing app, I use it every day and I really love the interface.",
    "Terrible customer service, I waited 2 hours for a reply.",
    "Customer service is terrible, I waited for two hours to get a reply.",
    "Great value for money, highly recommend to everyone.",
    "Highly recommend this, great value for money!",
    "The app keeps crashing every time I open the camera feature.",
    "Camera feature makes the app crash every single time I open it.",
    "Decent product but shipping took way too long.",
    "Not bad overall, just wish delivery was a bit faster.",
    "Battery life is excellent, lasts me a full two days easily.",
    "Instructions in the manual were confusing and poorly translated.",
]

EXPLAIN = {
    "tfidf": (
        "TF-IDF Cosine Similarity: each review is converted into a weighted "
        "vector where rare, distinctive words count more than common ones "
        "(Term Frequency x Inverse Document Frequency). Two reviews are compared "
        "using the cosine of the angle between their vectors - a value close to "
        "1 (100%) means the wording and topic overlap heavily, even if a few "
        "words differ."
    ),
    "jaccard": (
        "Jaccard Similarity: looks only at the SET of unique words in each "
        "review, ignoring how often each word repeats. Similarity = (words "
        "common to both) / (total unique words across both). Simpler and faster "
        "than TF-IDF, but more sensitive to small wording changes."
    ),
}

METHOD_NAMES = {"tfidf": "TF-IDF Cosine Similarity", "jaccard": "Jaccard Similarity"}


def tokenize(text: str) -> List[str]:
    """Lower-case, strip punctuation, split into words, drop stopwords."""
    text = text.lower().replace("'", "")          # "it's" -> "its" (a stopword)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return [w for w in text.split() if w and w not in STOPWORDS]


def build_tfidf(reviews_tokens: List[List[str]]) -> List[Dict[str, float]]:
    """One TF-IDF vector (dict term -> weight) per review.

    idf(t) = ln(N / (1 + df(t))) + 1      tf(t) = count(t) / len(review)
    """
    n = len(reviews_tokens)
    df: Counter = Counter()
    for toks in reviews_tokens:
        df.update(set(toks))
    idf = {t: math.log(n / (1 + d)) + 1 for t, d in df.items()}

    vectors = []
    for toks in reviews_tokens:
        total = len(toks) or 1
        tf = Counter(toks)
        vectors.append({t: (c / total) * idf[t] for t, c in tf.items()})
    return vectors


def cosine_sim(a: Dict[str, float], b: Dict[str, float]) -> float:
    dot = sum(v * b.get(k, 0.0) for k, v in a.items())
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def jaccard_sim(tok_a: List[str], tok_b: List[str]) -> float:
    a, b = set(tok_a), set(tok_b)
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def verdict_for(pct: int, threshold: int) -> str:
    if pct >= threshold:
        return "Duplicate"
    if pct >= threshold - 20:
        return "Similar"
    return "Unique"


def snippet(s: str, limit: int = 60) -> str:
    return s if len(s) <= limit else s[: limit - 3] + "..."


def analyze(reviews: List[str], method: str, threshold: int) -> Tuple[list, dict]:
    """Compare every pair of reviews.

    Returns (rows, stats) where rows is a list of dicts sorted by similarity
    (highest first): {i, j, sim, pct, verdict}.
    """
    tokens = [tokenize(r) for r in reviews]
    vecs = build_tfidf(tokens) if method == "tfidf" else None

    pairs = []
    for i in range(len(reviews)):
        for j in range(i + 1, len(reviews)):
            sim = cosine_sim(vecs[i], vecs[j]) if method == "tfidf" \
                else jaccard_sim(tokens[i], tokens[j])
            pct = int(sim * 100 + 0.5)             # round half up
            pairs.append({"i": i, "j": j, "sim": sim, "pct": pct,
                          "verdict": verdict_for(pct, threshold)})
    pairs.sort(key=lambda p: p["sim"], reverse=True)

    stats = {
        "reviews": len(reviews),
        "pairs": len(pairs),
        "duplicates": sum(p["verdict"] == "Duplicate" for p in pairs),
        "similar": sum(p["verdict"] == "Similar" for p in pairs),
    }
    return pairs, stats


# =====================================================================
#  CLI MODE
# =====================================================================

def read_reviews_file(path: str) -> List[str]:
    with open(path, encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]


def run_cli(args: argparse.Namespace) -> None:
    if args.file:
        reviews = read_reviews_file(args.file)
    elif args.sample:
        reviews = SAMPLE
    else:
        raise SystemExit("Use --sample or --file reviews.txt (one review per line).")
    if len(reviews) < 2:
        raise SystemExit("Need at least 2 reviews.")

    rows, stats = analyze(reviews, args.method, args.threshold)
    print(f"Method: {METHOD_NAMES[args.method]}   Threshold: {args.threshold}%\n")
    print(f"{'A':>4} {'B':>4} {'Sim':>6}  {'Verdict':<10} Review A / Review B")
    print("-" * 100)
    for p in rows:
        print(f"R{p['i']+1:<3} R{p['j']+1:<3} {p['pct']:>5}%  {p['verdict']:<10} "
              f"{snippet(reviews[p['i']], 38)}  |  {snippet(reviews[p['j']], 38)}")
    print(f"\nReviews: {stats['reviews']} | Pairs: {stats['pairs']} | "
          f"Duplicate pairs: {stats['duplicates']} | Similar pairs: {stats['similar']}")


# =====================================================================
#  GUI MODE (tkinter)
# =====================================================================

def run_gui() -> None:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    NAVY, TEAL = "#1B2A4A", "#0E7C7B"
    BG, CARD, TEXT, MUTED = "#F4F7FB", "#FFFFFF", "#263144", "#6B7686"
    RED_BG, AMBER_BG, GREEN_BG = "#F8D7D3", "#FBE8C4", "#D5EEDD"

    class App:
        def __init__(self, root: tk.Tk) -> None:
            self.root = root
            root.title("Duplicate Review Detection")
            root.configure(bg=BG)
            root.geometry("1100x820")
            self._build_ui()
            self._update_explain()

        # ---------- helpers ----------
        def _card(self, parent, title, expand=False):
            f = tk.Frame(parent, bg=CARD, highlightbackground="#E2E8F0",
                         highlightthickness=1)
            f.pack(fill="both" if expand else "x", expand=expand, padx=14, pady=(10, 0))
            tk.Label(f, text=title, bg=CARD, fg=NAVY,
                     font=("Segoe UI", 11, "bold")).pack(anchor="w", padx=12, pady=(10, 4))
            return f

        def _build_ui(self) -> None:
            header = tk.Frame(self.root, bg=NAVY)
            header.pack(fill="x")
            tk.Label(header, text="Duplicate Review Detection", bg=NAVY, fg="white",
                     font=("Segoe UI", 16, "bold")).pack(anchor="w", padx=18, pady=(12, 0))
            tk.Label(header, text="NLP-based tool to flag duplicate / near-duplicate "
                     "reviews using TF-IDF Cosine Similarity or Jaccard Similarity.",
                     bg=NAVY, fg="#CBD6EA", font=("Segoe UI", 10)
                     ).pack(anchor="w", padx=18, pady=(0, 12))

            # --- 1. input ---
            c1 = self._card(self.root, "1. Input Reviews")
            self.input = tk.Text(c1, height=7, wrap="word", font=("Segoe UI", 10),
                                 bg=BG, fg=TEXT, relief="flat", padx=8, pady=6)
            self.input.pack(fill="x", padx=12)
            self.input.bind("<KeyRelease>", lambda e: self._refresh_list())

            row = tk.Frame(c1, bg=CARD)
            row.pack(fill="x", padx=12, pady=8)
            tk.Button(row, text="Load Sample Reviews", command=self.load_sample
                      ).pack(side="left", padx=(0, 6))
            tk.Button(row, text="Load from File...", command=self.load_file
                      ).pack(side="left", padx=(0, 6))
            tk.Button(row, text="Clear", command=self.clear).pack(side="left", padx=(0, 14))

            mf = tk.Frame(row, bg=CARD)
            mf.pack(side="left", padx=8)
            tk.Label(mf, text="Similarity Method", bg=CARD, fg=MUTED,
                     font=("Segoe UI", 9)).pack(anchor="w")
            self.method_cb = ttk.Combobox(mf, state="readonly", width=28,
                                          values=[METHOD_NAMES["tfidf"],
                                                  METHOD_NAMES["jaccard"]])
            self.method_cb.current(0)
            self.method_cb.pack()
            self.method_cb.bind("<<ComboboxSelected>>", lambda e: self._update_explain())

            tf = tk.Frame(row, bg=CARD)
            tf.pack(side="left", padx=8)
            self.thresh_lbl = tk.Label(tf, text="Duplicate Threshold: 70%", bg=CARD,
                                       fg=MUTED, font=("Segoe UI", 9))
            self.thresh_lbl.pack(anchor="w")
            self.thresh = tk.Scale(tf, from_=30, to=95, orient="horizontal", length=200,
                                   showvalue=False, bg=CARD, highlightthickness=0,
                                   command=self._on_thresh)
            self.thresh.set(70)
            self.thresh.pack()

            tk.Button(row, text="Detect Duplicates", command=self.run_analysis,
                      bg=TEAL, fg="white", font=("Segoe UI", 10, "bold"),
                      relief="flat", padx=12, pady=6).pack(side="right")

            self.explain = tk.Label(c1, text="", bg=CARD, fg=MUTED, wraplength=1030,
                                    justify="left", font=("Segoe UI", 9))
            self.explain.pack(anchor="w", padx=12, pady=(0, 10))

            # --- 2. reviews list ---
            c2 = self._card(self.root, "2. Reviews Entered")
            self.listbox = tk.Listbox(c2, height=5, font=("Segoe UI", 9), bg=BG,
                                      fg=TEXT, relief="flat", activestyle="none")
            self.listbox.pack(fill="x", padx=12, pady=(0, 10))

            # --- 3. results ---
            c3 = self._card(self.root, "3. Pairwise Similarity Results", expand=True)
            self.stats = tk.Label(c3, text="", bg=CARD, fg=TEXT,
                                  font=("Segoe UI", 10, "bold"))
            self.stats.pack(anchor="w", padx=12)

            wrap = tk.Frame(c3, bg=CARD)
            wrap.pack(fill="both", expand=True, padx=12, pady=(4, 12))
            cols = ("a", "b", "sim", "verdict")
            self.tree = ttk.Treeview(wrap, columns=cols, show="headings", height=10)
            for col, text, w in (("a", "Review A", 380), ("b", "Review B", 380),
                                 ("sim", "Similarity", 90), ("verdict", "Verdict", 100)):
                self.tree.heading(col, text=text)
                self.tree.column(col, width=w, anchor="w" if col in "ab" else "center")
            sb = ttk.Scrollbar(wrap, orient="vertical", command=self.tree.yview)
            self.tree.configure(yscrollcommand=sb.set)
            self.tree.pack(side="left", fill="both", expand=True)
            sb.pack(side="right", fill="y")
            self.tree.tag_configure("Duplicate", background=RED_BG)
            self.tree.tag_configure("Similar", background=AMBER_BG)
            self.tree.tag_configure("Unique", background=GREEN_BG)

        # ---------- state ----------
        @property
        def method(self) -> str:
            return "tfidf" if self.method_cb.current() == 0 else "jaccard"

        def _reviews(self) -> List[str]:
            return [ln.strip() for ln in self.input.get("1.0", "end").splitlines()
                    if ln.strip()]

        def _update_explain(self) -> None:
            self.explain.config(text=EXPLAIN[self.method])

        def _on_thresh(self, val) -> None:
            self.thresh_lbl.config(text=f"Duplicate Threshold: {int(float(val))}%")

        def _refresh_list(self) -> None:
            self.listbox.delete(0, "end")
            reviews = self._reviews()
            if not reviews:
                self.listbox.insert("end", "No reviews yet - load the sample or paste your own.")
                return
            for i, r in enumerate(reviews, 1):
                self.listbox.insert("end", f"#R{i}   {r}")

        def _clear_results(self, message: str = "") -> None:
            self.tree.delete(*self.tree.get_children())
            self.stats.config(text=message)

        # ---------- actions ----------
        def load_sample(self) -> None:
            self.input.delete("1.0", "end")
            self.input.insert("1.0", "\n".join(SAMPLE))
            self._refresh_list()

        def load_file(self) -> None:
            path = filedialog.askopenfilename(
                title="Open reviews file",
                filetypes=[("Text files", "*.txt *.csv"), ("All files", "*.*")])
            if not path:
                return
            try:
                reviews = read_reviews_file(path)
            except (OSError, UnicodeDecodeError) as e:
                messagebox.showerror("Could not read file", str(e))
                return
            self.input.delete("1.0", "end")
            self.input.insert("1.0", "\n".join(reviews))
            self._refresh_list()

        def clear(self) -> None:
            self.input.delete("1.0", "end")
            self._refresh_list()
            self._clear_results()

        def run_analysis(self) -> None:
            reviews = self._reviews()
            self._refresh_list()
            self._clear_results()
            if len(reviews) < 2:
                self.stats.config(
                    text="Add at least 2 reviews, then click Detect Duplicates.")
                return
            rows, st = analyze(reviews, self.method, int(self.thresh.get()))
            for p in rows:
                self.tree.insert("", "end", tags=(p["verdict"],), values=(
                    f"#R{p['i']+1}  {snippet(reviews[p['i']])}",
                    f"#R{p['j']+1}  {snippet(reviews[p['j']])}",
                    f"{p['pct']}%", p["verdict"]))
            self.stats.config(
                text=f"{st['reviews']} Reviews   |   {st['pairs']} Pairs Compared   |   "
                     f"{st['duplicates']} Duplicate Pairs   |   {st['similar']} Similar Pairs")

    root = tk.Tk()
    app = App(root)
    app._refresh_list()
    root.mainloop()


# =====================================================================

def main() -> None:
    ap = argparse.ArgumentParser(description="Duplicate Review Detection")
    ap.add_argument("--cli", action="store_true", help="run in terminal without GUI")
    ap.add_argument("--sample", action="store_true", help="(CLI) use built-in sample reviews")
    ap.add_argument("--file", help="(CLI) text file with one review per line")
    ap.add_argument("--method", choices=["tfidf", "jaccard"], default="tfidf")
    ap.add_argument("--threshold", type=int, default=70, help="duplicate threshold, 30-95")
    args = ap.parse_args()
    if args.cli:
        run_cli(args)
    else:
        run_gui()


if __name__ == "__main__":
    main()
