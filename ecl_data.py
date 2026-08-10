"""Daten-Pipeline: Quellen, Few-Shot-Exemplare und Verifier.

Vereinheitlicht synthetische und reale, verifizierbare Datensätze auf ein
gemeinsames Schema `ECLExample`. Jedes Beispiel trägt:
  - die Frage und die akzeptierten Gold-Antworten,
  - den Aufgabentyp + Verifier-Art (numeric vs. lookup),
  - ein `tool_needed_label` (für Tool-Precision/Recall & Over-/Under-Retrieval, §11/§12),
  - die Oracle-Evidenz, die `get_evidence` zurückgibt (kostenpflichtiges externes Signal).

Designentscheidung: `get_evidence` ist ein *Oracle-Tool* (Rechner bzw. Retriever),
das die Gold-Information zu Kosten liefert. Damit isoliert v0 die Kernfrage —
*wann* lohnt sich Evidenz vs. direkt antworten vs. enthalten — statt das Lösen
selbst zu trainieren (das gehört zu reason_more, in v0 ausgeklammert).
"""

import json
import random
import re
import string
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Iterator


@dataclass
class ECLExample:
    question: str
    answers: List[str]                       # akzeptierte Gold-Antworten (>=1)
    task_type: str                           # "arithmetic" | "math" | "qa"
    verifier_kind: str                       # "numeric" | "lookup"
    evidence: str                            # was get_evidence (Oracle) zurückgibt
    tool_needed_label: Optional[bool] = None # ob Evidenz erwartbar hilft (für Metriken)
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def gold(self) -> str:
        return self.answers[0]


# ----------------------------------------------------------------------
# Quellen
# ----------------------------------------------------------------------
class DataSource:
    """Basisklasse: iterierbar, zieht zufällig (mit Replacement) ein Beispiel."""
    def __init__(self, examples: List[ECLExample], shuffle: bool = True):
        self._examples = examples
        self._shuffle = shuffle
        self._order: List[int] = []

    def __len__(self) -> int:
        return len(self._examples)

    def sample(self) -> ECLExample:
        if not self._order:
            self._order = list(range(len(self._examples)))
            if self._shuffle:
                random.shuffle(self._order)
        return self._examples[self._order.pop()]

    def __iter__(self) -> Iterator[ECLExample]:
        return iter(self._examples)


class SyntheticArithmeticSource(DataSource):
    """Kleine arithmetische Aufgaben — Smoke-Tests & schnelle Iteration."""
    def __init__(self, n: int = 512, seed: int = 0):
        rng = random.Random(seed)
        ops = ["+", "-", "*"]
        ex: List[ECLExample] = []
        for _ in range(n):
            op = rng.choice(ops)
            a, b = rng.randint(1, 50), rng.randint(1, 50)
            ans = {"+": a + b, "-": a - b, "*": a * b}[op]
            ex.append(ECLExample(
                question=f"Was ist {a} {op} {b}?",
                answers=[str(ans)],
                task_type="arithmetic",
                verifier_kind="numeric",
                evidence=f"Berechnung: {a} {op} {b} = {ans}.",
                tool_needed_label=(op == "*"),  # Multiplikation = härter -> Tool hilft eher
                meta={"op": op},
            ))
        super().__init__(ex)


def load_gsm8k(split: str = "train", limit: Optional[int] = 1000) -> DataSource:
    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main", split=split)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    ex: List[ECLExample] = []
    for row in ds:
        gold = row["answer"].split("####")[-1].strip().replace(",", "")
        steps = row["answer"].count("<<")  # Anzahl Rechenschritte als Schwierigkeitsproxy
        ex.append(ECLExample(
            question=row["question"],
            answers=[gold],
            task_type="math",
            verifier_kind="numeric",
            evidence=f"Die Lösung ist {gold}.",
            tool_needed_label=(steps >= 3),  # mehrschrittig -> externes Rechnen lohnt eher
            meta={"steps": steps},
        ))
    return DataSource(ex)


def load_popqa(split: str = "test", limit: Optional[int] = 1000,
               pop_threshold: int = 1000) -> DataSource:
    """PopQA: Faktenfragen. tool_needed_label aus Subjekt-Popularität abgeleitet —
    long-tail (niedrige Popularität) => Retrieval hilft; populär => Baseline reicht oft.
    Genau der saubere Over-/Under-Retrieval-Hebel aus §12."""
    from datasets import load_dataset
    ds = load_dataset("akariasai/PopQA", split=split)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    ex: List[ECLExample] = []
    for row in ds:
        try:
            answers = json.loads(row["possible_answers"])
        except (json.JSONDecodeError, TypeError):
            answers = [str(row["obj"])]
        answers = [a for a in answers if a] or [str(row["obj"])]
        s_pop = row.get("s_pop") or 0
        ex.append(ECLExample(
            question=row["question"],
            answers=answers,
            task_type="popqa",
            verifier_kind="lookup",
            evidence=f"Laut Wissensquelle: {answers[0]}.",
            tool_needed_label=(s_pop < pop_threshold),
            # PopQA-Standard (Mallen et al. 2212.10511): "Q:/A:" + Substring-Treffer
            meta={"s_pop": s_pop, "subj": row.get("subj"),
                  "metric": "substring", "qa_style": "short"},
        ))
    return DataSource(ex)


def load_webquestions(split: str = "train", limit: Optional[int] = 1000) -> DataSource:
    """WebQuestions: kurze Faktoid-Fragen (Google-Anfragen). Leichter/mehr direkt-lösbar als
    PopQA-long-tail -> liefert commit-würdige Diversität fürs Spektrum."""
    from datasets import load_dataset
    ds = load_dataset("web_questions", split=split)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    ex: List[ECLExample] = []
    for row in ds:
        answers = [a for a in row["answers"] if a] or ["?"]
        ex.append(ECLExample(
            question=row["question"], answers=answers,
            task_type="webq", verifier_kind="lookup",
            evidence=f"Laut Wissensquelle: {answers[0]}.",
            # lm-evaluation-harness webqs: "Question:/Answer:" + Exact-Match
            tool_needed_label=None,
            meta={"src": "webq", "metric": "exact_match", "qa_style": "long"},
        ))
    return DataSource(ex)


def load_triviaqa(split: str = "train", limit: Optional[int] = 1000) -> DataSource:
    """TriviaQA (rc.nocontext): Trivia-Faktoide; answer.value + Aliase."""
    from datasets import load_dataset
    ds = load_dataset("trivia_qa", "rc.nocontext", split=split)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    ex: List[ECLExample] = []
    for row in ds:
        a = row["answer"]
        answers = [a.get("value")] + list(a.get("aliases", []))
        answers = [x for x in answers if x] or [a.get("value") or "?"]
        ex.append(ECLExample(
            question=row["question"], answers=answers,
            task_type="triviaqa", verifier_kind="lookup",
            evidence=f"Laut Wissensquelle: {answers[0]}.",
            # lm-evaluation-harness triviaqa: "Question:/Answer:" + Exact-Match
            tool_needed_label=None,
            meta={"src": "triviaqa", "metric": "exact_match", "qa_style": "long"},
        ))
    return DataSource(ex)


def load_svamp(split: str = "train", limit: Optional[int] = 1000) -> DataSource:
    """SVAMP: einfache mathematische Wortprobleme (numerisch). Leichter als GSM8K → das Modell
    löst *einige* direkt → gemischte Direkt-Korrektheit *innerhalb* Mathe (saubere within-task-Varianz)."""
    from datasets import load_dataset
    ds = load_dataset("ChilleD/SVAMP", split=split)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    ex: List[ECLExample] = []
    for row in ds:
        gold = str(row["Answer"]).strip()
        ex.append(ECLExample(
            question=row["question_concat"], answers=[gold],
            task_type="svamp", verifier_kind="numeric",
            evidence=f"Die Lösung ist {gold}.", tool_needed_label=None, meta={"src": "svamp"},
        ))
    return DataSource(ex)


class MixedSource(DataSource):
    """Mischt mehrere Quellen gewichtet (z.B. GSM8K + PopQA)."""
    def __init__(self, sources: List[DataSource], weights: Optional[List[float]] = None):
        self.sources = sources
        self.weights = weights or [1.0] * len(sources)
        all_ex: List[ECLExample] = [e for s in sources for e in s]
        super().__init__(all_ex)

    def sample(self) -> ECLExample:
        src = random.choices(self.sources, weights=self.weights, k=1)[0]
        return src.sample()


_LOADERS = {
    "synthetic": lambda lim: SyntheticArithmeticSource(n=lim),
    "gsm8k": lambda lim: load_gsm8k(limit=lim),
    "popqa": lambda lim: load_popqa(limit=lim),
    "webq": lambda lim: load_webquestions(limit=lim),
    "triviaqa": lambda lim: load_triviaqa(limit=lim),
    "svamp": lambda lim: load_svamp(limit=lim),
}
_ALIASES = {"mixed": "gsm8k+popqa", "all": "synthetic+gsm8k+popqa+webq"}


def build_source(name: str, **kw) -> DataSource:
    """Factory: beliebige '+'-getrennte Kombination aus {synthetic, gsm8k, popqa, webq, triviaqa}.
    Aliase: 'mixed'=gsm8k+popqa, 'all'=synthetic+gsm8k+popqa+webq."""
    lim = kw.get("limit", 512)
    name = _ALIASES.get(name.lower(), name.lower())
    parts = [p for p in name.split("+") if p]
    unknown = [p for p in parts if p not in _LOADERS]
    if unknown:
        raise ValueError(f"Unbekannte Quelle(n): {unknown} (erlaubt: {list(_LOADERS)})")
    sources = [_LOADERS[p](lim) for p in parts]
    return sources[0] if len(sources) == 1 else MixedSource(sources)


# ----------------------------------------------------------------------
# Tool (Oracle) + Verifier, beide example-getrieben
# ----------------------------------------------------------------------
class OracleTool:
    """get_evidence: liefert die Oracle-Evidenz des Beispiels (Rechner/Retriever)."""
    def execute(self, example: ECLExample) -> str:
        return example.evidence


class ExampleVerifier:
    """Verifiziert eine Antwort gegen die Gold-Antworten eines ECLExample."""

    @staticmethod
    def _first_number(text: str) -> Optional[float]:
        m = re.search(r"-?\d+(?:\.\d+)?", str(text).replace(",", ""))
        return float(m.group(0)) if m else None

    @staticmethod
    def _norm_em(text: str) -> str:
        """SQuAD-/lm-evaluation-harness-Normalisierung: lowercase, Artikel raus,
        Interpunktion raus, Whitespace vereinheitlicht."""
        t = str(text).lower()
        t = re.sub(r"\b(a|an|the)\b", " ", t)
        t = "".join(c for c in t if c not in set(string.punctuation))
        return " ".join(t.split())

    def verify(self, answer: str, example: ECLExample, metric: Optional[str] = None) -> bool:
        """metric: None = Default des Datensatzes (`example.meta['metric']`), sonst erzwungen.

        Die Benchmarks nutzen UNTERSCHIEDLICHE Metriken, und das ist kein Detail:
          * PopQA (Mallen et al. 2212.10511): Treffer, wenn ein Gold-Alias Teilstring der
            Vorhersage ist — bewusst nachsichtig.
          * TriviaQA / WebQuestions (lm-evaluation-harness): normalisierter Exact-Match.
        Vorher lief auf allen dreien der Substring-Abgleich; für TriviaQA/WebQ war das zu
        nachsichtig und begünstigte lange Antworten (results.md Run 27).
        """
        m = metric or example.meta.get("metric") or ("numeric" if example.verifier_kind == "numeric"
                                                     else "substring")
        if m == "exact_match":
            a = self._norm_em(answer)
            return any(a == self._norm_em(g) for g in example.answers if g)
        if example.verifier_kind == "numeric":
            a = self._first_number(answer)
            g = self._first_number(example.gold)
            return a is not None and g is not None and abs(a - g) < 1e-5
        # lookup: Alias-Treffer (case-insensitiv, Teilstring in beide Richtungen)
        ans = answer.strip().lower()
        if not ans:
            return False
        for gold in example.answers:
            g = gold.strip().lower()
            if g and (g in ans or ans in g):
                return True
        return False
