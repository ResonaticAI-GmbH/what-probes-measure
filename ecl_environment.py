"""Zustandsrepräsentation für den ECL-Agenten.

Verifier, Tools und Aufgaben-Quellen leben jetzt in `ecl_data.py`
(example-getrieben). Hier bleibt nur der Episodenzustand.
"""


class ECLState:
    """Zustand einer ECL-Episode: Frage + akkumulierte Evidenz."""
    def __init__(self, question: str, context: str = ""):
        self.question = question
        self.context = context
        self.evidence = []

    def update(self, evidence: str):
        self.evidence.append(evidence)
        self.context += "\n" + evidence

    def to_string(self) -> str:
        full_context = self.context if self.context else self.question
        evid = "; ".join(self.evidence) if self.evidence else "Keine"
        return f"Frage: {self.question}\nKontext: {full_context}\nEvidenz: {evid}"
