import re
import contextlib
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Tuple, Optional

_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


class ECLPolicy:
    """Meta-Policy über epistemische Aktionen (Policy = Base + LoRA-Adapter).

    Wichtig (v0.4): Der Gradient fließt über die *Aktionswahl* (die epistemischen
    Entscheidungen der Meta-Policy), nicht über die generierten Antwort-Tokens.
    Die Antwortgenerierung ist Teil der Umgebung/des Rewards und läuft unter no_grad.

    π_ref ist KEIN separates Modell, sondern dieselbe Base mit *deaktiviertem* Adapter
    (siehe as_reference()). Das halbiert den Speicher und liefert KL(πθ‖π_ref)=0 bei Init.
    """

    def __init__(self, config, device, adapter_path: Optional[str] = None):
        self.config = config
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(config.base_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Left-Padding: dann ist das letzte Token jeder Zeile immer an Position -1,
        # sodass logits[:, -1, :] für jede Batch-Zeile die Next-Token-Verteilung liefert.
        self.tokenizer.padding_side = "left"

        base = AutoModelForCausalLM.from_pretrained(
            config.base_model_name,
            dtype=_DTYPE_MAP.get(config.dtype, torch.bfloat16),
        ).to(device)

        self._is_peft = bool(getattr(config, "use_lora", True))
        if self._is_peft:
            from peft import LoraConfig, get_peft_model, PeftModel
            if adapter_path:
                # Trainierten Adapter laden (z.B. Arm C im A/B/C-Eval)
                self.model = PeftModel.from_pretrained(base, adapter_path)
            else:
                lora_cfg = LoraConfig(
                    r=config.lora_r,
                    lora_alpha=config.lora_alpha,
                    lora_dropout=config.lora_dropout,
                    target_modules=list(config.lora_target_modules),
                    task_type="CAUSAL_LM",
                )
                self.model = get_peft_model(base, lora_cfg)  # Base eingefroren, nur Adapter trainierbar
        else:
            self.model = base  # Fallback (kein Ref-Unterschied; nur Debug)

        # Kennt das Modell überhaupt einen Thinking-Modus? Nur dann darf im Rohmodus ein leerer
        # Think-Block vorangestellt werden (bei Llama/Mistral wären das sinnlose Fremdtokens).
        # NICHT automatisch aus dem Template ableiten: Qwen3-4B/8B kennen enable_thinking,
        # emittieren im Rohmodus aber gar keine <think>-Blöcke. Hängt man ihnen trotzdem einen
        # leeren an, antwortet das 4B überhaupt nicht mehr (2814 leere Antworten von 5985) und
        # das 8B kippt in den Assistenzmodus. Nur Qwen3.6 braucht die Unterdrückung — deshalb
        # explizit pro Lauf setzen (extract_features.py --suppress-think).
        _tpl = getattr(self.tokenizer, "chat_template", None) or ""
        self.has_think_mode = "enable_thinking" in _tpl
        self._suppress_think = False

        # Aktionen über Vokabular-Tokens (erstes Subtoken). Policy und Ref teilen dieselben IDs.
        self.action_names: List[str] = list(config.actions)
        self.action_ids = torch.tensor(
            [self.tokenizer.encode(a, add_special_tokens=False)[0] for a in self.action_names],
            device=device,
        )

    # ------------------------------------------------------------------
    # π_ref = Base ohne Adapter
    # ------------------------------------------------------------------
    @contextlib.contextmanager
    def as_reference(self):
        """Innerhalb dieses Kontextes ist der LoRA-Adapter deaktiviert -> Forward = π_ref."""
        if self._is_peft:
            with self.model.disable_adapter():
                yield
        else:
            yield

    # ------------------------------------------------------------------
    # Prompt-Aufbau (chat-template-fähig für Instruct-Modelle)
    # ------------------------------------------------------------------
    def _build_prompt(self, state_str: str, kind: str, fewshot: str = "",
                      qa_style: str = "long", raw: bool = False) -> str:
        """kind='action' -> Aufforderung, EINE epistemische Aktion zu wählen.
        kind='answer'   -> Aufforderung, die Frage final zu beantworten.
        Nutzt das Chat-Template, falls vorhanden (Qwen3-Instruct); sonst Roh-Fallback."""
        if kind == "action":
            instr = (state_str + "\n\nWähle GENAU EINE epistemische Aktion und antworte nur mit "
                     "dem Wort: " + ", ".join(self.action_names) + ".")
            suffix = "\nAktion:"
        elif kind == "reason":
            instr = (state_str + "\n\nDenke kurz Schritt für Schritt und schließe mit "
                     "'Antwort: <Ergebnis>'.")
            suffix = "\nLösung:"
        else:  # answer
            # STANDARD-Format der Benchmarks, bewusst OHNE Instruktion (Run 27):
            #   TriviaQA/WebQuestions (lm-evaluation-harness): "Question: {q}\nAnswer:"
            #   PopQA (Mallen et al. 2212.10511): "Q: {q} A:" — dort steht explizit, dass
            #   aufwendigere Instruktionen nichts brachten.
            # Englisch statt vorher deutsch (Confound #5): die Gold-Antworten sind englisch,
            # bei deutscher Instruktion antworteten stärkere Modelle deutsch und der Verifier
            # zählte das falsch. P(True)/P(IK) bleiben deutsch — sonst änderte sich das
            # gemessene Signal selbst und Run 23–26 wären nicht mehr anschlussfähig.
            # Die Antwortlänge wird wie in der Literatur über FEW-SHOT-Exemplare gebunden
            # (siehe extract_features.py --shots), nicht über eine selbstgebaute Anweisung:
            # ein handgeschriebenes Exemplar zieht sonst den Antworttyp (z.B. Richtung
            # Personennamen) und wäre genau der Bias, den wir vermeiden wollen.
            # Strukturierte Zustände (ECLState.to_string(): "Frage:/Kontext:/Evidenz:") NICHT
            # in das Q/A-Schema pressen — nur die nackte Frage des direkten Pfads.
            # Jeder Benchmark hat sein eigenes Template — PopQA "Q:/A:" (Mallen et al.),
            # TriviaQA/WebQuestions "Question:/Answer:" (lm-evaluation-harness).
            q_tag, a_tag = ("Q:", "A:") if qa_style == "short" else ("Question:", "Answer:")
            body = state_str if "\n" in state_str else f"{q_tag} {state_str}"
            instr = f"{fewshot}{body}\n{a_tag}"
            # raw=True: KEIN Chat-Template, reine Fortsetzung wie im Harness. Chat-Wrapper
            # erzeugen sonst Markdown, "Answer:"-Imitate und Satzantworten — Artefakte, die
            # es im Standard-Setup nicht gibt.
            if raw:
                # Thinking unterdrücken (Confound #3, zweite Auflage): der Rohmodus umgeht das
                # Chat-Template und damit enable_thinking=False. Qwen3.6 schreibt dann spontan
                # <think>-Blöcke, die das Token-Budget auffressen (92 % der popqa-Antworten
                # ohne Ergebnis, results.md Run 27). enable_thinking=False ist im Template
                # nichts anderes als ein vorangestellter leerer Think-Block — den hängen wir hier
                # von Hand an, sofern das Modell dieses Konzept überhaupt kennt.
                if self._suppress_think:
                    return instr + "<think>\n\n</think>\n\n"
                return instr
            suffix = ""
        if getattr(self.tokenizer, "chat_template", None):
            return self._apply_chat(instr)
        # Fallback ohne Chat-Template (z.B. tiny Test-Modelle)
        return instr + suffix

    # ------------------------------------------------------------------
    # Aktionsverteilung
    # ------------------------------------------------------------------
    def action_log_probs_batch(self, states: List[str], with_grad: bool = True) -> torch.Tensor:
        """Log-Wahrscheinlichkeiten über den Aktionsraum für einen Batch von Zuständen.

        Ein einziger (batched) Forward für die ganze Gruppe. Gibt (B, num_actions) zurück;
        mit Graph, wenn with_grad=True. Dank Left-Padding ist Position -1 für jede Zeile
        das jeweils letzte echte Token.
        """
        prompts = [self._build_prompt(s, "action") for s in states]
        enc = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.device)
        ctx = torch.enable_grad() if with_grad else torch.no_grad()
        with ctx:
            logits = self.model(**enc).logits[:, -1, :]            # (B, vocab)
            action_logits = logits[:, self.action_ids]             # (B, num_actions)
            log_probs = F.log_softmax(action_logits / self.config.action_temperature, dim=-1)
        return log_probs

    def action_log_probs(self, state: str, with_grad: bool = True) -> torch.Tensor:
        """Single-State-Variante (für Eval). Form (num_actions,)."""
        return self.action_log_probs_batch([state], with_grad=with_grad)[0]

    def sample_action(self, state: str) -> Tuple[str, torch.Tensor, torch.Tensor]:
        """Sampelt eine Aktion. Liefert (name, log_prob_der_aktion[grad], log_probs_alle[grad])."""
        log_probs = self.action_log_probs(state, with_grad=True)
        idx = torch.multinomial(log_probs.detach().exp(), 1).item()
        return self.action_names[idx], log_probs[idx], log_probs

    # ------------------------------------------------------------------
    # Binäre Logit-Probe (Kadavath P(True)/P(IK)) — saubere Ja/Nein-Wahrscheinlichkeit
    # ------------------------------------------------------------------
    def _word_first_ids(self, words: List[str]) -> List[int]:
        ids = set()
        for w in words:
            for variant in (w, " " + w):
                toks = self.tokenizer.encode(variant, add_special_tokens=False)
                if toks:
                    ids.add(toks[0])
        return sorted(ids)

    def yesno_prob_batch(self, contents: List[str], yes_words: List[str],
                         no_words: List[str], raw: bool = False) -> List[float]:
        """P(Ja) = Masse(yes-Tokens) / (Masse(yes)+Masse(no)) am Entscheidungs-Token.

        `contents` = die User-Nachricht (inkl. der Ja/Nein-Frage). Liest die Logits am
        letzten Token (Left-Padding) und normalisiert über die Ja/Nein-Kandidaten — robust
        gegen Tokenizer-Eigenheiten. Im Gegensatz zur 3-Wege-Wort-Policy ein sauberes,
        kalibrierbares Skalar (keine answer_now-Degeneration)."""
        yes_ids = self._word_first_ids(yes_words)
        no_ids = self._word_first_ids(no_words)
        enc = self.tokenizer(self._yesno_prompts(contents, raw=raw),
                             return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            logits = self.model(**enc).logits[:, -1, :]
            probs = torch.softmax(logits.float(), dim=-1)
            py = probs[:, yes_ids].sum(-1)
            pn = probs[:, no_ids].sum(-1)
            out = (py / (py + pn + 1e-8))
        return out.tolist()

    def _apply_chat(self, content: str) -> str:
        """Chat-Template mit add_generation_prompt; Thinking-Mode AUS (sonst verbraten Basis-Qwen3
        die kurzen Antwort-Tokens im <think>-Block). Fällt zurück, falls das Template das kwarg nicht kennt."""
        msgs = [{"role": "user", "content": content}]
        try:
            return self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                                      enable_thinking=False)
        except TypeError:
            return self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    def _yesno_prompts(self, contents: List[str], raw: bool = False) -> List[str]:
        """raw=True erzwingt die Roh-Fortsetzung auch bei Modellen MIT Chat-Template.
        Nur für den Prompt-Variations-Test (ptrue_variants.py) gedacht: dort ist die
        Template-Wahl eine der Stellschrauben, mit denen P(True) gezielt geschwächt/
        gestärkt wird. Default bleibt unverändert."""
        if self.tokenizer.chat_template and not raw:
            return [self._apply_chat(c) for c in contents]
        return [c + "\nAntwort:" for c in contents]

    def last_hidden_batch(self, contents: List[str]):
        """Letzter-Token-Hidden-State (letzte Schicht) für dieselben Prompts wie yesno_prob_batch.
        Für die Hidden-State-Probe (Phase 3b): repräsentiert den internen Zustand an der
        Entscheidungs-Position. Gibt ein (B, hidden_dim) float32-numpy-Array zurück."""
        enc = self.tokenizer(self._yesno_prompts(contents), return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            h = self.model(**enc, output_hidden_states=True).hidden_states[-1][:, -1, :]
        return h.float().cpu().numpy()

    def answer_hidden_batch(self, questions: List[str], answers: List[str],
                            fracs=(0.25, 0.5, 0.75, 1.0), pool: str = "last",
                            fewshot: Optional[List[str]] = None,
                            styles: Optional[List[str]] = None, raw: bool = False):
        """Hidden States an der ANTWORT-Position — der faire Ort für eine Korrektheits-Probe.

        Unterschied zu `last_hidden_batch` (Confound #4, results.md Run 24): dort saß die Probe
        am letzten Token des Ja/Nein-Prompts, also unmittelbar vor dem Unembedding, das die
        Ja/Nein-Logits erzeugt -> sie rekonstruierte zwangsläufig P(True) (Spearman 0.97–0.99).
        Hier ist ÜBERHAUPT KEIN Selbstverifikations-Prompt im Kontext: nur Frage + generierte
        Antwort, wie in der Literatur (SAPLMA, Orgad et al.) üblich.

        fracs: relative Schichttiefen (1.0 = letzte Schicht) — mehrere Schichten fallen im selben
        Forward-Pass ab, das ist der Layer-Sweep gratis.
        pool:  "last" = letztes Antwort-Token · "mean" = Mittel über die Antwort-Tokens.

        Rückgabe: float32-Array (B, len(fracs), hidden_dim) sowie die Liste der Schichtindizes.
        """
        # Dieselben Few-Shot-Exemplare wie bei der Generierung: die Probe muss auf GENAU dem
        # Kontext sondieren, in dem die Antwort entstanden ist — sonst misst sie einen Zustand,
        # den es beim Antworten nie gab.
        fs = fewshot or [""] * len(questions)
        st = styles or ["long"] * len(questions)
        prompts = [self._build_prompt(q, "answer", f, qa, raw=raw) + a
                   for q, a, f, qa in zip(questions, answers, fs, st)]
        enc = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            hs = self.model(**enc, output_hidden_states=True).hidden_states
        n_layers = len(hs) - 1  # hs[0] = Embedding-Ausgabe
        idxs = sorted({max(1, min(n_layers, int(round(f * n_layers)))) for f in fracs})

        if pool == "mean":
            # Antwort-Tokens = die letzten k Positionen (Left-Padding); k aus der Antwort allein.
            klens = [max(1, len(self.tokenizer(a, add_special_tokens=False)["input_ids"]))
                     for a in answers]
            out = []
            for li in idxs:
                h = hs[li]
                rows = [h[b, -k:, :].mean(0) for b, k in enumerate(klens)]
                out.append(torch.stack(rows))
            H = torch.stack(out, dim=1)
        else:
            H = torch.stack([hs[li][:, -1, :] for li in idxs], dim=1)
        return H.float().cpu().numpy(), idxs

    # ------------------------------------------------------------------
    # Antwort + Confidence
    # ------------------------------------------------------------------
    def commit_answer_batch(self, states: List[str], deterministic: bool = False,
                            kind: str = "answer", max_new_tokens: Optional[int] = None,
                            return_tokens: bool = False,
                            fewshot: Optional[List[str]] = None,
                            styles: Optional[List[str]] = None,
                            raw: bool = False) -> List[Tuple]:
        """Generiert Antworten + Confidence für einen Batch von Zuständen in einem generate()-Call.

        Confidence-Proxy (config.confidence_method == "token_prob"):
            exp(mittlerer Token-Log-Prob der generierten Antwort) ∈ (0, 1].
        Echtes modellabgeleitetes Signal (kein hartkodierter Wert) -> Brier/AUROC messen
        reale Diskrimination.
        """
        fs = fewshot or [""] * len(states)
        st = styles or ["long"] * len(states)
        prompts = [self._build_prompt(s, kind, f, qa, raw=raw)
                   for s, f, qa in zip(states, fs, st)]
        enc = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.device)
        input_len = enc["input_ids"].shape[1]  # dank Left-Padding für alle Zeilen gleich

        mnt = max_new_tokens if max_new_tokens is not None else getattr(self.config, "max_new_tokens", 32)
        gen_kwargs = dict(
            max_new_tokens=mnt,
            pad_token_id=self.tokenizer.pad_token_id,
            return_dict_in_generate=True,
            output_scores=True,
        )
        if deterministic:
            gen_kwargs["do_sample"] = False
        else:
            gen_kwargs.update(do_sample=True, temperature=0.7, top_p=0.95)

        with torch.no_grad():
            out = self.model.generate(**enc, **gen_kwargs)

        seqs = out.sequences[:, input_len:]                       # (B, gen_len)
        scores = torch.stack(out.scores, dim=1)                   # (B, gen_len, vocab)

        results: List[Tuple] = []
        for b in range(len(states)):
            gen_ids = seqs[b]
            # Tokens nach dem ersten EOS/Pad ignorieren (kürzere Antworten im Batch)
            keep = (gen_ids != self.tokenizer.pad_token_id)
            n = int(keep.sum().item()) if keep.any() else gen_ids.shape[0]
            text = self.tokenizer.decode(gen_ids[:n], skip_special_tokens=True)
            answer = self._extract_answer(text)
            confidence = self._token_prob_confidence(scores[b, :n], gen_ids[:n])
            if self.config.confidence_method == "verbalized":
                verbal = self._parse_verbalized_confidence(text)
                if verbal is not None:
                    confidence = verbal
            results.append((answer, confidence, n) if return_tokens else (answer, confidence))
        return results

    def commit_answer(self, state: str, deterministic: bool = False,
                      kind: str = "answer", max_new_tokens: Optional[int] = None) -> Tuple[str, float]:
        """Single-State-Variante (für Eval/Referenz)."""
        return self.commit_answer_batch([state], deterministic=deterministic,
                                        kind=kind, max_new_tokens=max_new_tokens)[0]

    def reason_batch(self, states: List[str], deterministic: bool = True,
                     max_new_tokens: Optional[int] = None) -> List[Tuple[str, int]]:
        """reason_more: erzeugt einen CoT-Schritt pro Zustand. Gibt (roher_reasoning_text, n_tokens)
        zurück (kein Extrahieren/keine Confidence) — der Text wird in den State geschrieben."""
        prompts = [self._build_prompt(s, "reason") for s in states]
        enc = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.device)
        input_len = enc["input_ids"].shape[1]
        mnt = max_new_tokens if max_new_tokens is not None else getattr(self.config, "reason_max_tokens", 128)
        gen_kwargs = dict(max_new_tokens=mnt, pad_token_id=self.tokenizer.pad_token_id)
        if deterministic:
            gen_kwargs["do_sample"] = False
        else:
            gen_kwargs.update(do_sample=True, temperature=0.7, top_p=0.95)
        with torch.no_grad():
            seqs = self.model.generate(**enc, **gen_kwargs)[:, input_len:]
        out: List[Tuple[str, int]] = []
        for b in range(len(states)):
            gen_ids = seqs[b]
            keep = (gen_ids != self.tokenizer.pad_token_id)
            n = int(keep.sum().item()) if keep.any() else gen_ids.shape[0]
            out.append((self.tokenizer.decode(gen_ids[:n], skip_special_tokens=True).strip(), n))
        return out

    @staticmethod
    def _extract_answer(text: str) -> str:
        """Roh-Antwort (Whitespace normalisiert). Die kind-spezifische Extraktion macht der
        ExampleVerifier: Zahl bei numeric (GSM8K), Substring/Alias bei lookup (PopQA).
        Vorher wurde hier vorschnell die erste Zahl gezogen -> brach QA-Antworten."""
        return " ".join(text.strip().split())

    def _token_prob_confidence(self, scores: torch.Tensor, gen_ids: torch.Tensor) -> float:
        """scores: (gen_len, vocab) Logits der generierten Schritte; gen_ids: (gen_len,)."""
        if scores.shape[0] == 0 or gen_ids.shape[0] == 0:
            return 0.5
        log_probs = F.log_softmax(scores.float(), dim=-1)
        n = min(gen_ids.shape[0], log_probs.shape[0])
        if n == 0:
            return 0.5
        tok_lp = log_probs[torch.arange(n, device=log_probs.device), gen_ids[:n]]
        return float(tok_lp.mean().exp().clamp(0.0, 1.0).item())

    @staticmethod
    def _parse_verbalized_confidence(text: str) -> Optional[float]:
        m = re.search(r"(?:confidence|konfidenz|sicher)\D*(\d+(?:\.\d+)?)", text, re.IGNORECASE)
        if not m:
            return None
        val = float(m.group(1))
        if val > 1.0:  # vermutlich Prozent
            val /= 100.0
        return max(0.0, min(1.0, val))
