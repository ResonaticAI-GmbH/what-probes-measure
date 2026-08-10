from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import torch

@dataclass
class ECLConfig:
    # Modell-Setup
    base_model_name: str = "Qwen/Qwen3-4B-Instruct-2507"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: str = "bfloat16"  # bf16 ist auf Blackwell/GB10 stabiler als fp16 fürs Training

    # LoRA (PEFT): Policy = Base + Adapter, π_ref = Base via disable_adapter() -> KL startet bei 0.
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0  # 0 -> deterministischer Adapter; wichtig für saubere KL/Ref
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])

    # Epistemische Aktionen (v0.4)
    actions: List[str] = field(default_factory=lambda: ["answer_now", "get_evidence", "abstain"])

    # Training
    batch_size: int = 4
    group_size: int = 4  # G
    beta_kl: float = 0.1
    max_steps: int = 1000
    learning_rate: float = 1e-6
    clip_advantage: float = 10.0      # A_MAX fürs Advantage-Clipping
    min_group_std: float = 0.02       # Degenerate-Group-Guard: std<this -> Advantage=0 (kein Signal)
    # Exploration: Qwens Aktions-Logits sind extrem spitz (answer_now≈1.0). T=1.0/coef=0.01
    # kollabiert sofort (tool/abstain werden nie gesampelt). Auf GSM8K validiert: T=3.0, coef=0.1
    # -> Policy exploriert, lernt get_evidence (acc 0->1.0 in ~6 Steps). Eval ist greedy (T-invariant).
    action_temperature: float = 3.0    # Policy-/Sampling-Temperatur über den Aktionsraum
    entropy_coef: float = 0.1          # λ_H: Entropie-Bonus gegen answer_now-Kollaps
    max_episode_steps: int = 3        # Cap gegen Endlos-get_evidence; danach Zwang zu answer_now

    # Utility
    utility_abstain: float = 0.0  # U_ABSTAIN
    tool_cost: float = 0.1
    step_cost: float = 0.01
    reason_cost: float = 0.05     # Kosten eines reason_more-Schritts (Value of Computation)
    reason_max_tokens: int = 128  # Token-Budget pro reason_more-Schritt (CoT)
    max_new_tokens: int = 32      # Antwort-Generierungslänge (höher -> Base kann mehr direkt lösen)

    # Daten
    data_source: str = "synthetic"  # "synthetic" | "gsm8k" | "popqa" | "gsm8k+popqa"
    data_limit: int = 1000          # max Beispiele pro reale Quelle

    # Verifier (Legacy-Feld; reale Verifikation läuft example-getrieben über verifier_kind)
    verifier_type: str = "numeric"  # "numeric", "exact_match", "unit_test", "lookup"

    # Confidence
    # "token_prob": Sequenzwahrscheinlichkeit der Antwort als Proxy (robust, modellintern)
    # "verbalized": parst eine vom Modell genannte Zahl (Spec-v0, aber bei kleinen Modellen unzuverlässig)
    confidence_method: str = "token_prob"
    
    # Logging
    log_dir: str = "./runs/ecl_v04"
    eval_interval: int = 100
    
    # v0.4 spezifisch
    use_state_dependent_baseline: bool = True
    temperature_ref: float = 0.0  # Deterministisch für π_ref

    # Ablationen (§14)
    # "v04_direct":        Â=(R_i-U_ref)/std(R)        — Counterfactual-Anker, G-unabhängig
    # "v02_centered":      Â=(R_i-mean(R))/std(R)       — U_ref fällt algebraisch raus (No-op)
    # "v03_ref_in_group":  τ_ref als (G+1)-tes Mitglied — Anker um 1/(G+1) verdünnt
    advantage_variant: str = "v04_direct"
    use_brier: bool = True   # False -> Utility = reine Korrektheit (ohne Kalibrierung)
    seed: int = 0
