import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import accuracy_score, f1_score

# ------------------------------------------------------------------
# Page Configuration & Warm, Welcoming Light Theme Styling
# ------------------------------------------------------------------
st.set_page_config(
    page_title="CryptoShield • Protocol Risk & Lean 4 Workbench",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    /* Global Light & Welcoming Palette */
    .stApp {
        background-color: #FAF9F6;
        color: #1E293B;
    }
    section[data-testid="stSidebar"] {
        background-color: #F3F1EC;
        border-right: 1px solid #E2DFD7;
    }
    /* Hero Banner */
    .hero-card {
        background: linear-gradient(135deg, #FFFFFF 0%, #F0FDFA 60%, #FEF3C7 100%);
        border: 1px solid #CCFBF1;
        border-radius: 18px;
        padding: 28px 32px;
        margin-bottom: 22px;
        box-shadow: 0 4px 20px rgba(15, 118, 110, 0.06);
    }
    .hero-badge {
        display: inline-block;
        background-color: #CCFBF1;
        color: #0F766E;
        font-weight: 700;
        font-size: 0.78rem;
        padding: 4px 12px;
        border-radius: 999px;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        margin-bottom: 10px;
    }
    .hero-title {
        font-size: 2.05rem;
        font-weight: 800;
        color: #0F172A;
        margin: 0 0 8px 0;
        line-height: 1.2;
    }
    .hero-subtitle {
        font-size: 1.02rem;
        color: #475569;
        margin: 0;
        line-height: 1.55;
    }
    /* Metric Cards */
    .kpi-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 18px 20px;
        box-shadow: 0 2px 10px rgba(15, 23, 42, 0.03);
        transition: transform 0.15s ease;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        border-color: #99F6E4;
    }
    .kpi-label {
        font-size: 0.82rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.03em;
    }
    .kpi-value {
        font-size: 1.85rem;
        font-weight: 800;
        color: #0F766E;
        margin: 4px 0;
    }
    .kpi-sub {
        font-size: 0.82rem;
        color: #64748B;
    }
    /* Severity Badges */
    .badge-high {
        background-color: #FEE2E2;
        color: #B91C1C;
        padding: 4px 11px;
        border-radius: 999px;
        font-weight: 700;
        font-size: 0.82rem;
    }
    .badge-mod {
        background-color: #FEF3C7;
        color: #B45309;
        padding: 4px 11px;
        border-radius: 999px;
        font-weight: 700;
        font-size: 0.82rem;
    }
    .badge-low {
        background-color: #DCFCE7;
        color: #15803D;
        padding: 4px 11px;
        border-radius: 999px;
        font-weight: 700;
        font-size: 0.82rem;
    }
    /* Info Callout */
    .info-box {
        background-color: #FFFFFF;
        border-left: 4px solid #0F766E;
        border-top: 1px solid #E2E8F0;
        border-right: 1px solid #E2E8F0;
        border-bottom: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 16px 20px;
        margin: 12px 0;
    }
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #F0EFEA;
        padding: 6px;
        border-radius: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 18px;
        font-weight: 600;
        color: #334155;
    }
    .stTabs [aria-selected="true"] {
        background-color: #FFFFFF !important;
        color: #0F766E !important;
        box-shadow: 0 2px 6px rgba(0,0,0,0.06);
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# Load Data & Train Live ML Models (Cached for Instant Interactivity)
# ------------------------------------------------------------------
ATTACK_LABELS = [
    'known_mitm_history', 'known_replay_history',
    'known_downgrade_history', 'known_token_theft_risk'
]
ATTACK_NAMES = ['MITM', 'Replay', 'Downgrade', 'Token-theft']

DESIGN_FEATURES = [
    'uses_public_key', 'uses_symmetric_key', 'uses_nonce', 'uses_timestamp',
    'uses_certificates', 'has_mutual_auth', 'has_server_auth', 'has_client_auth',
    'key_exchange', 'forward_secrecy', 'integrity_protection',
    'explicit_replay_protection', 'trusted_third_party', 'identity_binding',
    'modern_aead'
]

FEATURE_DESCRIPTIONS = {
    'uses_public_key': ('Asymmetric / Public Key Crypto', 'Uses public/private key pairs (RSA, ECC, DH).'),
    'uses_symmetric_key': ('Symmetric Key Crypto', 'Uses shared secret keys (AES, DES, RC4).'),
    'uses_nonce': ('Challenge Nonces', 'Uses random one-time numbers for freshness.'),
    'uses_timestamp': ('Timestamps', 'Uses synchronized clocks to bound message lifetime.'),
    'uses_certificates': ('Digital Certificates (PKI)', 'Binds public keys to identities via CA signatures.'),
    'has_mutual_auth': ('Mutual Authentication', 'Both client and server verify each other.'),
    'has_server_auth': ('Server Authentication', 'Client verifies server identity.'),
    'has_client_auth': ('Client Authentication', 'Server verifies client identity.'),
    'key_exchange': ('Session Key Establishment', 'Negotiates fresh session keys dynamically.'),
    'forward_secrecy': ('Perfect Forward Secrecy (PFS)', 'Past sessions stay safe if long-term keys leak.'),
    'integrity_protection': ('Handshake Integrity / MAC', 'Protects handshake messages against tampering.'),
    'explicit_replay_protection': ('Explicit Replay Defense', 'Sequence numbers, nonces, or state tracking.'),
    'trusted_third_party': ('Trusted Third Party / KDC', 'Relies on central server (e.g., Kerberos, Woo-Lam).'),
    'identity_binding': ('Explicit Identity Binding', 'Seals peer identities inside encrypted payloads (NS-Lowe fix).'),
    'modern_aead': ('Modern AEAD Cipher', 'Authenticated Encryption with Associated Data (AES-GCM, ChaCha20-Poly1305).'),
}

@st.cache_data
def load_datasets():
    df_raw = pd.read_csv('final_dv1.csv')
    df_rank = pd.read_csv('protocol_ranking.csv')
    merged = pd.merge(
        df_rank,
        df_raw.drop(columns=['family', 'initial_risk_label']),
        on='protocol',
        how='left'
    )
    return df_raw, df_rank, merged

@st.cache_resource
def train_models(df_raw):
    X_all = df_raw.drop(columns=['initial_risk_label', 'protocol'])
    X_design = X_all.drop(columns=ATTACK_LABELS)
    y = df_raw['initial_risk_label']

    Y_att = df_raw[ATTACK_LABELS].astype(int).copy()
    Y_att.columns = ATTACK_NAMES
    X_att = df_raw.drop(columns=['protocol', 'initial_risk_label'] + ATTACK_LABELS)

    def make_pre(X):
        return ColumnTransformer([
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['family']),
            ('num', StandardScaler(), [c for c in X.columns if c != 'family']),
        ])

    # Selected severity model (ALL features)
    clf_all = Pipeline([
        ('prep', make_pre(X_all)),
        ('model', LogisticRegression(max_iter=2000, C=1.0, random_state=42))
    ])
    clf_all.fit(X_all, y)

    # Design-only severity model (no historical leak)
    clf_design = Pipeline([
        ('prep', make_pre(X_design)),
        ('model', LogisticRegression(max_iter=2000, C=1.0, random_state=42))
    ])
    clf_design.fit(X_design, y)

    # Random Forest for feature importance
    rf_model = Pipeline([
        ('prep', make_pre(X_all)),
        ('model', RandomForestClassifier(n_estimators=300, max_depth=6, random_state=42))
    ])
    rf_model.fit(X_all, y)

    # Multi-label attack predictor (Design features -> 4 attack vectors)
    att_clf = Pipeline([
        ('prep', make_pre(X_att)),
        ('model', MultiOutputClassifier(LogisticRegression(max_iter=2000, C=1.0, random_state=42)))
    ])
    att_clf.fit(X_att, Y_att)

    # Compute LOO probabilities for Sensitivity view
    loo = LeaveOneOut()
    all_proba = cross_val_predict(clf_all, X_all, y, cv=loo, method='predict_proba')
    des_proba = cross_val_predict(clf_design, X_design, y, cv=loo, method='predict_proba')
    classes = list(clf_all.named_steps['model'].classes_)
    high_idx = classes.index('High')

    sens_df = pd.DataFrame({
        'protocol': df_raw['protocol'],
        'family': df_raw['family'],
        'actual': y,
        'P_High_ALL': np.round(all_proba[:, high_idx], 3),
        'P_High_DESIGN': np.round(des_proba[:, high_idx], 3),
    })
    sens_df['delta'] = np.round(sens_df['P_High_ALL'] - sens_df['P_High_DESIGN'], 3)

    # Feature importances
    feat_names = rf_model.named_steps['prep'].get_feature_names_out()
    clean_names = [n.replace('num__', '').replace('cat__family_', 'Family: ') for n in feat_names]
    imp_df = pd.DataFrame({
        'Feature': clean_names,
        'Importance': rf_model.named_steps['model'].feature_importances_
    }).sort_values('Importance', ascending=False)

    return clf_all, clf_design, att_clf, sens_df, imp_df

df_raw, df_rank, df_merged = load_datasets()
clf_all, clf_design, att_clf, sens_df, imp_df = train_models(df_raw)

# ------------------------------------------------------------------
# Lean 4 Formal Proof Library (From ProtocolsDolevYao.lean, ProtocolsPaulson.lean & ChallengeResponseSafety.lean)
# ------------------------------------------------------------------
LEAN_PROOFS = {
    "SSL_2_0": {
        "title": "SSL 2.0 — Cipher-Suite Rollback Attack",
        "file": "ProtocolsDolevYao.lean (Part IX)",
        "style": "Shallow Dolev-Yao • Algebraic Negotiation Model",
        "verdict": "VULNERABLE (Attack Certified)",
        "flaw": "SSL 2.0's closing FINISHED handshake message covers only the master secret—not the offered cipher-suite list. A MITM deletes strong suites from client_offer, forcing fallback to weak 40-bit export ciphers without detection.",
        "theorem": "SSL_2_0.rollback_attack",
        "code": """namespace SSL_2_0
inductive Cipher where | strong | weak deriving DecidableEq

def negotiate : List Cipher → Cipher
  | []     => Cipher.weak
  | c :: _ => c

def client_offer : List Cipher := [Cipher.strong, Cipher.weak]
def attacker_rewrite (cs : List Cipher) : List Cipher := cs.filter isWeak

/-- Cipher-suite rollback attack, certified. -/
theorem rollback_attack :
    negotiate client_offer ≠ negotiate (attacker_rewrite client_offer) := by
  decide
end SSL_2_0"""
    },
    "Bluetooth_Legacy": {
        "title": "Bluetooth Legacy — Passive PIN Brute-Force",
        "file": "ProtocolsDolevYao.lean (Part X)",
        "style": "Shallow Dolev-Yao • Deductive Closure",
        "verdict": "VULNERABLE (Attack Certified)",
        "flaw": "Initial key K_init = E22(IN_RAND, BD_ADDR, PIN) relies on two public wire values and a low-entropy 4-digit PIN (10⁴ candidates). An eavesdropper enumerates PINs offline in under a second.",
        "theorem": "Bluetooth_Legacy.pin_recoverable",
        "code": """namespace Bluetooth_Legacy
theorem pin_recoverable (pin a r lk : Nat) :
    Derivable (pairing_trace pin a r lk) (Message.pin pin) := by
  apply Derivable.brute_pin pin a r lk
  · exact Derivable.seen (List.Mem.head _)
  · exact Derivable.seen (List.Mem.tail _ (List.Mem.head _))
  · exact Derivable.seen (List.Mem.tail _ (List.Mem.tail _ (List.Mem.head _)))
end Bluetooth_Legacy"""
    },
    "WEP": {
        "title": "WEP (802.11) — IV-Reuse Keystream Leakage",
        "file": "ProtocolsDolevYao.lean (Part VIII) & WEPDemo.lean",
        "style": "Shallow Dolev-Yao • Algebraic XOR Proof",
        "verdict": "VULNERABLE (Attack Certified)",
        "flaw": "WEP uses a 24-bit IV, guaranteeing collisions after ~16M packets. When two packets share (K, IV), XORing their ciphertexts cancels the keystream completely: c₁ ⊕ c₂ = p₁ ⊕ p₂.",
        "theorem": "WEP.wep_iv_reuse_leaks_plaintext_xor",
        "code": """namespace WEP
def encrypt (K IV plaintext : Nat) : Nat :=
  plaintext ^^^ keystream K IV

theorem wep_iv_reuse_leaks_plaintext_xor (K IV p1 p2 : Nat) :
    (encrypt K IV p1) ^^^ (encrypt K IV p2) = p1 ^^^ p2 := by
  unfold encrypt
  rw [Nat.xor_comm p2 (keystream K IV)]
  rw [← Nat.xor_assoc (p1 ^^^ keystream K IV) (keystream K IV) p2]
  rw [Nat.xor_assoc p1 (keystream K IV) (keystream K IV)]
  rw [Nat.xor_self, Nat.xor_zero]
end WEP"""
    },
    "WPA_TKIP": {
        "title": "WPA-TKIP — Beck-Tews Chopchop & Packet Forgery",
        "file": "ProtocolsDolevYao.lean (Part XI)",
        "style": "Shallow Dolev-Yao • Two-Stage Deduction",
        "verdict": "VULNERABLE (Attack Certified)",
        "flaw": "MIC-failure reports act as a byte-by-byte chopchop oracle on predictable ARP packets, revealing the per-IV keystream and allowing arbitrary packet injection under that IV.",
        "theorem": "WPA_TKIP.beck_tews_forgery",
        "code": """namespace WPA_TKIP
theorem beck_tews_forgery (iv plaintext malicious : Nat) :
    Derivable (observed iv plaintext) (Message.forged_packet iv malicious) := by
  apply Derivable.forge iv malicious
  apply Derivable.chopchop iv plaintext
  · exact Derivable.seen (List.Mem.head _)
  · exact Derivable.seen (List.Mem.tail _ (List.Mem.head _))
end WPA_TKIP"""
    },
    "WooLam_Original": {
        "title": "Woo-Lam Original — Abadi-Needham Impersonation",
        "file": "ProtocolsPaulson.lean (Part III)",
        "style": "Full Paulson Inductive Trace Verification (Reachable)",
        "verdict": "VULNERABLE (Trace Attack Certified)",
        "flaw": "Step 5 server response {N_b}_{K_bs} omits the initiator's identity A. Adversary I impersonates A and answers B's challenge using I's own key K_is; B falsely believes A authenticated.",
        "theorem": "WooLam_Original_Revised.wooLam_authentication_failure",
        "code": """namespace WooLam_Original_Revised
theorem wooLam_authentication_failure :
    ∃ tr, Reachable tr ∧
      B_authenticates tr Agent.A 1 ∧
      ev Agent.A Agent.B (step3 Kas 1) ∉ tr := by
  refine ⟨attack_trace, attack_trace_reachable, ?_, ?_⟩
  · refine ⟨?_, ⟨Kis, ?_⟩⟩
    · simp [attack_trace, ev]
    · simp [attack_trace, ev]
  · intro h; simp [attack_trace, ev] at h
end WooLam_Original_Revised"""
    },
    "PAP": {
        "title": "PAP (RFC 1334) — Cleartext Credential Disclosure",
        "file": "ProtocolsDolevYao.lean (Part XII)",
        "style": "Shallow Dolev-Yao • Zero-Primitive Witness",
        "verdict": "VULNERABLE (Attack Certified)",
        "flaw": "Transmits username and password in cleartext over PPP with zero cryptographic protection.",
        "theorem": "PAP.credentials_disclosed",
        "code": """namespace PAP
theorem credentials_disclosed (u p : Nat) :
    Derivable (transmitted u p) (Message.username u)
    ∧ Derivable (transmitted u p) (Message.password p) := by
  refine ⟨?_, ?_⟩
  · exact Derivable.seen (List.Mem.head _)
  · exact Derivable.seen (List.Mem.tail _ (List.Mem.head _))
end PAP"""
    },
    "NeedhamSchroeder_PK": {
        "title": "Needham-Schroeder Public Key — Lowe's 1995 MITM",
        "file": "ProtocolsDolevYao.lean (Part III) & ProtocolsPaulson.lean (Part IV)",
        "style": "Shallow Dolev-Yao + Full Paulson Inductive Trace",
        "verdict": "VULNERABLE (MITM + Secrecy Break Certified)",
        "flaw": "Message 2 {N_a, N_b}_{PK(A)} omits responder B's identity. When A initiates a session with I, I relays A's nonce to B and tricks A into decrypting B's secret challenge N_b.",
        "theorem": "NS_PK.ns_pk_authentication_failure & NS_PK.intruder_learns_Nb",
        "code": """namespace NS_PK
theorem ns_pk_authentication_failure :
    ∃ tr, Reachable tr ∧
      (∃ s, ev s Agent.B (msg1 Agent.A Agent.B 1) ∈ tr) ∧
      (∃ s, ev s Agent.B (msg3 Agent.B 2) ∈ tr) ∧
      ev Agent.A Agent.B (msg1 Agent.A Agent.B 1) ∉ tr := by ...

theorem intruder_learns_Nb :
    Derivable (Trace.messages lowe_trace) (Message.nonce 2) := by
  apply Derivable.decryptI
  apply Derivable.seen
  simp [Trace.messages, lowe_trace, ev, msg3]
end NS_PK"""
    },
    "HTTP_Basic_Auth": {
        "title": "HTTP Basic Auth (RFC 7617) — Base64 Disclosure",
        "file": "ProtocolsDolevYao.lean (Part XIII)",
        "style": "Shallow Dolev-Yao • Public Encoding Inversion",
        "verdict": "VULNERABLE (Attack Certified)",
        "flaw": "Encodes credentials with Base64—a publicly invertible encoding, not encryption. Any passive observer recovers plaintext credentials immediately.",
        "theorem": "HTTP_Basic_Auth.credentials_recoverable",
        "code": """namespace HTTP_Basic_Auth
theorem credentials_recoverable (u p : Nat) :
    Derivable (observed u p) (Message.credentials u p) := by
  apply Derivable.decode
  exact Derivable.seen (List.Mem.head _)
end HTTP_Basic_Auth"""
    },
    "EAP_MD5": {
        "title": "EAP-MD5 (RFC 3748) — Offline Dictionary Attack",
        "file": "ProtocolsDolevYao.lean (Part XIV)",
        "style": "Shallow Dolev-Yao • Wordlist Enumeration",
        "verdict": "VULNERABLE (Attack Certified)",
        "flaw": "Client sends MD5(id || password || challenge). Because MD5 and challenge are public, an eavesdropper hashes wordlist candidates offline until a match is found.",
        "theorem": "EAP_MD5.dictionary_recovery",
        "code": """namespace EAP_MD5
theorem dictionary_recovery (c p : Nat) (hp : p ∈ dictionary) :
    Derivable (observed c p) (Message.password p) := by
  apply Derivable.dict_attack c p hp
  · exact Derivable.seen (List.Mem.head _)
  · exact Derivable.seen (List.Mem.tail _ (List.Mem.head _))
end EAP_MD5"""
    },
    "DiffieHellman_Basic": {
        "title": "Diffie-Hellman (Unauthenticated) — Classic MITM",
        "file": "ProtocolsDolevYao.lean (Part VI)",
        "style": "Shallow Dolev-Yao • Trace Witness",
        "verdict": "VULNERABLE (MITM Certified)",
        "flaw": "Public values g^a and g^b lack authentication. Adversary I substitutes g^i to both A and B, establishing two separate shared keys while masquerading as each peer.",
        "theorem": "DH_Basic.dh_basic_mitm_attack",
        "code": """namespace DH_Basic
theorem dh_basic_mitm_attack :
    A_accepts (dh_pub 7) ∧ B_accepts (dh_pub 7) ∧
    ev Agent.I Agent.A (dh_pub 7) ∈ mitm_trace ∧
    ev Agent.I Agent.B (dh_pub 7) ∈ mitm_trace := by
  refine ⟨⟨7, rfl⟩, ⟨7, rfl⟩, ?_, ?_⟩
  · simp [mitm_trace, ev]
  · simp [mitm_trace, ev]
end DH_Basic"""
    },
    "NeedhamSchroeder_Symmetric": {
        "title": "Needham-Schroeder Symmetric — Denning-Sacco Replay",
        "file": "ProtocolsDolevYao.lean (Part V) & ProtocolsPaulson.lean (Part V)",
        "style": "Shallow Dolev-Yao + Full Paulson Inductive Trace",
        "verdict": "VULNERABLE (Replay + Secrecy Break Certified)",
        "flaw": "Ticket {K_ab, A}_{K_bs} has no timestamp or freshness nonce. If an old session key K_ab_old leaks, an intruder replays the ticket indefinitely and B accepts it as fresh.",
        "theorem": "NS_Sym.ns_sym_replay_attack & NS_Sym.intruder_learns_Nb",
        "code": """namespace NS_Sym
theorem ns_sym_replay_attack :
    ∃ tr, Reachable tr ∧
      ev Agent.B Agent.A (step4 7 Kab_old) ∈ tr ∧
      (∃ s, ev s Agent.B (step3 Kab_old) ∈ tr ∧ s = Agent.I) := by ...
end NS_Sym"""
    },
    "PPTP_MSCHAPv2": {
        "title": "PPTP MS-CHAPv2 — DES Key-Space Split (16-bit Break)",
        "file": "ProtocolsDolevYao.lean (Part XV)",
        "style": "Shallow Dolev-Yao • Entropy Split Deduction",
        "verdict": "VULNERABLE (Constant-Time Key Recovery Certified)",
        "flaw": "Splits 16-byte NT hash into 7+7+2 byte DES keys. The third key has only 2 unknown bytes (2¹⁶ = 65,536 possibilities), brute-forced in under 1 CPU-second.",
        "theorem": "PPTP_MSCHAPv2.block3_recovery",
        "code": """namespace PPTP_MSCHAPv2
theorem block3_recovery (c : Nat) (h : NT_Hash) :
    Derivable (observed c h) (Message.hash_block 3 h.block3) := by
  apply Derivable.brute_block3 c h.block3
  · exact Derivable.seen (List.Mem.head _)
  · exact Derivable.seen (List.Mem.tail _ (List.Mem.tail _ (List.Mem.tail _ (List.Mem.head _))))
end PPTP_MSCHAPv2"""
    },
    "SSL_3_0": {
        "title": "SSL 3.0 — Version Downgrade & POODLE Oracle",
        "file": "ProtocolsDolevYao.lean (Part VII) & ProtocolsPaulson.lean (Part VII)",
        "style": "Shallow Dolev-Yao + Paulson Padding Oracle Trace",
        "verdict": "VULNERABLE (Downgrade + POODLE Certified)",
        "flaw": "Unauthenticated handshake negotiation allows MITM downgrade to SSL 3.0, exposing CBC padding to iterative oracle queries (POODLE, ~1/256 per byte).",
        "theorem": "SSL3.ssl3_downgrade_attack & SSL3_POODLE.ssl3_poodle_attack_certified",
        "code": """namespace SSL3_POODLE
theorem ssl3_poodle_attack_certified :
    ∃ tr, Reachable tr ∧
      (∃ c, PEvent.query c true ∈ tr) ∧
      PEvent.session_established ∈ tr := by
  refine ⟨poodle_trace, poodle_trace_reachable, ?_, ?_⟩
  · refine ⟨winning_ciphertext, ?_⟩; simp [poodle_trace]
  · simp [poodle_trace]
end SSL3_POODLE"""
    },
    "NeedhamSchroeder_Lowe": {
        "title": "Needham-Schroeder-Lowe — Structural & Trace Safety",
        "file": "ProtocolsDolevYao.lean (Part IV) & ProtocolsPaulson.lean (Part VI)",
        "style": "Safety Contrast Proof (Structural Distinctness)",
        "verdict": "SAFE AGAINST NS-PK MITM (Certified Fix)",
        "flaw": "Lowe's 1995 fix seals responder B's identity inside Message 2: {N_a, N_b, B}_{PK(A)}. Lean proves msg2_old and msg2_lowe have distinct constructors and can never unify.",
        "theorem": "NS_Lowe.ns_lowe_blocks_old_attack & NS_Lowe.ns_lowe_trace_safety",
        "code": """namespace NS_Lowe
theorem msg2_shape_distinct (Na Nb Na' Nb' : Nat) :
    msg2_old Agent.A Na Nb ≠ msg2_lowe Agent.A Agent.B Na' Nb' := by
  intro h
  simp [msg2_old, msg2_lowe] at h

theorem ns_lowe_trace_safety :
    ∀ (tr : Trace) (Na Nb : Nat),
      Reachable tr -> ¬ StrongAcceptance (msg2_old Agent.A Na Nb) := by
  intro tr Na Nb _ h
  exact ns_lowe_blocks_old_attack Na Nb h
end NS_Lowe"""
    },
    "ChallengeResponse_Minimal": {
        "title": "Minimal Challenge-Response — Full Unbounded Inductive Safety",
        "file": "ChallengeResponseSafety.lean",
        "style": "Full Paulson Inductive Proof Over Unbounded Sessions",
        "verdict": "SAFE (Recentness & Replay-Freedom Proved)",
        "flaw": "Proves by structural induction over arbitrarily long Reachable traces under an unrestricted Dolev-Yao network that every accepted nonce was freshly issued and is consumed at most once.",
        "theorem": "accept_requires_issue & no_replay",
        "code": """/-- Recentness: every accepted nonce was issued earlier. -/
theorem accept_requires_issue {t : Trace} (h : Reachable t) :
    ∀ n, Event.accept n ∈ t → Event.issue n ∈ t := by ...

/-- Replay-freedom: no nonce is ever accepted twice. -/
theorem no_replay {t : Trace} (h : Reachable t) : Distinct (accepted t) := by
  induction h with
  | nil => simp [accepted, Distinct]
  | @issue t m ht hfresh ih => simpa [accepted] using ih
  | @accept t m ht hissued hnotacc ih =>
      simp only [accepted]
      refine ⟨?_, ih⟩
      intro hc
      exact hnotacc ((mem_accepted_iff m t).mp hc)"""
    }
}

# ------------------------------------------------------------------
# Sidebar Navigation & Quick Filters
# ------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🛡️ CryptoShield Workbench")
    st.caption("Empirical ML Triage + Lean 4 Formal Verification")
    st.divider()

    st.markdown("**📌 Project Metadata**")
    st.markdown(
        "- **Author:** Raj Guru Suwalka\n"
        "- **Supervisor:** Dr. Edon Kelmendi\n"
        "- **Institution:** Queen Mary University of London\n"
        "- **Verification Engine:** Lean 4 (`v4.32.2`, core-only)\n"
        "- **Dataset:** 100 Cryptographic Protocols"
    )
    st.divider()

    st.markdown("**🧭 Quick Protocol Jump**")
    quick_proto = st.selectbox(
        "Inspect any protocol:",
        options=df_rank['protocol'].tolist(),
        index=0
    )
    q_row = df_merged[df_merged['protocol'] == quick_proto].iloc[0]
    st.metric("Composite Vuln. Score", f"{q_row['vulnerability_score']:.3f}", f"Rank #{int(q_row['rank'])} / 100")
    st.metric("P(High Risk)", f"{q_row['P_High']*100:.1f}%", f"Actual: {q_row['actual_severity']}")
    st.info(f"**Lean 4 Status:** {q_row['lean_status']}")

# ------------------------------------------------------------------
# Hero Header & Top KPI Strip
# ------------------------------------------------------------------
st.markdown("""
<div class="hero-card">
    <div class="hero-badge">MSc Data Science & AI • Formal Methods + Machine Learning</div>
    <div class="hero-title">🛡️ Cryptographic Protocol Risk & Lean 4 Verification Workbench</div>
    <p class="hero-subtitle">
        A hybrid security assessment pipeline bridging <b>empirical machine-learning risk triage</b> across 100 cryptographic protocols
        with <b>machine-checked Dolev-Yao & Paulson inductive proofs</b> in Lean 4. Design custom protocols, test sensitivity to
        historical vulnerabilities, and inspect formal attack and safety theorems interactively.
    </p>
</div>
""", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown("""
    <div class="kpi-card">
        <div class="kpi-label">Protocols Analysed</div>
        <div class="kpi-value">100</div>
        <div class="kpi-sub">9 families • 1976 to 2023</div>
    </div>
    """, unsafe_allow_html=True)
with c2:
    st.markdown("""
    <div class="kpi-card">
        <div class="kpi-label">LOO Classifier Accuracy</div>
        <div class="kpi-value">77.0%</div>
        <div class="kpi-sub">Macro-F1: 0.760 (Logistic Reg.)</div>
    </div>
    """, unsafe_allow_html=True)
with c3:
    st.markdown("""
    <div class="kpi-card">
        <div class="kpi-label">High-Risk Recall (Top 13)</div>
        <div class="kpi-value">13 / 13</div>
        <div class="kpi-sub">100% formalised in Lean 4</div>
    </div>
    """, unsafe_allow_html=True)
with c4:
    st.markdown("""
    <div class="kpi-card">
        <div class="kpi-label">Design-Only Stability</div>
        <div class="kpi-value">ρ = 0.919</div>
        <div class="kpi-sub">Spearman correlation without history</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# ------------------------------------------------------------------
# Interactive Tabs
# ------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "🧪 Interactive Protocol Risk Simulator",
    "📊 100-Protocol Explorer & Rankings",
    "🔬 ML Sensitivity & Feature Drivers",
    "📐 Lean 4 Formal Proof Inspector"
])

# ==================================================================
# TAB 1: Interactive Protocol Risk Simulator
# ==================================================================
with tab1:
    st.subheader("🧪 Interactive Protocol Risk & Attack Vector Simulator")
    st.write(
        "Configure a protocol's cryptographic primitives—or load an existing protocol as a starting template—to see how our trained "
        "models score its severity, predict specific attack vectors (MITM, Replay, Downgrade, Token-theft), and recommend formal hardening."
    )

    # Preset selector
    col_preset, col_mode = st.columns([2, 1])
    with col_preset:
        preset_choice = st.selectbox(
            "Load a protocol template (or start from scratch):",
            ["Custom / Blank Protocol"] + df_raw['protocol'].tolist(),
            index=0
        )
    with col_mode:
        eval_mode = st.radio(
            "Scoring Model:",
            ["Design-Only Model (Pure Primitives)", "Full Model (Includes CVE History)"],
            index=0,
            help="Design-Only evaluates pure architectural security without relying on historical breach flags."
        )

    if preset_choice != "Custom / Blank Protocol":
        base_row = df_raw[df_raw['protocol'] == preset_choice].iloc[0].to_dict()
    else:
        base_row = {
            'family': 'authentication',
            'year_introduced': 2024,
            'uses_public_key': 1, 'uses_symmetric_key': 0, 'uses_nonce': 1,
            'uses_timestamp': 0, 'uses_certificates': 1, 'has_mutual_auth': 1,
            'has_server_auth': 1, 'has_client_auth': 1, 'key_exchange': 1,
            'forward_secrecy': 1, 'integrity_protection': 1,
            'explicit_replay_protection': 1, 'trusted_third_party': 0,
            'identity_binding': 1, 'modern_aead': 1,
            'known_mitm_history': 0, 'known_replay_history': 0,
            'known_downgrade_history': 0, 'known_token_theft_risk': 0
        }

    st.markdown("---")
    left_col, right_col = st.columns([1.35, 1], gap="large")

    with left_col:
        st.markdown("#### 1. Protocol Architecture & Cryptographic Primitives")
        c_fam, c_yr = st.columns(2)
        families = sorted(df_raw['family'].unique().tolist())
        fam_idx = families.index(base_row['family']) if base_row['family'] in families else 0
        sel_family = c_fam.selectbox("Protocol Family", families, index=fam_idx)
        sel_year = c_yr.slider("Year Introduced", 1975, 2026, int(base_row['year_introduced']))

        user_inputs = {'family': sel_family, 'year_introduced': sel_year}

        st.markdown("**Core Cryptographic & Security Controls**")
        f_cols = st.columns(3)
        for idx, feat in enumerate(DESIGN_FEATURES):
            label, tip = FEATURE_DESCRIPTIONS[feat]
            with f_cols[idx % 3]:
                user_inputs[feat] = int(st.checkbox(label, value=bool(base_row[feat]), help=tip, key=f"sim_{feat}"))

        if eval_mode == "Full Model (Includes CVE History)":
            st.markdown("**Known Historical Vulnerability Flags**")
            h_cols = st.columns(4)
            hist_labels = [
                ('known_mitm_history', 'MITM History'),
                ('known_replay_history', 'Replay History'),
                ('known_downgrade_history', 'Downgrade History'),
                ('known_token_theft_risk', 'Token-Theft Risk')
            ]
            for i, (h_col, h_name) in enumerate(hist_labels):
                user_inputs[h_col] = int(h_cols[i].checkbox(h_name, value=bool(base_row[h_col]), key=f"sim_{h_col}"))
        else:
            for h_col in ATTACK_LABELS:
                user_inputs[h_col] = 0

    with right_col:
        st.markdown("#### 2. Live Risk Assessment & Attack Predictions")
        input_df = pd.DataFrame([user_inputs])

        if eval_mode == "Full Model (Includes CVE History)":
            probs = clf_all.predict_proba(input_df)[0]
            classes = list(clf_all.named_steps['model'].classes_)
        else:
            probs = clf_design.predict_proba(input_df.drop(columns=ATTACK_LABELS))[0]
            classes = list(clf_design.named_steps['model'].classes_)

        prob_map = dict(zip(classes, probs))
        p_high = prob_map.get('High', 0.0)
        p_mod = prob_map.get('Moderate', 0.0)
        p_low = prob_map.get('Low', 0.0)

        # Attack vector probabilities
        att_input = input_df.drop(columns=ATTACK_LABELS)
        att_probas = att_clf.predict_proba(att_input)
        att_scores = {ATTACK_NAMES[i]: float(att_probas[i][0][1]) for i in range(len(ATTACK_NAMES))}
        pred_attacks_count = sum(1 for v in att_scores.values() if v >= 0.5)

        # Composite score formula from dissertation
        comp_score = p_high + 0.5 * p_mod + 0.1 * pred_attacks_count

        if p_high >= 0.5 or comp_score >= 0.78:
            verdict_html = '<span class="badge-high">HIGH SEVERITY • FORMAL AUDIT REQUIRED</span>'
            gauge_color = "#DC2626"
        elif p_mod >= 0.45 or comp_score >= 0.45:
            verdict_html = '<span class="badge-mod">MODERATE SEVERITY • HARDENING ADVISED</span>'
            gauge_color = "#D97706"
        else:
            verdict_html = '<span class="badge-low">LOW SEVERITY • SOUND DESIGN PROFILE</span>'
            gauge_color = "#059669"

        st.markdown(f"**Triage Classification:** {verdict_html}", unsafe_allow_html=True)

        # Gauge Chart for Composite Score
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(comp_score, 3),
            number={'font': {'size': 36, 'color': '#0F172A'}},
            title={'text': "Composite Vulnerability Score (0.0 – 1.4)", 'font': {'size': 14, 'color': '#475569'}},
            gauge={
                'axis': {'range': [0, 1.4], 'tickwidth': 1, 'tickcolor': "#94A3B8"},
                'bar': {'color': gauge_color},
                'bgcolor': "#F8FAFC",
                'steps': [
                    {'range': [0, 0.45], 'color': '#DCFCE7'},
                    {'range': [0.45, 0.78], 'color': '#FEF3C7'},
                    {'range': [0.78, 1.4], 'color': '#FEE2E2'}
                ],
            }
        ))
        fig_gauge.update_layout(
            height=230, margin=dict(l=20, r=20, t=40, b=10),
            paper_bgcolor='rgba(0,0,0,0)', font={'color': '#1E293B'}
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Class Probability Bars
        prob_df = pd.DataFrame({
            'Severity': ['Low Risk', 'Moderate Risk', 'High Risk'],
            'Probability': [p_low, p_mod, p_high]
        })
        fig_prob = px.bar(
            prob_df, x='Probability', y='Severity', orientation='h',
            color='Severity',
            color_discrete_map={'Low Risk': '#10B981', 'Moderate Risk': '#F59E0B', 'High Risk': '#EF4444'},
            text=prob_df['Probability'].apply(lambda x: f"{x*100:.1f}%")
        )
        fig_prob.update_layout(
            height=170, showlegend=False, xaxis_range=[0, 1.12],
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            xaxis_title="", yaxis_title=""
        )
        fig_prob.update_traces(textposition='outside')
        st.plotly_chart(fig_prob, use_container_width=True)

        # Attack vector breakdown
        st.markdown("**Predicted Attack Vector Susceptibility**")
        att_df = pd.DataFrame({
            'Attack Vector': list(att_scores.keys()),
            'Risk Probability': list(att_scores.values())
        })
        fig_att = px.bar(
            att_df, x='Attack Vector', y='Risk Probability',
            color='Risk Probability',
            color_continuous_scale=['#99F6E4', '#F59E0B', '#DC2626'],
            text=att_df['Risk Probability'].apply(lambda x: f"{x*100:.0f}%")
        )
        fig_att.update_layout(
            height=210, coloraxis_showscale=False, yaxis_range=[0, 1.15],
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            xaxis_title="", yaxis_title="Probability"
        )
        fig_att.update_traces(textposition='outside')
        st.plotly_chart(fig_att, use_container_width=True)

        # Actionable Security Recommendations
        missing_tips = []
        if not user_inputs['identity_binding']:
            missing_tips.append("Bind peer identities inside encrypted payloads (prevents Needham-Schroeder / Woo-Lam MITM).")
        if not user_inputs['integrity_protection']:
            missing_tips.append("Add handshake integrity / MAC verification (prevents SSL 2.0/3.0 downgrade & tampering).")
        if not user_inputs['explicit_replay_protection'] and not user_inputs['uses_nonce']:
            missing_tips.append("Introduce fresh nonces or sequence numbers (prevents Denning-Sacco replay attacks).")
        if not user_inputs['forward_secrecy']:
            missing_tips.append("Use ephemeral key exchange (ECDHE/DHE) for Perfect Forward Secrecy.")
        if not user_inputs['modern_aead']:
            missing_tips.append("Upgrade legacy ciphers (RC4/DES/CBC) to modern AEAD (AES-GCM / ChaCha20-Poly1305).")

        if missing_tips:
            st.markdown("<div class='info-box'><b>💡 Formal Hardening Recommendations:</b><ul>" +
                        "".join(f"<li>{t}</li>" for t in missing_tips[:3]) + "</ul></div>", unsafe_allow_html=True)
        else:
            st.success("✅ Strong defensive posture: core freshness, identity binding, integrity, and AEAD controls are active.")

# ==================================================================
# TAB 2: 100-Protocol Explorer & Rankings
# ==================================================================
with tab2:
    st.subheader("📊 ML → Lean 4 Handover: All 100 Ranked Protocols")
    st.write(
        "Explore the complete 100-protocol dataset ranked by composite vulnerability score. "
        "Notice how the top 13 ranked protocols align 1-to-1 with the 13 High-severity protocols formally verified in Lean 4."
    )

    f1, f2, f3 = st.columns([1.2, 1.2, 1.6])
    sel_fams = f1.multiselect(
        "Filter by Family:",
        options=sorted(df_merged['family'].unique()),
        default=sorted(df_merged['family'].unique())
    )
    sel_sevs = f2.multiselect(
        "Filter by Actual Severity:",
        options=['High', 'Moderate', 'Low'],
        default=['High', 'Moderate', 'Low']
    )
    search_txt = f3.text_input("Search Protocol Name:", placeholder="e.g., TLS, Kerberos, WireGuard, WEP...")

    filtered = df_merged[
        (df_merged['family'].isin(sel_fams)) &
        (df_merged['actual_severity'].isin(sel_sevs))
    ]
    if search_txt.strip():
        filtered = filtered[filtered['protocol'].str.contains(search_txt.strip(), case=False, na=False)]

    # Scatter plot: Year Introduced vs Vulnerability Score
    fig_scatter = px.scatter(
        filtered,
        x='year_introduced',
        y='vulnerability_score',
        color='actual_severity',
        size='P_High',
        hover_name='protocol',
        hover_data={'rank': True, 'family': True, 'P_High': ':.3f', 'lean_status': True},
        color_discrete_map={'High': '#DC2626', 'Moderate': '#D97706', 'Low': '#0D9488'},
        labels={
            'year_introduced': 'Year Introduced',
            'vulnerability_score': 'Composite Vulnerability Score',
            'actual_severity': 'Severity Tier'
        },
        title="Historical Evolution of Protocol Risk (1976 – 2023)"
    )
    fig_scatter.add_hline(y=0.78, line_dash="dash", line_color="#DC2626", annotation_text="Lean 4 High-Risk Cutoff (Top 13)")
    fig_scatter.update_layout(
        height=390,
        paper_bgcolor='#FFFFFF',
        plot_bgcolor='#FAF9F6',
        margin=dict(l=20, r=20, t=45, b=20)
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    # Display clean interactive table
    display_cols = [
        'rank', 'protocol', 'family', 'year_introduced',
        'actual_severity', 'predicted_severity', 'P_High',
        'vulnerability_score', 'lean_status'
    ]
    st.dataframe(
        filtered[display_cols].rename(columns={
            'rank': 'Rank',
            'protocol': 'Protocol',
            'family': 'Family',
            'year_introduced': 'Year',
            'actual_severity': 'Actual Risk',
            'predicted_severity': 'ML Predicted',
            'P_High': 'P(High)',
            'vulnerability_score': 'Vuln. Score',
            'lean_status': 'Lean 4 Verification Status'
        }),
        use_container_width=True,
        hide_index=True,
        height=420
    )

# ==================================================================
# TAB 3: ML Sensitivity & Feature Drivers
# ==================================================================
with tab3:
    st.subheader("🔬 Empirical Rigour: Sensitivity Analysis & Feature Drivers")
    st.write(
        "A key research question in the dissertation is whether the ML classifier merely memorizes historical breach flags "
        "or genuinely learns structural cryptographic weaknesses. Removing all 4 historical-attack columns yields a Spearman rank correlation "
        "of **ρ = 0.919** (LOO accuracy 76.0% vs 77.0%)."
    )

    col_s1, col_s2 = st.columns(2, gap="large")

    with col_s1:
        st.markdown("#### ALL-Features vs DESIGN-Only Risk Probability")
        fig_sens = px.scatter(
            sens_df,
            x='P_High_ALL',
            y='P_High_DESIGN',
            color='actual',
            hover_name='protocol',
            hover_data={'family': True, 'delta': ':.3f'},
            color_discrete_map={'High': '#DC2626', 'Moderate': '#D97706', 'Low': '#0D9488'},
            labels={
                'P_High_ALL': 'P(High) — All 21 Features',
                'P_High_DESIGN': 'P(High) — 17 Design-Only Features',
                'actual': 'Actual Risk'
            }
        )
        fig_sens.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(color="#94A3B8", dash="dot"))
        fig_sens.update_layout(
            height=410,
            paper_bgcolor='#FFFFFF',
            plot_bgcolor='#FAF9F6',
            margin=dict(l=20, r=20, t=20, b=20)
        )
        st.plotly_chart(fig_sens, use_container_width=True)

    with col_s2:
        st.markdown("#### Top 12 Cryptographic Risk Drivers (Random Forest)")
        top_imp = imp_df.head(12).iloc[::-1]
        fig_imp = px.bar(
            top_imp,
            x='Importance',
            y='Feature',
            orientation='h',
            color='Importance',
            color_continuous_scale=['#99F6E4', '#0F766E'],
            text=top_imp['Importance'].apply(lambda x: f"{x:.3f}")
        )
        fig_imp.update_layout(
            height=410,
            coloraxis_showscale=False,
            paper_bgcolor='#FFFFFF',
            plot_bgcolor='#FAF9F6',
            margin=dict(l=20, r=30, t=20, b=20),
            xaxis_title="Gini Impurity Importance",
            yaxis_title=""
        )
        fig_imp.update_traces(textposition='outside')
        st.plotly_chart(fig_imp, use_container_width=True)

# ==================================================================
# TAB 4: Lean 4 Formal Proof Inspector
# ==================================================================
with tab4:
    st.subheader("📐 Lean 4 Machine-Checked Proof Inspector")
    st.write(
        "Inspect the formal Lean 4 theorems verifying all **13 High-severity protocols** plus **2 safety proofs** "
        "(`NeedhamSchroeder_Lowe` structural safety and `ChallengeResponseSafety` unbounded-session inductive safety). "
        "Every file compiles against the stock Lean 4 core library (`v4.32.2`) with zero external dependencies."
    )

    proof_keys = list(LEAN_PROOFS.keys())
    selected_proof_key = st.selectbox(
        "Select a Formalised Protocol or Safety Proof:",
        options=proof_keys,
        format_func=lambda k: f"{LEAN_PROOFS[k]['title']}  [{LEAN_PROOFS[k]['verdict']}]"
    )

    p_data = LEAN_PROOFS[selected_proof_key]

    p_c1, p_c2, p_c3 = st.columns(3)
    p_c1.info(f"**Source Artefact:** `{p_data['file']}`")
    p_c2.info(f"**Proof Methodology:** {p_data['style']}")
    if "SAFE" in p_data['verdict']:
        p_c3.success(f"**Result:** {p_data['verdict']}")
    else:
        p_c3.error(f"**Result:** {p_data['verdict']}")

    st.markdown(f"<div class='info-box'><b>Cryptographic Mechanism & Proof Summary:</b><br>{p_data['flaw']}<br><br><b>Certified Lean 4 Theorem(s):</b> <code>{p_data['theorem']}</code></div>", unsafe_allow_html=True)

    st.markdown("**Machine-Checked Lean 4 Source Extract:**")
    st.code(p_data['code'], language="lean")
