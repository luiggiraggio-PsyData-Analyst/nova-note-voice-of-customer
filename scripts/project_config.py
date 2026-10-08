from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATA_RAW = REPO / "data" / "raw"
TRANSCRIPTS_DIR = DATA_RAW / "transcripts"
PARTICIPANTS_CSV = DATA_RAW / "participants.csv"
PROCESSED = REPO / "data" / "processed"
OUTPUTS = REPO / "outputs"
FIGURES = OUTPUTS / "figures"
TABLES = OUTPUTS / "tables"
NETWORK = OUTPUTS / "network"
DOCS = REPO / "docs"

for _d in (PROCESSED, FIGURES, TABLES, NETWORK, DOCS):
    _d.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"
SENTIMENT_LABELS = ["negative", "neutral", "positive"]
VADER_POS = 0.05
VADER_NEG = -0.05

THEMES = [
    "onboarding",
    "pricing",
    "integrations",
    "api",
    "performance",
    "mobile",
    "collaboration",
    "templates",
    "migration",
    "support",
    "security",
    "roadmap",
    "competitor",
]

PATTERNS = {
    "onboarding": r"\bonboarding\b|setup wizard|quick tour|guest access|learning curve|starter workspace|getting started|new hires?",
    "pricing": r"\bpric(e|ing)\b|\bplans?\b|renewal|billed?|billing|charged?|charge|cost|\bexpensive\b|upgrade|refund|tiers?",
    "integrations": r"integrat|connector|webhook|automation|sync|crm|lms|exporter|embedded|native connectors?",
    "api": r"\bapi\b|\bsdk\b|rate limits?|endpoints?",
    "performance": r"performance|slow|lags?|lagged?|freeze|froze|fast|speed|seconds|unusable|drained|loading a page|load quickly",
    "mobile": r"mobile|\bphone\b|offline|on the go",
    "collaboration": r"collaborat|comments?|real-?time|permissions?|access control|share[d]?|guests?|assign|concurren|revision history|deleted a page",
    "templates": r"templates?|blueprint|master cop",
    "migration": r"migrat|import|export|lock-?in|orphaned|moved|switched to",
    "support": r"support|ticket|agent|responded?|reply|replied|closed as|escalat|response",
    "security": r"securit|sso|audit|retention|residency|data center|compliance|\bssa",
    "roadmap": r"roadmap|planned|timeline|feature requests?|suggestion forum|released a|public roadmap",
    "competitor": r"notion|obsidian|other (?:tool|app)|competitor|alternatives?|compared|phasing out|different product",
}

ACTION_BY_CLUSTER = {
    "trust_cost": "Review pricing tiers, add transparency on plan changes and publish an honest value case.",
    "scale_code": "Fix performance at scale, stabilize the API and webhooks, and publish a public integration roadmap.",
    "collab_structure": "Add document-level permissions, assignable comments, versioned templates and faster collaboration at scale.",
    "support_ops": "Shorten first reply time, add escalation paths and publish known-bug status on the public roadmap.",
    "mobile_security": "Rebuild offline reliability on mobile and publish security, residency and SSO documentation.",
}