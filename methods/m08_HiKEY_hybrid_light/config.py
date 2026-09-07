"""
m08_HiKEY_hybrid_light configuration.

This method keeps m05's question+option retrieval flow, but adds a local
structure-aware scoring layer over HiKEY evidence instead of using an LLM
section reranker.
"""
import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from methods._shared.config_base import *  # noqa: F401,F403
from methods._shared.config_base import DOMAINS, PROJECT_ROOT

METHOD_NAME = "m08_HiKEY_hybrid_light"
METHOD_ID = METHOD_NAME
METHOD_ROOT = Path(__file__).resolve().parent

MODEL_NAME = os.getenv("M08_MODEL", "qwen-plus")
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")

# Reuse the m04 HiKEY parser cache: doc_card / sections / units / field_cards.
HIKEY_INDEX_DIR = PROJECT_ROOT / "methods" / "_shared" / "_HiKEY_cache"
REFERENCE_ANSWERS_PATH = Path(
    os.getenv("M08_REFERENCE_ANSWERS", str(PROJECT_ROOT / "GPT-Pro-Answer" / "gpt_pro_all_answers.json"))
)

# Output owner. BaseRunner will write to runs/{CODER}/{METHOD_NAME}/{run_id}/.
CODER = os.getenv("CODER", "qhl")
RUN_DESC = os.getenv("M08_RUN_DESC", "hybrid-light")

CONCURRENCY = int(os.getenv("M08_CONCURRENCY", "10"))
TEMPERATURE = float(os.getenv("M08_TEMPERATURE", "0.1"))

# Qwen thinking mode follows m04 defaults. Disable via M08_ENABLE_THINKING=0.
ENABLE_THINKING = os.getenv("M08_ENABLE_THINKING", "1") != "0"
THINKING_BUDGET = int(os.getenv("M08_THINKING_BUDGET", "10000"))
MAX_OUTPUT_TOKENS = int(os.getenv("M08_MAX_OUTPUT_TOKENS", "12000"))
MAX_OUTPUT_TOKENS_REFLECTION = int(os.getenv("M08_MAX_OUTPUT_TOKENS_REFLECTION", "16000"))

# Retrieval: keep m05's multi-query recall, then add local structure scoring.
GLOBAL_TOPK = int(os.getenv("M08_GLOBAL_TOPK", "10"))
OPTION_TOPK = int(os.getenv("M08_OPTION_TOPK", "10"))
FINAL_EVIDENCE_TOPK = int(os.getenv("M08_FINAL_EVIDENCE_TOPK", "28"))
MAX_SIBLINGS = int(os.getenv("M08_MAX_SIBLINGS", "12"))
MIN_OPTION_PROXY_COVERAGE = float(os.getenv("M08_MIN_OPTION_PROXY_COVERAGE", "0.5"))

# Keep the final prompt bounded even though we run multiple local searches.
EVIDENCE_TOKEN_BUDGET = int(os.getenv("M08_EVIDENCE_TOKEN_BUDGET", "24000"))
SKIP_DOC_ROUTING = os.getenv("M08_SKIP_DOC_ROUTING", "1") != "0"

# Hybrid-local scoring knobs.
HYBRID_FIELD_CARD_BONUS = float(os.getenv("M08_FIELD_CARD_BONUS", "3.0"))
HYBRID_FINANCIAL_FIELD_BONUS = float(os.getenv("M08_FINANCIAL_FIELD_BONUS", "2.0"))
HYBRID_TABLE_ROW_BONUS = float(os.getenv("M08_TABLE_ROW_BONUS", "1.8"))
HYBRID_TABLE_BONUS = float(os.getenv("M08_TABLE_BONUS", "1.0"))
HYBRID_HEADING_BONUS = float(os.getenv("M08_HEADING_BONUS", "0.3"))
HYBRID_DEPTH_BONUS = float(os.getenv("M08_DEPTH_BONUS", "0.15"))
HYBRID_QUERY_HIT_BONUS = float(os.getenv("M08_QUERY_HIT_BONUS", "1.0"))
HYBRID_GENERIC_PENALTY = float(os.getenv("M08_GENERIC_PENALTY", "0.8"))
HYBRID_NUMERIC_BONUS = float(os.getenv("M08_NUMERIC_BONUS", "0.4"))
HYBRID_YEAR_BONUS = float(os.getenv("M08_YEAR_BONUS", "0.8"))
HYBRID_COMPANY_BONUS = float(os.getenv("M08_COMPANY_BONUS", "1.0"))
HYBRID_METRIC_ALIAS_BONUS = float(os.getenv("M08_METRIC_ALIAS_BONUS", "2.2"))
HYBRID_METRIC_EXACT_BONUS = float(os.getenv("M08_METRIC_EXACT_BONUS", "4.0"))

HIKEY_DOMAINS = ["financial_reports", "insurance", "regulatory", "financial_contracts", "research"]

SYSTEM_PROMPT = """浣犳槸涓€浣嶉噾铻嶆枃妗ｅ垎鏋愪笓瀹躲€備綘蹇呴』涓ユ牸鏍规嵁缁欏畾璇佹嵁鍥炵瓟閫夋嫨棰樸€?
鍙緭鍑虹瓟妗堝瓧姣嶏紝涓嶈杈撳嚭瑙ｉ噴銆?鍗曢€夐杈撳嚭涓€涓瓧姣嶏紝渚嬪 A銆?澶氶€夐杈撳嚭鎵€鏈夋纭€夐」瀛楁瘝锛屾寜瀛楁瘝椤哄簭鎺掑垪锛屼緥濡?ACD銆?鍒ゆ柇棰樿緭鍑?A 鎴?B锛屼互棰樼洰閫夐」鍚箟涓哄噯銆?"""

SYSTEM_PROMPT_REFLECTION = """浣犳槸涓€浣嶉噾铻嶆枃妗ｅ垎鏋愪笓瀹躲€傝涓ユ牸鏍规嵁璇佹嵁閫愰」鏍搁獙閫夐」銆?鏈€鍚庝竴琛屽彧杈撳嚭鏈€缁堢瓟妗堝瓧姣嶃€?"""

FORMAT_HINTS = {
    "mcq": "鏈鏄崟閫夐锛屽彧鑳戒粠 A/B/C/D 涓€夋嫨鍞竴姝ｇ‘绛旀銆?",
    "multi": "鏈鏄閫夐锛岄渶瑕侀€夋嫨鎵€鏈夋纭€夐」锛岀瓟妗堟寜瀛楁瘝椤哄簭鎺掑垪锛屼笉瑕侀仐婕忔纭€夐」銆?",
    "tf": "鏈鏄垽鏂锛岃鎸夐鐩€夐」鍚箟杈撳嚭 A 鎴?B銆?",
}

DOMAIN_HINTS = {
    "insurance": "娉ㄦ剰淇濋櫓璐ｄ换銆佸厤璐ｆ潯娆俱€佺瓑寰呮湡銆佽禂浠樻潯浠躲€佽矗浠昏寖鍥村拰渚嬪鏉′欢銆?",
    "regulatory": "娉ㄦ剰娉曡閫傜敤鑼冨洿銆佷箟鍔″己搴︺€佹椂闄愯姹傘€佷緥澶栨潯浠讹紝浠ュ強鈥滃簲褰?鍙互/涓嶅緱鈥濈殑鍖哄埆銆?",
    "financial_contracts": "娉ㄦ剰鍙戣閲戦涓庢敞鍐岄噾棰濄€佽瘎绾с€佹湡闄愩€佸埄鐜囥€佸洖鍞祹鍥炪€佽繚绾︽潯娆惧拰涓讳綋瑙掕壊銆?",
    "financial_reports": "娉ㄦ剰骞翠唤銆佸崟浣嶃€佸悓姣斻€佹瘡鑲?姣?0鑲°€佸綊姣?鎵ｉ潪銆佺爺鍙戞姇鍏?鐮斿彂璐圭敤绛夊彛寰勫樊寮傘€?",
    "research": "娉ㄦ剰鐮旀姤涓殑瓒嬪娍銆佸浘琛ㄦ暟鍊笺€佸崟浣嶃€佸叕鍙?琛屼笟瀵规瘮鍜岀粨璁洪€傜敤鑼冨洿銆?",
}

QA_PROMPT_TEMPLATE = """涓嬮潰鏄粠鐩稿叧鏂囨。涓绱㈠埌鐨勮瘉鎹€傝瘉鎹凡缁忕敤鈥滈骞测€濆拰鈥淎/B/C/D鍚勯€夐」鈥濆垎鍒绱㈠悗鍘婚噸鍚堝苟銆?
{evidence}

---

{format_hint}
{domain_hint}

棰樼洰锛?{question}

閫夐」锛?{options}

璇锋牴鎹瘉鎹垽鏂摢涓€夐」姝ｇ‘銆傚彧杈撳嚭绛旀瀛楁瘝锛屼笉瑕佽緭鍑哄叾浠栧唴瀹广€?"""

MULTI_QA_PROMPT_TEMPLATE = """涓嬮潰鏄粠鐩稿叧鏂囨。涓绱㈠埌鐨勮瘉鎹€傝瘉鎹凡缁忕敤鈥滈骞测€濆拰鈥淎/B/C/D鍚勯€夐」鈥濆垎鍒绱㈠悗鍘婚噸鍚堝苟銆?
{evidence}

---

{format_hint}
{domain_hint}

棰樼洰锛?{question}

閫夐」锛?{options}

璇烽€愰」鍒ゆ柇 A/B/C/D 鏄惁琚瘉鎹敮鎸侊紝鏈€鍚庡彧杈撳嚭鎵€鏈夋纭€夐」瀛楁瘝锛屼緥濡?AC 鎴?BCD銆備笉瑕佽緭鍑鸿В閲娿€?"""

TF_PROMPT_TEMPLATE = """涓嬮潰鏄粠鐩稿叧鏂囨。涓绱㈠埌鐨勮瘉鎹€傝瘉鎹凡缁忕敤鈥滈骞测€濆拰鈥淎/B/C/D鍚勯€夐」鈥濆垎鍒绱㈠悗鍘婚噸鍚堝苟銆?
{evidence}

---

{format_hint}
{domain_hint}

棰樼洰锛?{question}

閫夐」锛?{options}

璇锋牴鎹瘉鎹垽鏂鐩檲杩版槸鍚︽纭€傚彧杈撳嚭 A 鎴?B锛屼笉瑕佽緭鍑哄叾浠栧唴瀹广€?"""

MULTI_REFLECTION_PROMPT = """浣犱箣鍓嶅涓€閬撳閫夐鍙€夋嫨浜?{previous_answer}銆傝繖鍙兘閬楁紡浜嗗叾浠栨纭€夐」銆?
璇佹嵁锛?{evidence}

棰樼洰锛?{question}

閫夐」锛?{options}

{domain_hint}

璇烽噸鏂伴€愰」鍒ゆ柇 A/B/C/D銆傛渶鍚庝竴琛屽彧杈撳嚭鏈€缁堢瓟妗堝瓧姣嶏紝鎸夊瓧姣嶉『搴忔帓鍒椼€?"""
