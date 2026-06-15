import requests
import logging

logger = logging.getLogger(__name__)


class LLMService:

    DEFAULT_MODEL = "phi3"
    TIMEOUT       = 30

    SYSTEM_PROMPT = (
        "You are an expert agricultural advisor specializing in tomato diseases. "
        "Give concise, practical English advice. Maximum 3-4 sentences. "
        "Be direct — no unnecessary preamble."
    )

    def __init__(self, ollama_url: str = "http://localhost:11434", model: str = None):
        self.ollama_url = ollama_url.rstrip("/")
        self.model      = model or self.DEFAULT_MODEL
        self._available = None

    def get_advice(self, item_type: str, condition: str, extra_info: str = "") -> str:
        prompt = self._build_prompt(item_type, condition, extra_info)
        try:
            resp = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model":  self.model,
                    "prompt": prompt,
                    "system": self.SYSTEM_PROMPT,
                    "stream": False,
                    "options": {"temperature": 0.3, "num_predict": 200},
                },
                timeout=self.TIMEOUT,
            )
            if resp.status_code == 200:
                advice = resp.json().get("response", "").strip()
                self._available = True
                return advice or self._fallback(item_type, condition)
            logger.warning(f"Ollama status: {resp.status_code}")
        except requests.exceptions.ConnectionError:
            logger.warning("Ollama not reachable — using fallback advice.")
            self._available = False
        except Exception as e:
            logger.error(f"LLM error: {e}")
        return self._fallback(item_type, condition)

    def _build_prompt(self, item_type: str, condition: str, extra_info: str) -> str:
        if item_type == "leaf":
            return (
                f"A tomato leaf has been diagnosed with '{condition}'. "
                f"{extra_info}. "
                f"Give 3 practical treatment steps for the farmer."
            )
        return (
            f"A tomato fruit has the following issue: '{condition}'. "
            f"{extra_info}. "
            f"Give 3 practical steps to preserve quality or address the problem."
        )

    def _fallback(self, item_type: str, condition: str) -> str:
        FALLBACKS = {
            "Tomato__Bacterial_spot": (
                "Bacterial spot detected. Remove and destroy infected leaves immediately. "
                "Apply copper-based bactericide every 7-10 days. "
                "Avoid overhead irrigation to reduce leaf wetness."
            ),
            "Tomato__Early_blight": (
                "Early blight detected. Remove infected lower leaves and destroy them. "
                "Apply chlorothalonil or copper-based fungicide. "
                "Ensure proper plant spacing for air circulation."
            ),
            "Tomato__Late_blight": (
                "Late blight detected — act immediately, it spreads fast. "
                "Apply metalaxyl or mancozeb-based fungicide right away. "
                "Remove and bag all infected plant material; do not compost."
            ),
            "Tomato__Leaf_Mold": (
                "Leaf mold detected. Reduce greenhouse humidity below 85% and improve ventilation. "
                "Apply mancozeb or chlorothalonil fungicide. "
                "Remove infected leaves to slow the spread."
            ),
            "Tomato__Septoria_leaf_spot": (
                "Septoria leaf spot detected. Remove all affected lower leaves. "
                "Apply fungicide (chlorothalonil or copper-based) weekly. "
                "Avoid wetting foliage during irrigation."
            ),
            "Tomato__Spider_mites_Two-spotted_spider_mite": (
                "Two-spotted spider mite infestation detected. "
                "Apply miticide (abamectin or bifenazate) — rotate products to prevent resistance. "
                "Increase humidity around plants; introduce predatory mites (Phytoseiulus persimilis) if available."
            ),
            "Tomato__Target_Spot": (
                "Target spot detected. Apply azoxystrobin or chlorothalonil fungicide. "
                "Remove infected leaves and improve canopy airflow. "
                "Avoid excessive nitrogen fertilization which promotes lush, susceptible growth."
            ),
            "Tomato__Tomato_mosaic_virus": (
                "Tomato mosaic virus detected. No cure exists — remove and destroy infected plants immediately. "
                "Disinfect tools with 10% bleach solution between plants. "
                "Control aphid and whitefly populations which spread the virus."
            ),
            "Tomato__Tomato_Yellow_Leaf_Curl_Virus": (
                "Yellow leaf curl virus detected. Remove infected plants to prevent spread. "
                "Control whitefly populations aggressively with imidacloprid or reflective mulches. "
                "Use virus-resistant tomato varieties for future planting."
            ),
            "Tomato__healthy": (
                "Leaf appears healthy. Continue regular monitoring and preventive fungicide schedule. "
                "Maintain good air circulation and avoid overhead watering."
            ),
            "Damaged": (
                "Fruit damage detected. Separate affected fruits from healthy ones immediately. "
                "Inspect for insect entry points or mechanical damage during handling. "
                "Damaged fruits should be processed quickly to avoid secondary fungal infection."
            ),
            "Old": (
                "Fruit is overripe or showing age-related deterioration. "
                "Harvest immediately if still on the plant. "
                "Consider processing into by-products (sauce, paste) rather than fresh sale."
            ),
        }

        if condition in FALLBACKS:
            return FALLBACKS[condition]

        for key, advice in FALLBACKS.items():
            if key.lower() in condition.lower():
                return advice

        base = "tomato leaf" if item_type == "leaf" else "tomato fruit"
        return (
            f"Issue detected on {base}: '{condition}'. "
            "Remove affected areas, apply appropriate treatment, and consult an agricultural specialist."
        )

    def is_available(self) -> bool:
        if self._available is not None:
            return self._available
        try:
            r = requests.get(f"{self.ollama_url}/api/tags", timeout=3)
            self._available = r.status_code == 200
        except Exception:
            self._available = False
        return self._available