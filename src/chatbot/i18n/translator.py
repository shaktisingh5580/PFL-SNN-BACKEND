"""
Multilingual translation layer.
Auto-detects input language and translates between English, Hindi, and Gujarati.
"""
import logging
import re
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

try:
    from googletrans import Translator as GoogleTranslator
    HAS_GOOGLETRANS = True
except ImportError:
    HAS_GOOGLETRANS = False
    logger.info("googletrans not installed. Using basic translation fallback.")


class Translator:
    """Multilingual translation for Hindi, Gujarati, and English."""

    LANG_CODES = {"en": "en", "hi": "hi", "gu": "gu"}
    LANG_NAMES = {"en": "English", "hi": "Hindi", "gu": "Gujarati"}

    def __init__(self):
        self._translator = GoogleTranslator() if HAS_GOOGLETRANS else None

    def detect_language(self, text: str) -> str:
        """
        Detect input language.

        Returns:
            Language code: "en", "hi", or "gu"
        """
        # Gujarati Unicode block: U+0A80 to U+0AFF
        gujarati_chars = len(re.findall(r'[\u0A80-\u0AFF]', text))
        # Devanagari Unicode block: U+0900 to U+097F (Hindi)
        devanagari_chars = len(re.findall(r'[\u0900-\u097F]', text))
        total_chars = len(text.strip())

        if total_chars == 0:
            return "en"

        if gujarati_chars / total_chars > 0.2:
            return "gu"
        elif devanagari_chars / total_chars > 0.2:
            return "hi"
        return "en"

    def translate(self, text: str, source: str, target: str) -> str:
        """
        Translate text between languages.

        Args:
            text: Input text
            source: Source language code ("en", "hi", "gu")
            target: Target language code

        Returns:
            Translated text
        """
        if source == target:
            return text

        if self._translator:
            try:
                result = self._translator.translate(text, src=source, dest=target)
                return result.text
            except Exception as e:
                logger.warning(f"Translation failed ({source}→{target}): {e}")

        # Fallback: return original with a note
        return f"[{self.LANG_NAMES.get(target, target)}] {text}"

    def translate_to_english(self, text: str) -> Tuple[str, str]:
        """
        Detect language and translate to English if needed.

        Returns:
            (english_text, detected_language)
        """
        lang = self.detect_language(text)
        if lang == "en":
            return text, "en"
        translated = self.translate(text, lang, "en")
        return translated, lang

    def translate_from_english(self, text: str, target: str) -> str:
        """Translate English text to the target language."""
        if target == "en":
            return text
        return self.translate(text, "en", target)

    def process_pipeline(self, text: str) -> dict:
        """
        Full translation pipeline for chatbot:
        1. Detect language
        2. Translate to English (for LLM processing)
        3. Return original language for response translation

        Returns:
            {"original": str, "english": str, "language": str}
        """
        lang = self.detect_language(text)
        english = self.translate_to_english(text)[0] if lang != "en" else text

        return {
            "original": text,
            "english": english,
            "language": lang,
        }
