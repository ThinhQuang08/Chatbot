from __future__ import annotations
from typing import Any, Dict, List, Optional, Text

import regex

import rasa.shared.utils.io
import rasa.utils.io

from rasa.engine.graph import ExecutionContext
from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.nlu.tokenizers.tokenizer import Token, Tokenizer
from rasa.shared.nlu.training_data.message import Message


@DefaultV1Recipe.register(
    DefaultV1Recipe.ComponentType.MESSAGE_TOKENIZER, is_trainable=False
)
class PyViTokenizer(Tokenizer):
    @staticmethod
    def not_supported_languages() -> Optional[List[Text]]:
        return ["zh", "ja", "th"]

    @staticmethod
    def get_default_config() -> Dict[Text, Any]:
        return {
            "intent_tokenization_flag": False,
            "intent_split_symbol": "_",
            "token_pattern": None,
            "prefix_separator_symbol": None,
        }

    def __init__(self, config: Dict[Text, Any]) -> None:
        super().__init__(config)
        self.emoji_pattern = rasa.utils.io.get_emoji_regex()
        try:
            from pyvi import ViTokenizer
            self._pyvi_tokenize = ViTokenizer.tokenize
        except ImportError:
            raise ImportError(
                "pyvi is not installed. Run `pip install pyvi`."
            )

        if "case_sensitive" in self._config:
            rasa.shared.utils.io.raise_warning(
                "The option 'case_sensitive' was moved from the tokenizers to the "
                "featurizers.",
            )

    @classmethod
    def create(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> PyViTokenizer:
        return cls(config)

    def remove_emoji(self, text: Text) -> Text:
        match = self.emoji_pattern.fullmatch(text)
        if match is not None:
            return ""
        return text

    def tokenize(self, message: Message, attribute: Text) -> List[Token]:
        text = message.get(attribute)
        if not text:
            return []

        raw_tokens = regex.sub(
            r"[^\w#@&]+(?=\s|$)|"
            r"(\s|^)[^\w#@&]+(?=[^0-9\s])|"
            r"(?<=[^0-9\s])[^\w._~:/?#\[\]()@!$&*+,;=-]+(?=[^0-9\s])",
            " ",
            text,
        ).split()

        pyvi_text = " ".join(raw_tokens) if raw_tokens else text
        pyvi_result = self._pyvi_tokenize(pyvi_text)

        pyvi_tokens = pyvi_result.split()

        words = []
        for pt in pyvi_tokens:
            clean = self.remove_emoji(pt.replace("_", " "))
            if clean:
                words.append(clean)

        if not words:
            words = [text]

        tokens = self._convert_words_to_tokens(words, text)
        return self._apply_token_pattern(tokens)
