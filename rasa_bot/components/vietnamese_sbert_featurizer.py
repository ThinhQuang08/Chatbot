from __future__ import annotations
import numpy as np
import logging
from typing import Any, Text, List, Dict, Type

from rasa.engine.graph import ExecutionContext, GraphComponent
from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.nlu.featurizers.dense_featurizer.dense_featurizer import DenseFeaturizer
from rasa.nlu.tokenizers.tokenizer import Tokenizer
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.shared.nlu.training_data.message import Message
from rasa.nlu.constants import (
    DENSE_FEATURIZABLE_ATTRIBUTES,
    FEATURIZER_CLASS_ALIAS,
    TOKENS_NAMES,
)
from rasa.shared.nlu.constants import TEXT
from rasa.utils.tensorflow.constants import POOLING, MEAN_POOLING

logger = logging.getLogger(__name__)


@DefaultV1Recipe.register(
    DefaultV1Recipe.ComponentType.MESSAGE_FEATURIZER, is_trainable=False
)
class VietnameseSBertFeaturizer(DenseFeaturizer, GraphComponent):
    @classmethod
    def required_components(cls) -> List[Type]:
        return [Tokenizer]

    @staticmethod
    def required_packages() -> List[Text]:
        return ["sentence_transformers"]

    @staticmethod
    def get_default_config() -> Dict[Text, Any]:
        return {
            **DenseFeaturizer.get_default_config(),
            "model_name": "keepitreal/vietnamese-sbert",
            "cache_dir": None,
            POOLING: MEAN_POOLING,
        }

    def __init__(
        self, config: Dict[Text, Any], execution_context: ExecutionContext
    ) -> None:
        super().__init__(execution_context.node_name, config)
        self.pooling_operation = self._config[POOLING]
        self._load_model()

    @classmethod
    def validate_config(cls, config: Dict[Text, Any]) -> None:
        pass

    @classmethod
    def create(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> VietnameseSBertFeaturizer:
        return cls(config, execution_context)

    def _load_model(self) -> None:
        from sentence_transformers import SentenceTransformer

        model_name = self._config["model_name"]
        cache_dir = self._config.get("cache_dir")
        logger.info(f"Loading SentenceTransformer model: {model_name}")
        self.model = SentenceTransformer(model_name, cache_folder=cache_dir)

    def process(self, messages: List[Message]) -> List[Message]:
        for message in messages:
            self._set_features(message)
        return messages

    def process_training_data(self, training_data: TrainingData) -> TrainingData:
        self.process(training_data.training_examples)
        return training_data

    def _set_features(self, message: Message, attribute: Text = TEXT) -> None:
        text = message.get(attribute)
        if not text:
            return

        tokens = message.get(TOKENS_NAMES[attribute], [])
        embedding = self.model.encode(text, show_progress_bar=False)

        sentence_features = np.reshape(embedding, (1, -1))

        if tokens:
            sequence_features = np.tile(
                embedding, (len(tokens), 1)
            )
        else:
            sequence_features = np.array([embedding])

        self.add_features_to_message(
            sequence=sequence_features,
            sentence=sentence_features,
            attribute=attribute,
            message=message,
        )
