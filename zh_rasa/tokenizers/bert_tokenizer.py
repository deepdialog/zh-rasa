import os
import logging
from typing import Any, Dict, List, Text

from rasa.nlu.tokenizers.tokenizer import Token, Tokenizer
from rasa.shared.nlu.training_data.message import Message

logger = logging.getLogger(__name__)

current_dir = os.path.realpath(os.path.dirname(__file__))
vocab_path = os.path.join(current_dir, 'vocab.txt')


class BertTokenizer(Tokenizer):
    """HuaggingFace's Transformers based tokenizer."""

    defaults = {
        # URL to tokenizer service
        "tokenizer_url": None,
    }

    def __init__(self, component_config: Dict[Text, Any] = None) -> None:  # noqa: D107
        from tokenizers import BertWordPieceTokenizer
        self.tokenizer = BertWordPieceTokenizer(component_config.get('vocab_path', vocab_path))
        super().__init__(component_config)

    @classmethod
    def required_packages(cls) -> List[Text]:  # noqa: D102
        return ["tokenizers"]

    def tokenize(self, message: Message, attribute: Text) -> List[Token]:  # noqa: D102
        text = message.get(attribute)

        ret = self.tokenizer.encode(text, add_special_tokens=False)
        tokens = []

        for token_text, (start, end) in ret:
            token = Token(token_text, start, end)
            tokens.append(token)

        return self._apply_token_pattern(tokens)
