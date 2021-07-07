import logging
import os
import pickle
from typing import Any, Dict, Optional, Text

from rasa.nlu.model import Metadata
from rasa.nlu.components import Component
from rasa.nlu.config import RasaNLUModelConfig
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.shared.nlu.constants import INTENT, TEXT

logger = logging.getLogger(__name__)


class TFNLUClassifier(Component):

    supported_language_list = ["zh"]

    name = "addons_intent_classifier_tfnlu"

    provides = ["intent", "intent_ranking"]

    def __init__(self,
                 component_config: Optional[Dict[Text, Any]],
                 model=None) -> None:

        self.model = model
        self.result_dir = None if 'result_dir' not in component_config else component_config['result_dir']
        self.batch_size = component_config.get("batch_size", 32)
        self.epochs = component_config.get("epochs", 20)
        self.encoder_path = component_config.get('encoder_path', None)

        super(TFNLUClassifier, self).__init__(component_config)

    @classmethod
    def required_packages(cls):
        return ["tensorflow", "tfnlu"]

    def train(self,
              training_data: TrainingData,
              config: RasaNLUModelConfig,
              **kwargs: Any) -> None:

        from tfnlu import Classification

        X = []
        Y = []
        for ex in training_data.intent_examples:
            text = ex.get(TEXT)
            intent = ex.get(INTENT)
            X.append(list(text))
            Y.append(intent)

        self.model = model = Classification(encoder_path=self.encoder_path)
        model.fit(X, Y, batch_size=min(32, len(X)), epochs=20)

    @classmethod
    def load(
        cls,
        meta: Dict[Text, Any],
        model_dir: Optional[Text] = None,
        model_metadata: Optional["Metadata"] = None,
        cached_component: Optional["Component"] = None,
        **kwargs: Any
    ) -> Component:
        if cached_component:
            return cached_component
        else:
            path = os.path.join(model_dir, meta['name'] + '.pkl')
            with open(path, 'rb') as fp:
                model = pickle.load(fp)
            return cls(meta, model)

    def process(self, message: Message, **kwargs: Any) -> None:
        text = message.get(TEXT)
        if text:
            logger.debug('predict intent %s', text)
            pred, probs = self.model.predict_proba([list(text)], verbose=0)
            intent = {"name": pred[0], "confidence": probs[0]}
            logger.debug('predict intent %s %s', text, pred[0])
            print(intent)
            message.set(INTENT, intent, add_to_output=True)

        if message.get(INTENT) is not None:
            return

    def persist(self, file_name: Text, model_dir: Text) -> Dict[Text, Any]:
        """Persist this model into the passed directory.
        Returns the metadata necessary to load the model again."""

        path = os.path.join(model_dir, self.name + '.pkl')
        with open(path, 'wb') as fp:
            pickle.dump(self.model, fp)
        return { 'name': self.name }
