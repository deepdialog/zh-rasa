import logging
import os
import pickle
import shutil
import tempfile
from typing import Any, Dict, Optional, Text

from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.extractors.extractor import EntityExtractor
from rasa.nlu.model import Metadata
from rasa.nlu.components import Component
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.shared.nlu.constants import ENTITIES, TEXT


logger = logging.getLogger(__name__)


class TFNLUExtractor(EntityExtractor):
    name = "addons_ner_tfnlu"

    provides = ["entities"]

    requires = ["tensorflow", "tfnlu"]

    def __init__(self,
                 component_config: Optional[Dict[Text, Any]] = None,
                 model=None) -> None:

        self.model = model
        self.result_dir = None if 'result_dir' not in component_config else component_config['result_dir']
        self.batch_size = component_config.get("batch_size", 32)
        self.epochs = component_config.get("epochs", 20)
        self.encoder_path = component_config.get('encoder_path', None)

        super(TFNLUExtractor, self).__init__(component_config)

    @classmethod
    def required_packages(cls):
        return ["tensorflow", "tfnlu"]

    def train(
        self, training_data: TrainingData, cfg: RasaNLUModelConfig, **kwargs: Any
    ) -> None:

        from tfnlu import Tagger

        X = []
        Y = []
        for ex in training_data.nlu_examples:
            text = ex.get(TEXT)
            entities = ex.get(ENTITIES)
            x = list(text)
            y = ['O'] * len(x)
            if entities is not None:
                for e in entities:
                    for i in range(e['start'], e['end']):
                        y[i] = 'I' + e['entity']
                    y[e['start']] = 'B' + e['entity']
            X.append(x)
            Y.append(y)

        model = Tagger(encoder_path=self.encoder_path)
        model.fit(X, Y, validation_data=(X, Y), batch_size=min(len(X), 32), epochs=100)
        self.result_dir = tempfile.mkdtemp()
        model_path = os.path.join(self.result_dir, 'model.pkl')
        with open(model_path, 'wb') as fp:
            pickle.dump(model, fp)

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
            real_result_dir = os.path.join(model_dir, meta['result_dir'])
            model_path = os.path.join(real_result_dir, 'model.pkl')
            with open(model_path, 'rb') as fp:
                model = pickle.load(fp)
            return cls(meta, model)

    def process(self, message: Message, **kwargs: Any) -> None:
        from tfnlu.tagger.extract_entities import extract_entities
        text = message.get(TEXT)
        if text:
            pred = self.model.predict([list(text)], verbose=0)
            entities = extract_entities(pred[0], text)
            ent_data = []
            for ent in entities:
                ent_data.append({
                    "entity": ent[2],
                    "value": ent[3],
                    "start": ent[0],
                    "end": ent[1],
                    "confidence": None
                })
            print(ent_data)
            message.set("entities",
                        message.get(ENTITIES, []) + ent_data,
                        add_to_output=True)

    def persist(self, file_name: Text, model_dir: Text) -> Optional[Dict[Text, Any]]:
        """Persist this model into the passed directory.
        Returns the metadata necessary to load the model again."""

        saved_model_dir = os.path.join(model_dir, self.name)
        shutil.copytree(self.result_dir, saved_model_dir)
        return {'result_dir': self.name}
