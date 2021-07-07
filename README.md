# ZH RASA

## Usage

config.yml

```yaml
language: zh

pipeline:
  - name: zh_rasa.classifiers.TFNLUClassifier
  - name: zh_rasa.extractors.TFNLUExtractor
```

or using bert, please check https://github.com/qhduan/bert-model

```yaml
language: zh

pipeline:
  - name: zh_rasa.classifiers.TFNLUClassifier
    encoder_path: /path/to/bert/
  - name: zh_rasa.extractors.TFNLUExtractor
    encoder_path: /path/to/bert/
```
