# ZH RASA

## Usage

config.yml

```yaml
language: zh

pipeline:
  - name: zh_rasa.TFNLUClassifier
  - name: zh_rasa.TFNLUExtractor
```

or using bert, please check https://github.com/qhduan/bert-model

```yaml
language: zh

pipeline:
  - name: zh_rasa.TFNLUClassifier
    encoder_path: /path/to/bert/
  - name: zh_rasa.TFNLUExtractor
    encoder_path: /path/to/bert/
```
