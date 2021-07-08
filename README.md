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

使用三层roberta的例子：

```yaml
language: zh
pipeline:
  - name: zh_rasa.TFNLUClassifier
    epochs: 10
    encoder_path: >-
      https://code.aliyun.com/qhduan/bert_part/raw/6c8b798cf7d6d0a12de20c4f90c870df2e107977/roberta_wwm_3.tar.gz
  - name: zh_rasa.TFNLUExtractor
    epochs: 30
    encoder_path: >-
      https://code.aliyun.com/qhduan/bert_part/raw/6c8b798cf7d6d0a12de20c4f90c870df2e107977/roberta_wwm_3.tar.gz
policies:
  - name: MemoizationPolicy
  - name: RulePolicy
  - name: TEDPolicy
    max_history: 5
    epochs: 200
    constrain_similarities: true
    model_confidence: linear_norm
```
