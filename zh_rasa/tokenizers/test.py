from tokenizers import BertWordPieceTokenizer

tokenizer = BertWordPieceTokenizer('vocab.txt')


ret = tokenizer.encode('我爱important北京天安门', add_special_tokens=False)
print(ret.tokens)
print(ret.offsets)

