from natasha import (
    Segmenter,
    NewsEmbedding,
    NewsNERTagger,
    Doc
)
from examples import example_1, example_2, example_3, example_4

# 1. Инициализация инструментов
segmenter = Segmenter()
emb = NewsEmbedding()
ner_tagger = NewsNERTagger(emb)

# 2. Текст для анализа
text = example_1
examples = [example_1, example_2, example_3, example_4]
for i in examples:
    print(i)
    doc = Doc(i)
    doc.segment(segmenter)
    doc.tag_ner(ner_tagger)
    print(doc.spans)
