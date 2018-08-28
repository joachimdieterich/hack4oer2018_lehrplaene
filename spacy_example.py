import spacy

# python -m spacy download de_core_news_sm --user
from spacy.lang.de import German

nlp = spacy.load('de_core_news_sm')

tokenizer = German().Defaults.create_tokenizer(nlp)

doc = nlp('Hallo Welt, ich   finde es hier so spannend. der die das. dwwomdowmd 404')
for token in doc:
    if token.text in nlp.vocab:
        print('Im Wörterbuch')
    print(token, token.lemma, token.lemma_)

# for x in tokenizer('Hello world!'):
#     print(x)
