from keras.preprocessing.text import Tokenizer
texts = ['a a a', 'b b', 'c']
tokenizer = Tokenizer(num_words=2)
tokenizer.fit_on_texts(texts)
wd = tokenizer.word_index
tokenizer.texts_to_sequences(texts)
