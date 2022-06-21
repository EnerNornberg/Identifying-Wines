# Identifying-Wines
Python Code to identify different types of wines by the reviews

from google.colab import drive 
drive.mount("/content/drive")

import os
os.getcwd()

import tensorflow as tf

import numpy as np 
import pandas as pd 

import matplotlib.pyplot as plt
import matplotlib.style as style
import seaborn as sns
import sklearn.model_selection as sk

import plotly.express as px

import re

import os
print("Input files:")
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

print()
print("TF Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())
print("GPU is", "available" if tf.config.experimental.list_physical_devices("GPU") else "NOT AVAILABLE")

if tf.test.gpu_device_name():
    print('GPU found')
else:
    print("No GPU found")
    
    
    
    dfWine = pd.read_csv("winemag-data-130k-v2.csv", index_col=0)
    
    dfWine.head(5) # Visualizando os primeiros registros da base
    
    dfWine.shape

print("A base de dados carregada possui {} registros e {} diferentes variáveis.\n".format(dfWine.shape[0], dfWine.shape[1]))

print(dfWine.columns)


# Remove id de usuários do twitters, devido a sua baixa importância para o escopo deste trabalho
dfWine = dfWine.drop(['taster_twitter_handle'], axis=1)
dfWine.head(3)

yearSearch = []    
for value in dfWine['title']:
    regexresult = re.search(r'19\d{2}|20\d{2}', value)
    if regexresult:
        yearSearch.append(regexresult.group())
    else: yearSearch.append(None)
    
    
    # Cria uma nova coluna para os vintages
dfWine['year'] = yearSearch
dfWine.head(3)


# Analisando registros que não retornaram ano a partir do nome
print("Foram extraídos {} valores 'ano' a partir do nome do vinho".format(len(dfWine[dfWine['year'].notna()])))
print("Apenas {} não retornaram nenhum valor referente ao 'ano'.\n".format(len(dfWine[dfWine['year'].isna()].index)))

dfWine['year'].describe()


# Verificando os tipos de vinho listados na variavel 'variety'
print(dfWine['variety'].unique())
print()
print("No total, forma encontrados {} registros únicos em 'variety'.".format(dfWine['variety'].nunique()))


# Obtendo a frequência dos registros de 'variety' em ordem decrescente 
label_freq = dfWine['variety'].apply(lambda s: str(s)).explode().value_counts().sort_values(ascending=False)
label_freq

# Melhorando o entedimento dos dados plotando os valores em um gráfico
style.use("fivethirtyeight")
plt.figure(figsize=(12,10))
sns.barplot(y=label_freq.index.values, x=label_freq, order=label_freq.iloc[:15].index)
plt.title("Grape frequency", fontsize=14)
plt.xlabel("")
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()

# Definindo o Classifier
dfWineClassifier = dfWine[[ 'description', 'year', 'variety', 'province' ]]

# Verificando a exisência de valores nulos
print(dfWineClassifier.isnull().sum())


# Lendo registro com valor nulo para a variavel 'variety'
print(dfWineClassifier[dfWineClassifier['variety'].isna()].head(10))

# Removendo da base valores nulos para variaveis importantes ('description', 'variety')
dfWineClassifier=dfWineClassifier.dropna(subset=['description', 'variety'])

print('Removed ' + str(dfWine.shape[0]-dfWineClassifier.shape[0]) + ' rows with empty values.' + "\n")

# dfWineClassifier = dfWine
RARE_CUTOFF = 700 # Devem existir pelo menos 700 registros para cada tipo de uva, se não, 
                  # classificamos como "Other"

# Criando lista com valores "raros"
rare = list(label_freq[label_freq<RARE_CUTOFF].index)
# print("Iremos ignorar os seguintes valores raros: \n", rare)

# Transformando valores raros em label "Other"
dfWineClassifier['variety'] = dfWineClassifier['variety'].apply(lambda s: str(s) if s not in rare else 'Other')

label_words = list(label_freq[label_freq>=RARE_CUTOFF].index)
label_words.append('Other')
print(label_words)

num_labels = len(label_words)
print("\n"  + str(num_labels) + " tipos de vinho diferentes.")

# pd.DataFrame(dfWineClassifier.variety.unique()).values

# Verificando alguns registros
for i in range(1,5):
    print(dfWineClassifier['variety'].iloc[i])
    print(dfWineClassifier['description'].iloc[i])
    print()
    
    
    # Definindo tamanho do dicionário
NUM_WORDS = 4000

# Definindo tamanho de cada review 256 pq é twitter
SEQ_LEN = 256

# Criando tokenizer para os dados 
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=NUM_WORDS, oov_token='<UNK>')
tokenizer.fit_on_texts(dfWineClassifier['description'])

# Convertendo dados alfanumericos para índices numéricos 
wine_seqs=tokenizer.texts_to_sequences(dfWineClassifier['description'])

# Estabelecendo tamanho padrão para comentário até SEQ_LEN
wine_seqs=tf.keras.preprocessing.sequence.pad_sequences(wine_seqs, maxlen=SEQ_LEN, padding="post")

print(wine_seqs)
  
  # Criando Série com os valores da váriável target
wine_labels=pd.DataFrame({'variety': dfWineClassifier['variety']})

# Subsituindo espaços em branco entre nomes do tipo do vinho
wine_labels=wine_labels.replace(' ', '_', regex=True)

# Criando lista com os nomes de tipo de vinhos
wine_labels_list = []
for item in wine_labels['variety']:
    wine_labels_list.append(str(item))

# Filtrando valores com caracteres especiais e criando Tokenizer
label_tokenizer = tf.keras.preprocessing.text.Tokenizer(split=' ', filters='!"#$%&()*+,./:;<=>?@[\\]^`{|}~\t\n')
label_tokenizer.fit_on_texts(wine_labels_list)

print(len(label_words))
print(label_tokenizer.word_index)

wine_label_seq = np.array(label_tokenizer.texts_to_sequences(wine_labels_list))
wine_label_seq.shape
  
  
  
  reverse_word_index = dict([(value, key) for (key, value) in tokenizer.word_index.items()])

def decode_article(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])


reverse_label_index = dict([(value, key) for (key, value) in label_tokenizer.word_index.items()])

def decode_label(text):
    return ' '.join([reverse_label_index.get(i, '?') for i in text])
  
  
  # Verificando formato de input

test_entry=3

print(decode_article(wine_seqs[test_entry]))
print('---')
print(wine_seqs[test_entry])

print(decode_label(wine_label_seq[test_entry]))
print('---')
print(wine_label_seq[test_entry])
  
  
  # Treino-teste-divisão
X_train, X_test, y_train, y_test = sk.train_test_split(wine_seqs,
                                                    wine_label_seq,
                                                    test_size=0.20,
                                                    random_state=42)

print('Test: ' + str(len(X_test)) + ' Train: ' + str(len(X_train)))

X_train = X_train.astype("float32")
X_test = X_test.astype("float32")
y_train = y_train.astype("float32")
y_test = y_test.astype("float32")

print(type(X_train), X_train.shape)

print(X_train.shape)
print(y_train.shape)
  
  EMBEDDING_SIZE = 256 #neurons
EMBEDDING_SIZE_2 = 64
EMBEDDING_SIZE_3 = (num_labels+1)
BATCH_SIZE = 512  
EPOCHS = 10
LR = 1e-5 


model = tf.keras.Sequential([
    
    # Adicione uma camada de incorporação, esperando o vocabulário de entrada de um determinado tamanho e a 
    #dimensão de incorporação de saída do tamanho ajustado que definimos na parte superior
    tf.keras.layers.Embedding(NUM_WORDS, EMBEDDING_SIZE),
    
    tf.keras.layers.Conv1D(128, 5, activation='relu'), 
    tf.keras.layers.GlobalMaxPooling1D(), 
    
    # use ReLU no lugar da função tanh, pois são alternativas muito boas uma da outra
    tf.keras.layers.Dense(EMBEDDING_SIZE_2, activation='relu'),
    
    # Adicione uma camada densa com unidades adicionais e ativação softmax
    # Quando temos várias saídas, o softmax converte as camadas das saídas em uma distribuição de probabilidade
    tf.keras.layers.Dense(EMBEDDING_SIZE_3, activation='softmax')
])

model.summary()

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
  
  
  # Diretorio onde os checkpoints serao salvos
checkpoint_dir = './checkpoints/classifer_training_checkpoints'
# Nome do checkpoint do arquivo
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    monitor='accuracy',
    save_best_only=True,
    mode='auto',
    save_weights_only=True)

  
  history = model.fit(X_train, y_train,
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS,
                    validation_data=(X_test, y_test),
                    validation_steps=30,
                   callbacks=[checkpoint_callback])

loss, accuracy = model.evaluate(X_test, y_test)

print("Loss: ", loss)
print("Accuracy: ", accuracy)
  
  
  tf.train.latest_checkpoint(checkpoint_dir)

model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

model.build(tf.TensorShape([1, None]))

model.summary()
  
  history_dict = history.history
history_dict.keys()

acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

# "bo" é para "ponto azul"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b é para "linha azul sólida"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()
  
  
  plt.clf()   # Limpar Figura

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')

plt.show()
  
  
  def sample_predict(sample_pred_text, pad):
  encoded_sample_pred_text = tokenizer.texts_to_sequences(sample_pred_text)
  print(encoded_sample_pred_text)
  print(type(encoded_sample_pred_text))

  if pad:
    encoded_sample_pred_text = pad_to_size(encoded_sample_pred_text, SEQ_LEN)
    
  encoded_sample_pred_text = np.array(encoded_sample_pred_text)
  encoded_sample_pred_text = encoded_sample_pred_text.astype("float32")
  predictions = model.predict(encoded_sample_pred_text)
  
  
  new_review = ['Crisp grapefruit and grassy lemon.']
encoded_sample_pred_text = tokenizer.texts_to_sequences(new_review)
# Alguns modelos precisam de suavizaçao, alguns não - depende da linha embedada.
encoded_sample_pred_text = tf.keras.preprocessing.sequence.pad_sequences(encoded_sample_pred_text, maxlen=SEQ_LEN, padding="post")
predictions = model.predict(encoded_sample_pred_text)

for n in reversed((np.argsort(predictions))[0]):
    predicted_id = [n]
    print("Guess: %s \n Probability: %f" %(decode_label(predicted_id).replace('_', ' '), 100*predictions[0][predicted_id][0]) + '%')
  
  
  
