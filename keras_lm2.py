# coding: utf-8d
from keras.layers import *
from keras.models import *
from keras.optimizers import *
from auxiliar_keras import *
import time
import auxiliar_keras as aux
import math
from KerasLayer.FixedEmbedding import FixedEmbedding
import theano

theano.config.floatX = 'float32'
# Number of units in the hidden (recurrent) layer
N_HIDDEN = 450
# Optimization learning rate
LEARNING_RATE = 1.0
# All gradients above this will be clipped
GRAD_CLIP = 10.0
TAM_WE = 100
TAM_VOCAB = 10
EPOCHS = 15
TAM_BATCH = 100
VALIDATION = False

#####----------AUXILIAR---------##########
dadosP = 'dados/Unigram/'
vocabs = 'dados/Pessoais/vocab_'
unigrams = 'dados/Unigram/'

trainss = ['Flaviobase_clean','todos_floresta_clean','web_crawler_web_clean','web_crawler_twitter_clean']

testes = ['Juniorbase_clean']

tipo = '.txt'
############################################

def calc_perplexity(model,test_in,test_out,TAM_VOCAB_OUT,validation=False):
    result = {}
    if validation: result['tipo'] = 'validacao'
    else: result['tipo'] = 'teste'
    print(test_in.shape)
    print(test_out.shape)
    out = model.predict(test_in)
    total = 1.0; entropy1 = 0.0; entropy2 = 0.0; cont=0
    for i in xrange(0,len(out)):
        a = test_out[i]
        total +=1.0
        if (a-1 >= TAM_VOCAB_OUT): px = out[i][0]; cont+=1
        else: px = out[i][a-1]
        log = math.log(px,2); entropy1 += -1.0*log; entropy2 += -1.0*log*px
    result['Entropy1'] = entropy1/total
    result['Entropy2'] = entropy2
    result['Perplexidade'] = math.pow(2,entropy1/total)
    return result

def main():
    for trein in trainss:
        for test in testes:
            print("Carregando bases de dados")
            baseTeste = dadosP + test + test + tipo
            vocabPath = vocabs + test + tipo
            baseTrain = unigrams + trein + test + tipo
            token_indice, indice_token = aux.mapWordInt(vocabPath)
            embeddingsNP, embeddings = aux.loadWordEmbeddings(token_indice=token_indice)
            embeddingsNP = embeddingsNP.astype(np.float32)
            print("Shape do embedding" + str(embeddingsNP.shape))

            if VALIDATION:
                tuplas_train, tuplas_val = criarTuplas(baseTrain,token_indice,True)
            else:
                tuplas_train = criarTuplas(baseTrain,token_indice,False,mult=True)

            print("Quantidade de tuplas de treinamento: " + str(len(tuplas_train)))
            tuplas_test = criarTuplas(baseTeste,token_indice,val=False)

            TAM_VOCAB_OUT = len(token_indice.keys())

            print("LSTM...")
            model = Sequential()
            model.add(FixedEmbedding(output_dim=embeddingsNP.shape[1],input_dim=embeddingsNP.shape[0],input_length=1, weights=[embeddingsNP]))
            model.add(LSTM(output_dim=N_HIDDEN,init='glorot_uniform',activation='tanh', batch_input_shape=(None,1,embeddingsNP.shape[1]), return_sequences=True))
            model.add(Dropout(0.5))
            model.add(LSTM(output_dim=N_HIDDEN,init='glorot_uniform',activation='tanh', batch_input_shape=(None,1,embeddingsNP.shape[1])))
            model.add(Dropout(0.5))
            model.add(Dense(output_dim=TAM_VOCAB_OUT, activation='softmax'))
            
            model.compile(loss='categorical_crossentropy', optimizer='adagrad')

            print("Convertendo tuplas de treinamento")
            embaralhar(tuplas_train,len(tuplas_train)*2)
            blocks = aux.build_blocks(tuplas_train,5000)
            test_in, test_out = aux.convert2(tuplas_test,embeddings)
            print('test_in e test_out')
            print(test_in.shape)
            print(test_out.shape)
            if VALIDATION: val_in, val_out = aux.convert2(tuplas_val,embeddings)

            print("Treinamento...")
            for epoca in range(0,EPOCHS):
                t0 = time.time()
                print("Total de blocks: " + str(len(blocks)))
                for i in xrange(0,len(blocks)):
                    if i % 1000 == 0: print('block i=' + str(i))
                    entrada, target = aux.convert(blocks[i], embeddings, TAM_VOCAB_OUT)
                    model.fit(entrada,target,nb_epoch=1,batch_size=TAM_BATCH, verbose=False, shuffle=True,show_accuracy=True)
                result_final = {}
                if VALIDATION:
                    result_val = calc_perplexity(model,val_in,val_out,TAM_VOCAB_OUT,True)
                    result_final['validacao'] = result_val
                t1 = time.time()
                result_test = calc_perplexity(model,test_in,test_out,TAM_VOCAB_OUT,False)
                result_final['train'] = trein
                result_final['teste'] = result_test
                result_final['tempo'] = str(t1 - t0)
                print(result_final)
                aux.send_email(body=result_final)
                salvarResultados(trein + test + tipo,result_final)
if __name__ == '__main__':
    main()
