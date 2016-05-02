# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
import random as rd
import pip

def criarTuplas(path,token_indice={},val=True,mult=False):
    print('Criando tuplas de -> ' + path)
    arq = open(path)
    linhas = arq.readlines()
    tuplas = []
    for linha in linhas:
        vec = linha.split()
        if len(vec) >= 3:
            prox = 1
            if mult: prox =  int(vec[2])
            for i in range(0, prox):
                tuplas.append((token_indice.get(vec[0],0),
                               token_indice.get(vec[1],0)))
    d_val = len(tuplas)/5
    rest = len(tuplas) - d_val
    if val:
        return tuplas[:rest], tuplas[-d_val:]
    return tuplas

def embaralhar(tuplas, qtd):
    for i in xrange(0,qtd):        
        p1 = rd.randint(0,len(tuplas)-1)
        p2 = rd.randint(0,len(tuplas)-1)
        #print("p1 = " + str(p1))
        #print("p2 = " + str(p2))
        aux = tuplas[p1]
        tuplas[p1] = tuplas[p2]
        tuplas[p2] = aux
    
#dado um número, gerar o seu one-hot-vector
def build_ohv(a,tam_vocab):
    onehot = [0]*tam_vocab
    if ( a > tam_vocab): onehot[0] = 1; return onehot
    onehot[a-1] = 1
    return onehot

def mapWordInt(path):
    arq = open(path)
    linhas = arq.readlines()
    vocab = []
    for linha in linhas:
        vocab.append(linha.split()[0])
        #print('palavra: '+linha.split()[0])
    token_indice = {}
    indice_token = {}
    for i in xrange(0,len(vocab)):
        token_indice[vocab[i]] = i
        indice_token[i] = vocab[i]
    return token_indice, indice_token

def loadWordEmbeddings(path='dados/WordEmbeddings/pathtosave22.txt',token_indice={}):
    arq = open(path)
    linhas = arq.readlines()
    word_list = {}
    for linha in linhas:
        targ = []
        tokens = linha.split(' ')
        for i in xrange(1,101):
            targ.append(float(tokens[i]))
        word_list[token_indice.get(tokens[0],0)] = targ
    array_aux = []
    keys = word_list.keys()
    for i in xrange(0,len(keys)):
        array_aux.append(word_list[keys[i]])
    return np.array(array_aux),array_aux

def loadTuplas(path,val=True):
    print('Criando tuplas de -> ' + path)
    arq = open(path)
    linhas = arq.readlines()
    tuplas = []
    for linha in linhas:
        vec = linha.split()
        if len(vec) >= 2:
            tuplas.append((vec[0],vec[1]))
    d_val = len(tuplas)/5
    rest = len(tuplas) - d_val
    if val:
        return tuplas[:rest], tuplas[-d_val:]
    return tuplas

def build_blocks(tuplas_train,TAM_BLOCKS):
    vec_train = []
    cont_BLOCK = 0
    trains = []
    for i in range(0,len(tuplas_train)):
        if (cont_BLOCK < TAM_BLOCKS):
            vec_train.append(tuplas_train[i])
            cont_BLOCK += 1
        elif(cont_BLOCK == TAM_BLOCKS):
            cont_BLOCK = 0
            trains.append(vec_train)
            vec_train = []
    return trains

def convert(tuplasTrain,embeddings,VOCAB,returnNumpy=True):
    matrixTrain = []
    matrixTarget = []
    for tupla in tuplasTrain:
        try:
            matrixTrain.append(embeddings[tupla[0]])
            matrixTarget.append(build_ohv(tupla[1],VOCAB))
        except:            
            pass
    if returnNumpy:
        return np.array(matrixTrain), np.array(matrixTarget)
    else:
        return matrixTrain, matrixTarget

def convert2(tuplasTrain, embeddings):
    matrixTrain = []
    target = []
    cont = 0
    for tupla in tuplasTrain:
        try:
            matrixTrain.append(embeddings[tupla[0]])
            target.append(tupla[1])
        except:
            cont +=1
    print('convert2 - oo tuplas:' + str(cont))
    return np.array(matrixTrain), np.array(target)

def send_email(user='flavioarthurufs@gmail.com', pwd='ogmufkgofsiplcwj', recipient='flavioarthurufs@gmail.com',
               subject='resultados', body='python: nunca foi tão fácil'):
    import smtplib
    print(str(body))

    gmail_user = user
    gmail_pwd = pwd
    FROM = user
    TO = recipient if type(recipient) is list else [recipient]
    SUBJECT = subject
    TEXT = body

    message = """\From: %s\nTo: %s\nSubject: %s\n\n%s
    """ % (FROM, ", ".join(TO), SUBJECT, TEXT)
    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.ehlo()
        server.starttls()
        server.login(gmail_user, gmail_pwd)
        server.sendmail(FROM, TO, message)
        server.close()
        print('successfully sent the mail')
    except:
        print("failed to send mail")
    

def salvarResultados(nome,resultado):
    dados = open("dados/resultados/"+nome,"a")
    dados.writelines(str(resultado))
    dados.close()

def installKeras():
    pip.main(['install', 'keras==0.3.2'])
