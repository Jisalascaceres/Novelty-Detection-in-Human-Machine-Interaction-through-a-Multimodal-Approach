
import os
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_data(paths):
    
    
    csv = pd.read_csv(paths[0])
    
    embeddings = np.load(paths[1])
    
    return csv,embeddings

def reshape_paths (paths):
    a = paths.values.tolist()
    a = np.array(a)
    a = a.reshape(1,-1)
    return a[0]
    
def reshape_embed (embeddings):
    shape = embeddings.shape
    embeddings = embeddings.reshape(shape[0],shape[1],shape[3])
    return embeddings
    
def load_pair(name,carpeta,set):
    paths_csv = 'Dataset_'+ set +  '/50_50/' + carpeta + '/' + name + '.csv'
    paths_embed = 'Embeddings_'+ set + '/50_50/' + carpeta + '/' + name + '.npy'
    
    csv,embeddings = load_data([paths_csv,paths_embed])
    
    return csv,embeddings

def load_train(set): 
    Img_Con_train_paths,Img_Con_train_embeddings = load_pair('Img_Con_Train','Img',set)
    Aud_Con_train_paths,Aud_Con_train_embeddings = load_pair('Aud_Con_Train','Audios',set)
    Mix_Con_train_paths,Mix_Con_train_embeddings = load_pair('Mix_Con_Train','Mix',set)
    
# Althouth we load this for training, we are not gonna use it, is here for consistency and future use in the Open Set problem.
    Img_Descon_train_paths,Img_Descon_train_embeddings = load_pair('Img_Descon_Train','Img',set) #
    Aud_Descon_train_paths,Aud_Descon_train_embeddings = load_pair('Aud_Descon_Train','Audios',set) #
    Mix_Descon_train_paths,Mix_Descon_train_embeddings = load_pair('Mix_Descon_Train','Mix',set) #
        
        
    Paths = [[Img_Con_train_paths,Aud_Con_train_paths,Mix_Con_train_paths],[Img_Descon_train_paths,Aud_Descon_train_paths,Mix_Descon_train_paths]]
    Embeddings = [[Img_Con_train_embeddings,Aud_Con_train_embeddings,Mix_Con_train_embeddings],[Img_Descon_train_embeddings,Aud_Descon_train_embeddings,Mix_Descon_train_embeddings]]
        
    return Paths,Embeddings

def load_validation(set):
    Img_Con_validation_paths,Img_Con_validation_embeddings = load_pair('Img_Con_Validation','Img',set)
    Aud_Con_validation_paths,Aud_Con_validation_embeddings = load_pair('Aud_Con_Validation','Audios',set)
    Mix_Con_validation_paths,Mix_Con_validation_embeddings = load_pair('Mix_Con_Validation','Mix',set)
    
    Img_Descon_validation_paths,Img_Descon_validation_embeddings = load_pair('Img_Descon_Validation','Img',set)
    Aud_Descon_validation_paths,Aud_Descon_validation_embeddings = load_pair('Aud_Descon_Validation','Audios',set)
    Mix_Descon_validation_paths,Mix_Descon_validation_embeddings = load_pair('Mix_Descon_Validation','Mix',set)
    
    Paths = [[Img_Con_validation_paths,Aud_Con_validation_paths,Mix_Con_validation_paths],[Img_Descon_validation_paths,Aud_Descon_validation_paths,Mix_Descon_validation_paths]]
    Embeddings = [[Img_Con_validation_embeddings,Aud_Con_validation_embeddings,Mix_Con_validation_embeddings],[Img_Descon_validation_embeddings,Aud_Descon_validation_embeddings,Mix_Descon_validation_embeddings]]
    
    return Paths,Embeddings

def load_test(set):
    Img_Con_test_paths,Img_Con_test_embeddings = load_pair('Img_Con_Test','Img',set)
    Aud_Con_test_paths,Aud_Con_test_embeddings = load_pair('Aud_Con_Test','Audios',set)
    Mix_Con_test_paths,Mix_Con_test_embeddings = load_pair('Mix_Con_Test','Mix',set)
    
    Img_Descon_test_paths,Img_Descon_test_embeddings = load_pair('Img_Descon_Test','Img',set)
    Aud_Descon_test_paths,Aud_Descon_test_embeddings = load_pair('Aud_Descon_Test','Audios',set)
    Mix_Descon_test_paths,Mix_Descon_test_embeddings = load_pair('Mix_Descon_Test','Mix',set)
    
    Paths = [[Img_Con_test_paths,Aud_Con_test_paths,Mix_Con_test_paths],[Img_Descon_test_paths,Aud_Descon_test_paths,Mix_Descon_test_paths]]
    Embeddings = [[Img_Con_test_embeddings,Aud_Con_test_embeddings,Mix_Con_test_embeddings],[Img_Descon_test_embeddings,Aud_Descon_test_embeddings,Mix_Descon_test_embeddings]]
    
    return Paths,Embeddings

def save_ga (ga_instance,filename):    
    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    print("Parameters of the best solution : {solution}".format(solution=solution))
    print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
    print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))
    ga_instance.save(filename = filename)
    
    
    
    
def destringfy(pair):
    aux = pair.replace('[','').replace(']','').replace("'",'').replace(' ','').split('\n')
    return aux 




def test_conocidos(dataset_con_train,predictor):
    reconocidas = 0
    reconocidas_persona_incorrecta = 0
    no_reconocidas = 0

    # We are going to test the model with the dataset_con_train
    


    for i in range(len(dataset_con_train)):
        [resp,id,etiquetas] = predictor(dataset_con_train[i])
        if resp == True:
            reconocidas += 1
            person = dataset_con_train[i][0]
            if id not in person:
                print("Lo reconocio como: " + id + " cuando era: " + person)
                reconocidas_persona_incorrecta += 1
        if resp == False:
            no_reconocidas += 1
            

    acierto_conocidos = reconocidas/len(dataset_con_train)
    conocidas_incorrectas = reconocidas_persona_incorrecta/len(dataset_con_train)
    
    
    return [acierto_conocidos,conocidas_incorrectas,no_reconocidas]


def test_desconocidos(dataset_descon_train,predictor):
    reconocidas = 0
    no_reconocidas = 0

    for i in range(len(dataset_descon_train)):
        [resp,id,distancias] = predictor(dataset_descon_train[i])
        if resp == True:
            reconocidas += 1
        if resp == False:
            no_reconocidas += 1
        
        
        


    acierto_desconocidos = no_reconocidas/len(dataset_descon_train)

    return acierto_desconocidos


def calculate_metrics (TP,TN,FP,FN): 
    # We calculate the 4 basic metrics:
    # Precision = TP/(TP+FP)
    # Recall = TP/(TP+FN)
    # F1 = 2*Precision*Recall/(Precision+Recall)
    # Accuracy = (TP+TN)/(TP+TN+FP+FN)
    
   
    Acurracy = (TP+TN)/(TP+TN+FP+FN)
    Recall = TP/(TP+FN)
    
    
    if TP+FP == 0:
        Precision = 0
        F1 = 0    
        
    elif TP+FN == 0:
        Recall = 0
        F1 = 0
    else:    
        Precision = TP/(TP+FP)

        F1 = 2*Precision*Recall/(Precision+Recall)
        
  
    
    #print  ("Precision: {} , Recall: {} , F1: {} , Acurracy: {}".format(Precision,Recall,F1,Acurracy))
    
    return Precision,Recall,F1,Acurracy

def plot_confussion(TP,TN,FP,FN,title): 
    
    plt.figure(figsize=(10,8))
    sns.heatmap([[TP,FP],[FN,TN]],annot=True, fmt='d', cmap='crest', annot_kws={'size':20}, linewidths=0.5, linecolor='black', square=True, xticklabels=['0', '1'], yticklabels=['0', '1'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.show()


