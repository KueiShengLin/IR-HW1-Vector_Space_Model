
# coding: utf-8

# In[1]:


import os
import math
import copy

# In[2]:


def readDoc_creatDict(Dirs):

    Doc_T = []    #檔案
    Dict_T = []   #字典檔
    
    #生成檔案 list : Doc_T 每個元素為一個 Doc文本
    for i in range(len(Dirs)):
        
        f_name = "Document\\" + str(Dirs[i])
        f = open(f_name)
        Doc = []
        Doc.append(f.read())
        
        '''
        #將Doc拆成line
        line = '0'
        while(line != ""):
            new_line = []
            line = f.readline()
            new_line.append(line)
            Doc.append(new_line)
        '''
        
        Doc_T.append(Doc)
        f.close()
    
    #生成字典檔 list : Dict_T 每個元素為一個 Doc的 Dict (tf)
    T = ''
    
    for j in range(0,len(Doc_T)):    
        
        x = str(Doc_T[j])
        Dict = {}
        ignor = 1        
        
        for k in range(2,len(x)-1):
            if(x[k]!=' ' and x[k]+x[k+1] != "\\n" ):
                if(x[k] !='n'):
                    T += x[k]
            else:
                if(ignor >5): #忽略前3行
                    if(not(T in Dict)):
                        Dict[T] =1
                    else:
                        Dict[T] +=1                    
                T = ''
                ignor += 1  
                
        del Dict["-1"]
        Dict_T.append(Dict)
                   
    return Dict_T
               
    


# In[3]:


def readQuery_creatDict(QDirs):
    
    Q_T = []
    QDict_T = []
    for i in range(len(QDirs)):
        
        f_name = "Query\\" + str(QDirs[i])
        f = open(f_name)
        Q = []
        Q.append(f.read())

        Q_T.append(Q)
        f.close()

    #生成字典檔 list : QDict_T 每個元素為一個 Query的 QDict (tf)
    T = ''    
    for j in range(0,len(Q_T)):    
        
        x = str(Q_T[j])
        QDict = {}

        for k in range(2,len(x)-1):
            if(x[k]!=' ' and x[k]+x[k+1] != "\\n" ):
                if(x[k] !='n'):
                    T += x[k]
            else:
                if(not(T in QDict)):
                        QDict[T] =1
                else:
                        QDict[T] +=1  
                T = ''
                
        del QDict["-1"]
        QDict_T.append(QDict)

    return QDict_T

#readQuery_creatDict(os.listdir("Query"))


# In[4]:


#1 Doc : tf = Raw freq idf = Inverse Frequency, Query : same as Doc
def tf_idf_RFIF_S(Dirs,Doc_freq,Dict_T,QDict_T):
    
    N = len(Dirs)

    Doc_idf = Doc_freq.copy()
    x = list(Doc_idf.keys())
    for i in range(0,len(Doc_idf)-1):
        Doc_idf[x[i]] = (math.log10(N/Doc_idf[x[i]])) #idf compute

    Doc_tfidf = copy.deepcopy(Dict_T)

    for j in range(0,len(Doc_tfidf)-1):
        x = list(Dict_T[j].keys())
        for i in range(0,len(x)-1):
            #u can add the tf compute on this line 
            Doc_tfidf[j][x[i]] = Doc_tfidf[j][x[i]]*Doc_idf[x[i]] #tf*idf
            
    Q_tfidf = copy.deepcopy(QDict_T)
    for j in range(0,len(Q_tfidf)-1):
        x = list(QDict_T[j].keys())
        for i in range(0,len(x)-1):
            #u can add the tf compute on this line
            if(x[i] in Doc_idf):
                Q_tfidf[j][x[i]] = QDict_T[j][x[i]]*Doc_idf[x[i]] #tf*idf
            else:
                Q_tfidf[j][x[i]] = 0
                
    return(Doc_tfidf,Q_tfidf)


#2 Doc : tf = Raw freq idf = Inverse Frequency, Query : Double Normalization 0.5
def tf_idf_RFIF_DN05(Dirs,Doc_freq,Dict_T,QDict_T):
    
    N = len(Dirs)

    Doc_idf = Doc_freq.copy()
    x = list(Doc_idf.keys())
    for i in range(0,len(Doc_idf)-1):
        Doc_idf[x[i]] = (math.log10(N/Doc_idf[x[i]])) #idf compute

    Doc_tfidf = copy.deepcopy(Dict_T)
    for j in range(0,len(Doc_tfidf)-1):
        x = list(Dict_T[j].keys())
        for i in range(0,len(x)-1):
            #u can add the tf compute on this line 
            Doc_tfidf[j][x[i]] = Dict_T[j][x[i]]*Doc_idf[x[i]] #tf*idf
    
    Q_tfidf = copy.deepcopy(QDict_T)
    
    for j in range(0,len(Q_tfidf)-1): 
        
        x = list(QDict_T[j].keys())
        Max = max(QDict_T[j].values()) 

        for i in range(0,len(x)-1):
            #u can add the tf compute on there
            Q_tfidf[j][x[i]] = 0.5 + (0.5*(QDict_T[j][x[i]]/Max)) #Double Normalization 0.5
            #====================================
            if(x[i] in Doc_idf):
                Q_tfidf[j][x[i]] = Q_tfidf[j][x[i]]*Doc_idf[x[i]] #tf*idf
            else:
                Q_tfidf[j][x[i]] = 0
                            
    return(Doc_tfidf,Q_tfidf)

#3 Doc : tf = log Normalization idf = Inverse Frequency, Query : Double Normalization 0.5
def tf_idf_LNIF_DN05(Dirs,Doc_freq,Dict_T,QDict_T):
    
    N = len(Dirs)
    
    
    Doc_idf = Doc_freq.copy()
    x = list(Doc_idf.keys())
    for i in range(0,len(Doc_idf)-1):
        Doc_idf[x[i]] = (math.log10(N/Doc_idf[x[i]])) #idf compute

    Doc_tfidf = copy.deepcopy(Dict_T)
    for j in range(0,len(Doc_tfidf)-1):
        x = list(Dict_T[j].keys())
        for i in range(0,len(x)-1):
            #u can add the tf compute on this line
            y = Dict_T[j][x[i]]
            Doc_tfidf[j][x[i]] = 1+(math.log(y,2))
            #=====================================
            Doc_tfidf[j][x[i]] = Dict_T[j][x[i]]*Doc_idf[x[i]] #tf*idf

    Q_tfidf = copy.deepcopy(QDict_T)
    
    for j in range(0,len(Q_tfidf)-1): 
        
        x = list(QDict_T[j].keys())
        Max = max(QDict_T[j].values()) 

        for i in range(0,len(x)-1):
            #u can add the tf compute on there
            Q_tfidf[j][x[i]] = 0.5 + (0.5*(QDict_T[j][x[i]]/Max)) #Double Normalization 0.5
            #====================================
            if(x[i] in Doc_idf):
                Q_tfidf[j][x[i]] = Q_tfidf[j][x[i]]*Doc_idf[x[i]] #tf*idf
            else:
                Q_tfidf[j][x[i]] = 0
                            
    return(Doc_tfidf,Q_tfidf)


#4 Doc : tf = log Normalization idf = Inverse Frequency, Query : Raw freq
def tf_idf_LNIF_RF(Dirs,Doc_freq,Dict_T,QDict_T):
    
    N = len(Dirs)
        
    Doc_idf = Doc_freq.copy()
    x = list(Doc_idf.keys())
    for i in range(0,len(Doc_idf)-1):
        Doc_idf[x[i]] = (math.log10(N/Doc_idf[x[i]])) #idf compute

    Doc_tfidf = copy.deepcopy(Dict_T)
    for j in range(0,len(Doc_tfidf)-1):
        x = list(Dict_T[j].keys())
        for i in range(0,len(x)-1):
            #u can add the tf compute on this line
            y = Dict_T[j][x[i]]
            Doc_tfidf[j][x[i]] = (1+(math.log(y,2)))
            #=====================================
            Doc_tfidf[j][x[i]] = Doc_tfidf[j][x[i]]*Doc_idf[x[i]] #tf*idf

    Q_tfidf = copy.deepcopy(QDict_T)
    
    for j in range(0,len(Q_tfidf)-1): 
        
        x = list(QDict_T[j].keys())
        Max = max(QDict_T[j].values()) 

        for i in range(0,len(x)-1):
            #u can add the tf compute there
            #====================================
            if(x[i] in Doc_idf):
                Q_tfidf[j][x[i]] = QDict_T[j][x[i]]*Doc_idf[x[i]] #tf*idf
            else:
                Q_tfidf[j][x[i]] = 0
                            
    return(Doc_tfidf,Q_tfidf)


#5 Doc : tf = Double Normalization 0.5 idf = Inverse Frequency, Query : Double Normalization 0.5
def tf_idf_DN05IF_DN05(Dirs,Doc_freq,Dict_T,QDict_T):
    
    N = len(Dirs)
    
    
    Doc_idf = Doc_freq.copy()
    x = list(Doc_idf.keys())
    for i in range(0,len(Doc_idf)-1):
        Doc_idf[x[i]] = (math.log10(N/Doc_idf[x[i]])) #idf compute

    Doc_tfidf = copy.deepcopy(Dict_T)
    for j in range(0,len(Doc_tfidf)-1):
        
        x = list(Dict_T[j].keys())
        Max = max(Dict_T[j].values())
        
        for i in range(0,len(x)-1):
            #u can add the tf compute on this line
            Doc_tfidf[j][x[i]] = 0.5 + (0.5*(Dict_T[j][x[i]]/Max)) #Double Normalization 0.5
            #=====================================
            Doc_tfidf[j][x[i]] = Doc_tfidf[j][x[i]]*Doc_idf[x[i]] #tf*idf

    Q_tfidf = copy.deepcopy(QDict_T)
    
    for j in range(0,len(Q_tfidf)-1): 
        
        x = list(QDict_T[j].keys())
        Max = max(QDict_T[j].values()) 

        for i in range(0,len(x)-1):
            #u can add the tf compute on there
            Q_tfidf[j][x[i]] = 0.5 + (0.5*(QDict_T[j][x[i]]/Max)) #Double Normalization 0.5
            #====================================
            if(x[i] in Doc_idf):
                Q_tfidf[j][x[i]] = Q_tfidf[j][x[i]]*Doc_idf[x[i]] #tf*idf
            else:
                Q_tfidf[j][x[i]] = 0
                            
    return(Doc_tfidf,Q_tfidf)

#6 Doc : tf = log Normalization idf = Inverse Frequency Smooth, Query : Double Normalization 0.5
def tf_idf_LNIFS_DN05(Dirs,Doc_freq,Dict_T,QDict_T):
    
    N = len(Dirs)
    
    Doc_idf = Doc_freq.copy()
    x = list(Doc_idf.keys())
    for i in range(0,len(Doc_idf)-1):
        Doc_idf[x[i]] = (math.log10(1+(N/Doc_idf[x[i]]))) #idf compute

    Doc_tfidf = copy.deepcopy(Dict_T)
    for j in range(0,len(Doc_tfidf)-1):
        x = list(Dict_T[j].keys())
        for i in range(0,len(x)-1):
            #u can add the tf compute on this line
            y = Dict_T[j][x[i]]
            Doc_tfidf[j][x[i]] = 1+(math.log(y,2))
            #=====================================
            Doc_tfidf[j][x[i]] = Doc_tfidf[j][x[i]]*Doc_idf[x[i]] #tf*idf

    Q_tfidf = copy.deepcopy(QDict_T)
    
    for j in range(0,len(Q_tfidf)-1): 
        
        x = list(QDict_T[j].keys())
        Max = max(QDict_T[j].values()) 

        for i in range(0,len(x)-1):
            #u can add the tf compute on there
            Q_tfidf[j][x[i]] = 0.5 + (0.5*(QDict_T[j][x[i]]/Max)) #Double Normalization 0.5
            #====================================
            if(x[i] in Doc_idf):
                Q_tfidf[j][x[i]] = Q_tfidf[j][x[i]]*Doc_idf[x[i]] #tf*idf
            else:
                Q_tfidf[j][x[i]] = 0
                            
    return(Doc_tfidf,Q_tfidf)


#7 Doc : tf = Double Normalization 0.5 idf = Inverse Frequency Smooth, Query : Double Normalization 0.5
def tf_idf_DN05IFS_DN05(Dirs,Doc_freq,Dict_T,QDict_T):
    
    N = len(Dirs)
    
    
    Doc_idf = Doc_freq.copy()
    x = list(Doc_idf.keys())
    for i in range(0,len(Doc_idf)-1):
        Doc_idf[x[i]] = (math.log10(1+(N/Doc_idf[x[i]]))) #idf compute

    Doc_tfidf = copy.deepcopy(Dict_T)
    for j in range(0,len(Doc_tfidf)-1):
        
        x = list(Dict_T[j].keys())
        Max = max(Dict_T[j].values())
        
        for i in range(0,len(x)-1):
            #u can add the tf compute on this line
            Doc_tfidf[j][x[i]] = 0.5 + (0.5*(Dict_T[j][x[i]]/Max)) #Double Normalization 0.5
            #=====================================
            Doc_tfidf[j][x[i]] = Doc_tfidf[j][x[i]]*Doc_idf[x[i]] #tf*idf

    Q_tfidf = copy.deepcopy(QDict_T)
    
    for j in range(0,len(Q_tfidf)-1): 
        
        x = list(QDict_T[j].keys())
        Max = max(QDict_T[j].values()) 

        for i in range(0,len(x)-1):
            #u can add the tf compute on there
            Q_tfidf[j][x[i]] = 0.5 + (0.5*(QDict_T[j][x[i]]/Max)) #Double Normalization 0.5
            #====================================
            if(x[i] in Doc_idf):
                Q_tfidf[j][x[i]] = Q_tfidf[j][x[i]]*Doc_idf[x[i]] #tf*idf
            else:
                Q_tfidf[j][x[i]] = 0
                            
    return(Doc_tfidf,Q_tfidf)

#8 Doc : tf = log Normalization idf = Unary, Query : Raw freq ,Qidf = Inverse Frequency
def tf_idf_LNUN_RFIF(Dirs,Doc_freq,Dict_T,QDict_T):
    
    N = len(Dirs)
    
    Query_idf = Doc_freq
    Doc_idf = Doc_freq.copy()
    x = list(Doc_freq.keys())
    for i in range(0,len(Doc_idf)-1):
        Query_idf[x[i]] = (math.log10(N/Doc_idf[x[i]])) #idf compute
        Doc_idf[x[i]] = 1 #idf compute
    
    Doc_tfidf = copy.deepcopy(Dict_T)
    for j in range(0,len(Doc_tfidf)-1):
        x = list(Dict_T[j].keys())
        for i in range(0,len(x)-1):
            #u can add the tf compute on this line
            y = Dict_T[j][x[i]]
            Doc_tfidf[j][x[i]] = (1+(math.log(y,2)))
            #=====================================
            Doc_tfidf[j][x[i]] = Doc_tfidf[j][x[i]]*Doc_idf[x[i]] #tf*idf

    Q_tfidf = copy.deepcopy(QDict_T)    
    for j in range(0,len(Q_tfidf)-1): 
        
        x = list(QDict_T[j].keys())
        Max = max(QDict_T[j].values()) 

        for i in range(0,len(x)-1):
            #u can add the tf compute there
            #====================================
            if(x[i] in Doc_idf):
                Q_tfidf[j][x[i]] = QDict_T[j][x[i]]*Query_idf[x[i]] #tf*idf
            else:
                Q_tfidf[j][x[i]] = 0
                            
    return(Doc_tfidf,Q_tfidf)



# In[5]:


def VSM(Dirs,Doc_tfidf,Q_tfidf):
    
    Ans_T = []
    #最外圈要加上for q in range(0,len(Q_tfidf)-1): 且所有Q_tfidf[1] 都要改成 q
    for q in range(0,len(Q_tfidf)):
        
        x = list(Q_tfidf[q].keys())
        Sim = []
        for j in range(0,len(Doc_tfidf)):

            a,b,c = 0,0,0

            for k in range(0,len(Q_tfidf[q])):
                if(x[k] in Doc_tfidf[j]):
                    a += (Q_tfidf[q][x[k]]*Doc_tfidf[j][x[k]]) #被除數
                b += pow(Q_tfidf[q][x[k]],2) #除數1
                
            y = list(Doc_tfidf[j].keys())
            for l in range(0,len(Doc_tfidf[j])):                
                c += pow(Doc_tfidf[j][y[l]],2) #除數2

            Sim.append(a / (math.sqrt(b)*math.sqrt(c)))
        
        Sim_sort = sorted(Sim)

        Ans = []
        for i in range(len(Sim_sort)-1,0,-1):
            Ans.append(Dirs[Sim.index(Sim_sort[i])])
        Ans_T.append(Ans)
            
#    for a in range(0,14):
#        print(str(a) + ':' + str(Ans_T[1][a]))
#    print('\n')
#    check(Ans_T[1])
    return(Ans_T)


# In[6]:


def writeAns(Num,Ans,QDirs):
    with open('Answer\Ans'+str(Num)+'.txt','w') as file:
        file.write("Query,RetrievedDocuments\n")
        for i in range(len(Ans)):
            file.write(str(QDirs[i]) + ',')
            for j in Ans[i]:
                file.write(str(j) + ' ')
            file.write('\n')    
        


# In[7]:


#============init without query===========================
Dirs = os.listdir("Document") #Document list
QDirs = os.listdir("Query") #Query list
Dict_T = copy.deepcopy(readDoc_creatDict(Dirs)) #tf
QDict_T = copy.deepcopy(readQuery_creatDict(QDirs))
Doc_freq = {} #df

#將 Dict_T merge, 不能用 update, data會被覆蓋
for i in range(0,len(Dict_T)-1):
    x = list(Dict_T[i].keys())
    for j in range(0,len(x)-1):
        if(str(x[j]) in Doc_freq):
            Doc_freq[x[j]] += 1
        else:
            Doc_freq[x[j]] = 1

#=============tf-idf=====================================


print("第一個:") 
(Doc_tfidf_RFIF_RF,Q_tfidf_RFIF_RF) = tf_idf_RFIF_S(Dirs,Doc_freq,Dict_T,QDict_T) 
Ans = VSM(Dirs,Doc_tfidf_RFIF_RF,Q_tfidf_RFIF_RF)
writeAns(1,Ans,QDirs)

print("第二個:") 
(Doc_tfidf_RFIF_DN05,Q_tfidf_RFIF_DN05) = tf_idf_RFIF_DN05(Dirs,Doc_freq,Dict_T,QDict_T)
Ans = VSM(Dirs,Doc_tfidf_RFIF_DN05,Q_tfidf_RFIF_DN05)
writeAns(2,Ans,QDirs)

print("第三個:") 
(Doc_tfidf_LNIF_DN05,Q_tfidf_LNIF_DN05) = tf_idf_LNIF_DN05(Dirs,Doc_freq,Dict_T,QDict_T)
Ans = VSM(Dirs,Doc_tfidf_LNIF_DN05,Q_tfidf_LNIF_DN05)
writeAns(3,Ans,QDirs)

print("第四個:")
(Doc_tfidf_LNIF_RF,Q_tfidf_LNIF_RF) = tf_idf_LNIF_RF(Dirs,Doc_freq,Dict_T,QDict_T)
Ans = VSM(Dirs,Doc_tfidf_LNIF_RF,Q_tfidf_LNIF_RF)
writeAns(4,Ans,QDirs)

print("第五個:") 
(Doc_tfidf_DN05IF_DN05,Q_tfidf_DN05NIF_DN05) = tf_idf_DN05IF_DN05(Dirs,Doc_freq,Dict_T,QDict_T)
Ans = VSM(Dirs,Doc_tfidf_DN05IF_DN05,Q_tfidf_DN05NIF_DN05)
writeAns(5,Ans,QDirs)

print("第六個:") 
(Doc_tfidf_LNIFS_DN05,Q_tfidf_LNNIFS_DN05) = tf_idf_LNIFS_DN05(Dirs,Doc_freq,Dict_T,QDict_T)
Ans = VSM(Dirs,Doc_tfidf_LNIFS_DN05,Q_tfidf_LNNIFS_DN05)
writeAns(6,Ans,QDirs)

print("第七個:") 
(Doc_tfidf_DN05IFS_DN05,Q_tfidf_DN05NIFS_DN05) = tf_idf_DN05IFS_DN05(Dirs,Doc_freq,Dict_T,QDict_T)
Ans = VSM(Dirs,Doc_tfidf_DN05IFS_DN05,Q_tfidf_DN05NIFS_DN05)
writeAns(7,Ans,QDirs)

print("第八個:")
(Doc_tfidf_LNUN_RFIF,Q_tfidf_LNUNF_RFIF) = tf_idf_LNUN_RFIF(Dirs,Doc_freq,Dict_T,QDict_T)
Ans = VSM(Dirs,Doc_tfidf_LNUN_RFIF,Q_tfidf_LNUNF_RFIF)
writeAns(8,Ans,QDirs)

print("down")

