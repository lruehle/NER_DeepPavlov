from typing_extensions import Concatenate
from deeppavlov import configs, build_model
#from nltk.tokenize import word_tokenize
from bert_dp.tokenization import FullTokenizer
import numpy as np
import datetime


# Functions
#testing
def reduce_token_indx(txt,idx):
    l = " ".join(txt.split()[:idx])
    index = return_token_indx(l)
    if(index>510):
        new_i = index - 510 # distance between index and 510
        idx = int(idx -new_i) # adjust idx to fit 510. roughly every word means two token.
        index = reduce_token_indx(txt,idx)
        return index
    else:
         return idx


def return_token_indx(txt):
    token = tokenizer.tokenize(txt)
    return len(token)

def split_txt(txt):
    seperator=500
    splits = []
    len = return_token_indx(txt)
    while(len>=501):
        index = reduce_token_indx(txt,seperator)
        head = " ".join(txt.split()[:index])
      #  sentenced_head = head.rsplit('.',1)[0] + '.'# reduce to full sentences
        splits.append(head)
        txt = " ".join(txt.split()[index:])
        len = return_token_indx(txt)
    splits.append(txt)
    return splits


#loads and starts the model ner_ontonotes_bert_mult
ner_model = build_model(configs.ner.ner_ontonotes_bert_mult, download=True)

#this ist the text that should be checked
text = open("input.txt","r",encoding="utf-8").read()
tokenizer = FullTokenizer(vocab_file='vocab.txt', do_lower_case=False)
seperator=450
l1 = split_txt(text)
#
# index = reduce_token_indx(text,seperator)
# head = " ".join(text.split()[:index])
# sentenced_head = head.rsplit('.',1)[0] + '.'# reduce to full sentences



#all_token = tokenizer.tokenize(text)
#head = all_token[:500] # what word is this 
#head_str = (" ".join(head)).replace('#','')
#split = all_token[0:50]
#head = " ".join(all_token[:500]) 
#tail = all_token[350:]

#this runs the model with the given text
cs_text = []
ner_text =[]
for l in l1:
    result=ner_model([l]) 
    #write text and NER-Tags into seperate arrays
    cs_text += result[0][0]#np.array(result[0][0])
    ner_text += result[1][0]#np.array(result[1][0])

# here the Models response is being filtered & resorted into wanted output Format
geo_indices = []
geo_B_ind = [] #B-Fac, B-GPE, B-Loc
geo_tags1 =[] # B-FAC, I-FAC, B-GPE, I-GPE, B-LOC, I-LOC
geo_tags2 =[] # B-FAC, I-FAC, B-GPE, I-GPE, B-LOC, I-LOC
for i in range(len(ner_text)):
    if ner_text[i]=="B-FAC":
        geo_indices.append(i)
        geo_B_ind.append(i)
        geo_tags1.append(["\n"+cs_text[i]])
        geo_tags2.append([ner_text[i]])
    elif ner_text[i]=="I-FAC":
        geo_indices.append(i)
        geo_tags1.append([cs_text[i]])
        geo_tags2.append([ner_text[i]])
    elif ner_text[i]=="B-GPE":
        geo_indices.append(i)
        geo_B_ind.append(i)
        geo_tags1.append(["\n"+cs_text[i]])
        geo_tags2.append([ner_text[i]])
    elif ner_text[i]=="I-GPE":
        geo_indices.append(i)
        geo_tags1.append([cs_text[i]])
        geo_tags2.append([ner_text[i]])
    elif ner_text[i]=="B-LOC":
        geo_indices.append(i)
        geo_B_ind.append(i)
        geo_tags1.append(["\n"+cs_text[i]])
        geo_tags2.append([ner_text[i]])
    elif ner_text[i]=="I-LOC":
        geo_indices.append(i)
        geo_tags1.append([cs_text[i]])
        geo_tags2.append([ner_text[i]])
    else:
        continue

#further Output formatting
contxt_geo_tags=[]
for j in range(len(geo_B_ind)):
    count= geo_B_ind[j] #count= geo_indices[j]
    max = len(ner_text)
    if(count<=9 ):
        if(count+16 <=max):
            tag_line = ["'",cs_text[count],"'","["+ner_text[count]+"]","in : '"]
            for k in range(0, count+16):
                tag_line.append(cs_text[k])
            tag_line.append("'")
            contxt_geo_tags.append(tag_line)
        elif(count+16 <=max):
            tag_line = ["'",cs_text[count],"'","["+ner_text[count]+"]","in : '"]
            for k in range(0, count+16):
                tag_line.append(cs_text[k])
            tag_line.append("'")
            contxt_geo_tags.append(tag_line)
        else:
            tag_line = ["'",cs_text[count],"'","["+ner_text[count]+"]","in : '"]
            for k in range(count+1):
                tag_line.append(cs_text[k])
            tag_line.append("'")
            contxt_geo_tags.append(tag_line)
    else:
        if(count+16 <= max):
            tag_line= ["'",cs_text[count],"'","["+ner_text[count]+"]","in : '"]
            for k in range(count-10,count+16):
                tag_line.append(cs_text[k])
            tag_line.append("'")
            contxt_geo_tags.append(tag_line)
        else:
            tag_line = ["'",cs_text[count],"'","["+ner_text[count]+"]","in : '"]
            for k in range(count-10,count+3):
                tag_line.append(cs_text[k])
            tag_line.append("'")
            contxt_geo_tags.append(tag_line)
        
    
#formatted output is being written into seperate files
merged_all=np.column_stack((cs_text,ner_text))
merged_geo=np.column_stack((geo_tags1,geo_tags2))
np.savetxt("output_merged.txt",merged_all,fmt='%s',delimiter=' : ',newline=',\n')
np.savetxt("output_geo.txt",merged_geo,fmt='%s',delimiter=' : ',newline='\n')
np.savetxt("output_geo_context.txt",contxt_geo_tags,fmt='%s',delimiter=' ',newline='\n\n\n')

#print(result)
print("done")
#python -m deeppavlov interact ner_ontonotes_bert_mult
#python DeepPavlov_Dost.py or press "run"

