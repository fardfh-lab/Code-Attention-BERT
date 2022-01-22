import torch
import torch.nn as nn
import torch
import javalang
from torch.autograd import Variable
import copy
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss

class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size*2, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, 2)

    def forward(self, features, **kwargs):
        x = features
        x = x.reshape(-1,x.size(-1)*2)
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
        
class Model(nn.Module):
    def __init__(self, encoder,config,tokenizer,args):
        super(Model, self).__init__()
        self.encoder = encoder
        self.config=config
        self.tokenizer=tokenizer
        self.classifier=RobertaClassificationHead(config)
        self.args=args



    def forward(self, input_ids=None,labels=None): 
        input_ids=input_ids.view(-1,self.args.block_size)
        customFeatures = self.makeStr(input_ids)
        customFeatures = torch.tensor(customFeatures)
        customFeatures = customFeatures.view(-1, 768) 
        squeezed = torch.squeeze(customFeatures)
  
        outputs = self.encoder(input_ids= input_ids,attention_mask=input_ids.ne(1), output_hidden_states=True)
        
        inputToClass = outputs['hidden_states'][-1][:, 0, :]
        
        inputToClass.copy_(customFeatures)

        logits=self.classifier(inputToClass)
        
        prob=F.softmax(logits)
        if labels is not None:
          loss_fct = CrossEntropyLoss()
          loss = loss_fct(logits, labels)
          return loss,prob
        else:
          return prob


    def getIdentifiers(self,idToString, lenHid):
        indices = []
        tree = list(javalang.tokenizer.tokenize(idToString + " }"))
        for index, i in enumerate(tree):
          
            j = str(i)
            j = j.split()
            typee = j[0]
            token = j[1].strip('"')

            if typee=="Identifier":
                indices.append(index)
            #print(token, " ", typee)

        while True:
            if lenHid>=int(indices[-1]):
                break
            else:
                indices = indices[:len(indices)-1]
        return indices

    def makeStr(self, inputIDS, ):
        # split two code samples
        #convert ids to tokens
        # remove padding
        hiddenToReturn = []
        #print(inputIDS.shape)
        
        layer = -1
        
        for eachInput in inputIDS:
            st = ""
            finToks = []
            finHids = []
            finAtts = []
            eachInput = eachInput[eachInput!=0]
            tokens = self.tokenizer.convert_ids_to_tokens(eachInput)
            outputs = self.encoder(input_ids= eachInput.reshape(1,-1), output_hidden_states=True, output_attentions=True)
            hidden = outputs['hidden_states'][layer] #LAST LAYER USED
            attention = outputs['attentions'][layer] #LAST LAYER USED
            into =False
            for i in range(1, len(tokens)-1):
                if(i==1):
                  st = str(tokens[i])
                  finToks.append(str(tokens[i]))
                  finHids.append(hidden[:, i, :])
                  finAtts.append(attention[0][-1].sum(1)[i])
                else:
                  if "##" in str(tokens[i]):
                    finToks[-1] = finToks[-1] + str(tokens[i]).replace("##", "")
                    finHids[-1] = finHids[-1] + hidden[:, i, :]
                    finAtts[-1] = finAtts[-1] + attention[0][-1].sum(1)[i]
                  elif tokens[i]=="str" and finToks[-1]=='"':
                    finToks[-1] = '"str"'
                    into = True
                  
                  else:
                    if into==True:
                      into = False
                      finHids[-1] = finHids[-1] + hidden[:, i, :]
                      finAtts[-1] = finAtts[-1] + attention[0][-1].sum(1)[i]
                      
                      
                    else:
                      finToks.append(str(tokens[i]))
                      finHids.append(hidden[:, i, :])
                      finAtts.append(attention[0][-1].sum(1)[i])

            idToString = " ".join(finToks)
            indexes = self.getIdentifiers(idToString, len(finHids))
            outHidden = torch.zeros((1,768)).cuda()
            #TODO: TOPK IF REQUIRED
            idfsToPrint = ""
            for eachIndex in indexes:
                  outHidden += finHids[eachIndex] * ( finAtts[eachIndex] / len(indexes) )
                  idfsToPrint = idfsToPrint + " " +  finToks[eachIndex] 
                  
            hiddenToReturn.append(outHidden.cpu().detach().numpy())
   
        return hiddenToReturn
