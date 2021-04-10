from typing import *
import time
import torch
import torch.nn as nn
import torch.optim as optim
import json
import random
import logging
from torch.types import Number
import tqdm
import gc
import pickle
import os
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)s] - %(message)s",
)

logger = logging.getLogger(__name__)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class BiLSTM_CRF_MODIFY_PARALLEL(nn.Module):
    START_TAG = "<START>"
    STOP_TAG = "<STOP>"
    PAD_TAG = "<PAD>"
    @staticmethod
    def argmax(vec):
        # return the argmax as a python int
        _, idx = torch.max(vec, 1)
        return idx.item()

    @staticmethod
    def prepare_sequence(seq, to_ix)->torch.Tensor:
        idxs = [to_ix[w] for w in seq]
        return torch.tensor(idxs, dtype=torch.long).cuda()

    @staticmethod
    def prepare_sequence_batch(data:List, word_to_ix, tag_to_ix,max_len)->Tuple[torch.Tensor,torch.Tensor]:
        # seqs = [i[0] for i in data]
        # tags = [i[1] for i in data]
        
        seqs_pad = []
        tags_pad = []
        temp=0
        with tqdm.tqdm(total=len(data), ncols=80,desc="prepare_sequence_batch") as tqbar:
            for i in data:
                seq,tag=i[0],i[1]
                seq_pad = seq + ['<PAD>'] * (max_len-len(seq))
                tag_pad = tag + ['<PAD>'] * (max_len-len(tag))
                seqs_pad.append(seq_pad)
                tags_pad.append(tag_pad)
                
            
                tqbar.update(1)
        
        idxs_pad = torch.tensor([[word_to_ix[w] for w in seq]
                                for seq in seqs_pad], dtype=torch.int32).long().cuda()
        tags_pad = torch.tensor([[tag_to_ix[t] for t in tag]
                                for tag in tags_pad], dtype=torch.int32).long().cuda()
        return idxs_pad,tags_pad
    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim):
        super(BiLSTM_CRF_MODIFY_PARALLEL, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)
        self.opetimizer=None
        self.word_embeds = nn.Embedding(vocab_size, embedding_dim).cuda()
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True, batch_first=True).cuda()

        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size).cuda()

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size)).cuda()

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag

        self.transitions.data[tag_to_ix[self.START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[self.STOP_TAG]] = -10000
        self.hidden = self.init_hidden()

    def set_optimizer(self,optimizer:optim.Optimizer):
        self.opetimizer=optimizer
    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_dim // 2).cuda(),
                torch.randn(2, 1, self.hidden_dim // 2).cuda())

    def _forward_alg(self, feats):
        begin = time.time()
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full((1, self.tagset_size), -10000.).to('cuda')
        # self.START_TAG has all of the score.
        init_alphas[0][self.tag_to_ix[self.START_TAG]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        forward_var = init_alphas
        # print('time consuming of crf_partion_function_prepare:%f' % (time.time() - begin))
        begin = time.time()
        # Iterate through the sentence
        for feat in feats:
            alphas_t = []  # The forward tensors at this timestep
            for next_tag in range(self.tagset_size):
                # broadcast the emission score: it is the same regardless of
                # the previous tag
                emit_score = feat[next_tag].view(
                    1, -1).expand(1, self.tagset_size)
                # the ith entry of trans_score is the score of transitioning to
                # next_tag from i
                trans_score = self.transitions[next_tag].view(1, -1)
                # The ith entry of next_tag_var is the value for the
                # edge (i -> next_tag) before we do log-sum-exp
                next_tag_var = (forward_var + trans_score + emit_score)
                # The forward variable for this tag is log-sum-exp of all the
                # scores.
                alphas_t.append(self.log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1).cuda()
        # print('time consuming of crf_partion_function1:%f' % (time.time() - begin))
        terminal_var = forward_var + self.transitions[self.tag_to_ix[self.STOP_TAG]]
        alpha = self.log_sum_exp(terminal_var)
        # print('time consuming of crf_partion_function2:%f' %(time.time()-begin))
        return alpha

    def _forward_alg_new(self, feats):
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full([self.tagset_size], -10000.).to('cuda')
        # self.START_TAG has all of the score.
        init_alphas[self.tag_to_ix[self.START_TAG]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        # Iterate through the sentence
        forward_var_list = []
        forward_var_list.append(init_alphas)
        for feat_index in range(feats.shape[0]):  # -1
            gamar_r_l = torch.stack(
                [forward_var_list[feat_index]] * feats.shape[1]).cuda()
            # gamar_r_l = torch.transpose(gamar_r_l,0,1)
            t_r1_k = torch.unsqueeze(
                feats[feat_index], 0).transpose(0, 1).cuda()  # +1
            aa = gamar_r_l + t_r1_k + self.transitions
            # forward_var_list.append(log_add(aa))
            forward_var_list.append(torch.logsumexp(aa, dim=1).cuda())
        terminal_var = forward_var_list[-1] + \
            self.transitions[self.tag_to_ix[self.STOP_TAG]]
        terminal_var = torch.unsqueeze(terminal_var, 0).cuda()
        alpha = torch.logsumexp(terminal_var, dim=1)[0]
        return alpha

    def _forward_alg_new_parallel(self, feats):
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full(
            [feats.shape[0], self.tagset_size], -10000.) .to('cuda')
        # self.START_TAG has all of the score.
        init_alphas[:, self.tag_to_ix[self.START_TAG]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        # Iterate through the sentence
        forward_var_list = []
        forward_var_list.append(init_alphas)
        for feat_index in range(feats.shape[1]):  # -1
            gamar_r_l = torch.stack(
                [forward_var_list[feat_index]] * feats.shape[2]).transpose(0, 1).cuda()
            # gamar_r_l = torch.transpose(gamar_r_l,0,1)
            t_r1_k = torch.unsqueeze(
                feats[:, feat_index, :], 1).transpose(1, 2).cuda()  # +1
            # t_r1_k = feats[:,feat_index,:].repeat(feats.shape[0],1,1).transpose(1, 2)
            aa = gamar_r_l + t_r1_k + torch.unsqueeze(self.transitions, 0).cuda()
            # forward_var_list.append(log_add(aa))
            forward_var_list.append(torch.logsumexp(aa, dim=2))
        terminal_var = forward_var_list[-1] + \
            self.transitions[self.tag_to_ix[self.STOP_TAG]].repeat(
                [feats.shape[0], 1]).cuda()
        # terminal_var = torch.unsqueeze(terminal_var, 0)
        alpha = torch.logsumexp(terminal_var, dim=1)
        return alpha

    def _get_lstm_features(self, sentence):
        self.hidden = self.init_hidden()
        embeds = self.word_embeds(sentence).unsqueeze(dim=0).cuda()
        #embeds = self.word_embeds(sentence).view(len(sentence), 1, -1).transpose(0,1)
        lstm_out, self.hidden = self.lstm(embeds)
        #lstm_out = lstm_out.view(embeds.shape[1], self.hidden_dim)
        lstm_out = lstm_out.squeeze()
        lstm_feats = self.hidden2tag(lstm_out).cuda()
        return lstm_feats

    def _get_lstm_features_parallel(self, sentence):
        self.hidden = self.init_hidden()
        embeds = self.word_embeds(sentence).cuda()
        lstm_out, self.hidden = self.lstm(embeds)
        lstm_feats = self.hidden2tag(lstm_out.cuda())
        return lstm_feats

    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = torch.zeros(1).cuda()
        # score = autograd.Variable(torch.Tensor([0])).to('cuda')
        tags = torch.cat(
            [torch.tensor([self.tag_to_ix[self.START_TAG]], dtype=torch.int32).cuda(), tags.view(-1)]).cuda()

        # if len(tags)<2:
        #     print(tags)
        #     sys.exit(0)
        for i, feat in enumerate(feats):
            score = score + \
                self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix[self.STOP_TAG], tags[-1]]
        return score

    def _score_sentence_parallel(self, feats, tags):
        # Gives the score of provided tag sequences
        #feats = feats.transpose(0,1)

        score = torch.zeros(tags.shape[0]).to('cuda')
        tags = torch.cat([torch.full(
            [tags.shape[0], 1], self.tag_to_ix[self.START_TAG], dtype=torch.int32).cuda(), tags], dim=1).cuda()
        for i in range(feats.shape[1]):
            feat = feats[:, i, :]
            score = score + \
                self.transitions[tags[:, i + 1], tags[:, i]] + \
                feat[range(feat.shape[0]), tags[:, i + 1]]
        score = score + self.transitions[self.tag_to_ix[self.STOP_TAG], tags[:, -1]]
        return score

    def _viterbi_decode(self, feats):
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.full((1, self.tagset_size), -10000.).cuda()
        init_vvars[0][self.tag_to_ix[self.START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars

        for feat in feats:
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.tagset_size):
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                next_tag_var = forward_var.to(
                    'cuda') + self.transitions[next_tag]
                best_tag_id = self.argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # Transition to self.STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag_to_ix[self.STOP_TAG]]
        best_tag_id = self.argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.tag_to_ix[self.START_TAG]  # Sanity check
        best_path.reverse()
        return path_score, best_path

    def _viterbi_decode_new(self, feats):
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.full((1, self.tagset_size), -10000.) .to('cuda')
        init_vvars[0][self.tag_to_ix[self.START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var_list = []
        forward_var_list.append(init_vvars)

        for feat_index in range(feats.shape[0]):
            gamar_r_l = torch.stack(
                [forward_var_list[feat_index]] * feats.shape[1]).cuda()
            gamar_r_l = torch.squeeze(gamar_r_l).cuda()
            next_tag_var = gamar_r_l + self.transitions
            # bptrs_t=torch.argmax(next_tag_var,dim=0)
            viterbivars_t, bptrs_t = torch.max(next_tag_var, dim=1)

            t_r1_k = torch.unsqueeze(feats[feat_index], 0).cuda()
            forward_var_new = torch.unsqueeze(viterbivars_t, 0).cuda() + t_r1_k

            forward_var_list.append(forward_var_new)
            backpointers.append(bptrs_t.tolist())

        # Transition to self.STOP_TAG
        terminal_var = forward_var_list[-1] + \
            self.transitions[self.tag_to_ix[self.STOP_TAG]]
        best_tag_id = torch.argmax(terminal_var).tolist()
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.tag_to_ix[self.START_TAG]  # Sanity check
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, sentence, tags):
        feats = self._get_lstm_features(sentence)
        forward_score = self._forward_alg_new(feats)
        gold_score = self._score_sentence(feats, tags)[0]
        return forward_score - gold_score

    def neg_log_likelihood_parallel(self, sentences, tags):
        feats = self._get_lstm_features_parallel(sentences)
        forward_score = self._forward_alg_new_parallel(feats)
        gold_score = self._score_sentence_parallel(feats, tags)
        return torch.sum(forward_score - gold_score).cuda()

    def forward(self, sentence):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence)

        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode_new(lstm_feats)
        return score, tag_seq
    def train_model(self,train_data:List[Dict],tag_to_idx:Dict[str,int],word_to_idx:Dict[str,int],max_len:Number,epoch:int,batch_size=5000)->None:
        idxs_pad,tags_pad=self.prepare_sequence_batch(train_data,word_to_idx,tag_to_idx,max_len)
        batch=0
        with tqdm.tqdm(total=len(train_data)*epoch,ncols=80,desc="train") as tqbar:
            for current_epoch in range(epoch):
                batch=0
                self.zero_grad()
                while(batch+batch_size)<=len(train_data):
                    setences=idxs_pad[batch:batch+batch_size].cuda()
                    target=tags_pad[batch:batch+batch_size].cuda()
                    loss=self.neg_log_likelihood_parallel(setences,target)
                    batch+=batch_size
                    logger.info("\nepoch {0} : {1}/{2} ;loss is {3}".format(
                        current_epoch,batch,len(train_data),loss.cpu()/batch_size
                    ))
                    loss.backward()
                    self.opetimizer.step()
                    self.zero_grad()
                    tqbar.update(batch_size)
                if batch==len(train_data):
                    continue
                setences=idxs_pad[batch:].cuda()
                target=tags_pad[batch:].cuda()
                loss=self.neg_log_likelihood_parallel(setences,target)
                logger.info("\nepoch {0} : {1}/{2} ;loss is {3}".format(
                        current_epoch,len(train_data),len(train_data),loss.cpu()/(len(train_data)-batch)
                ))
                loss.backward()
                self.opetimizer.step()
                self.zero_grad()
                tqbar.update(len(train_data)-batch)
        torch.save(self,"model")
def make_training_data(a):
    training_data1=[]
    with tqdm.tqdm(total=len(a), ncols=80,desc="make training data") as tqbar:
        for i in range(len(a)):
            wrongdata = 0
            record=a[i]
            try:
                if '原因中的核心名词' in record['key_words'] and record['key_words']['原因中的核心名词']['start'] == -1:
                    wrongdata += 1
                else:
                    x = ['O']*len(list(record['text']))
                    if '原因中的核心名词' in record['key_words']:
                        for j in range(record['key_words']['原因中的核心名词']['start'], record['key_words']['原因中的核心名词']['end']+1):
                            if j == record['key_words']['原因中的核心名词']['start']:
                                x[j] = 'reason_noun_B'
                            else:
                                x[j] = 'reason_noun_I'
                    if '原因中的谓语或状态' in record['key_words']:
                        for j in range(record['key_words']['原因中的谓语或状态']['start'], record['key_words']['原因中的谓语或状态']['end']+1):
                            if j == record['key_words']['原因中的谓语或状态']['start']:
                                x[j] = 'reason_state_B'
                            else:
                                x[j] = 'reason_state_I'
                    if '结果中的核心名词' in record['key_words']:
                        for j in range(record['key_words']['结果中的核心名词']['start'], record['key_words']['结果中的核心名词']['end']+1):
                            if j == record['key_words']['结果中的核心名词']['start']:
                                x[j] = 'result_noun_B'
                            else:
                                x[j] = 'result_noun_I'
                    if '结果中的谓语或状态' in record['key_words']:
                        for j in range(record['key_words']['结果中的谓语或状态']['start'], record['key_words']['结果中的谓语或状态']['end']+1):
                            if j == record['key_words']['结果中的谓语或状态']['start']:
                                x[j] = 'result_state_B'
                            else:
                                x[j] = 'result_state_I'
                    temp = (list(record['text']), x)
                    training_data1.append(temp)
            except Exception as ex:
                logger.warning(ex,record)
                pass
            tqbar.update(1)
    return training_data1
def preprocess(rawdata:List[Dict])->Tuple[List,Dict[str,int],int]:
    
    data=make_training_data(raw_data)
    
    word_to_idx:Dict[str,int]={}
    word_to_idx['<PAD>']=0
    max_len=0
    with tqdm.tqdm(total=len(data),ncols=80,desc="word to index") as tqbar:
            for sentence,tag in data:
                max_len=max(len(sentence),max_len)
                for word in sentence:
                    if not(word in word_to_idx):
                        word_to_idx[word]=len(word_to_idx)
                tqbar.update(1)
    return data,word_to_idx,max_len
    
if __name__=="__main__":
    with open('./train_data.json', 'r') as f:
        raw_data:List = json.load(f) 
    random.shuffle(raw_data)
    random.shuffle(raw_data)

    logger.info("finish loading training data")
    EMBEDDING_DIM = 50
    HIDDEN_DIM = 32
    tag_to_ix = {'reason_noun_B': 0, 'reason_noun_I': 1, 'reason_state_B': 2, 'reason_state_I': 3, 'result_noun_B': 4,
                 'result_noun_I': 5, 'result_state_B': 6, 'result_state_I': 7, 'O': 8,BiLSTM_CRF_MODIFY_PARALLEL.START_TAG: 9, BiLSTM_CRF_MODIFY_PARALLEL.STOP_TAG: 10, BiLSTM_CRF_MODIFY_PARALLEL.PAD_TAG: 11}
    
    training_data, word_to_idx,max_len=preprocess(raw_data)
    model=BiLSTM_CRF_MODIFY_PARALLEL(len(word_to_idx),tag_to_ix,EMBEDDING_DIM,HIDDEN_DIM)
    model.set_optimizer(optim.Adam(model.parameters(),lr=0.01,weight_decay=1e-4))
    model.train_model(training_data,tag_to_ix,word_to_idx,max_len,epoch=20,batch_size=5000)
