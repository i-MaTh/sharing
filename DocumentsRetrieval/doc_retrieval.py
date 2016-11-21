# -*- coding: utf-8 -*-
"""\
---------------------------------------------------------------------------------------
    USE: python <PROGNAME>  -d DOCUMENTS -q QUERIES -s STOP_LIST -i INDEX -o OUTPUT
        -h : print this help message
        -d <file> : input <file> as documents
        -q <file> : input <file> as queries
        -s <file> : input <file> as stop_list
        -i <file> : if inverted index <file> does not exist, building a new one to <file>.
        -o <file> : output result to <file>
        -k <int> : top-k relevant results for each query
        
    EXAMPLES: python doc_retrieval.py -d documents.txt -q queries.txt -s stop_list.txt -i inverted_index.txt -o output.txt
---------------------------------------------------------------------------------------
"""
import os, sys, re
import getopt
from os.path import basename
import numpy as np
from read_documents import ReadDocuments
from nltk.stem import PorterStemmer


class CommandLine:
    def __init__(self):
        opts, args = getopt.getopt(sys.argv[1:],'hd:q:s:i:o:')
        opts = dict(opts)

        if '-h' in opts:
            self.printHelp()
        
        if '-d' in opts:
            self.documents = opts['-d']
            
        if '-q' in opts:
            self.queries = opts['-q']
            
        if '-s' in opts:
            self.stop_list = opts['-s']
            
        if '-i' in opts:
            self.inverted_index = opts['-i']
        
        if '-o' in opts:
            self.output = opts['-o']
            
        if '-n' in opts:
            self.top_k = int(opts['-n'])
        else:
            self.top_k = 10

    def printHelp(self):
        help = __doc__.replace('<PROGNAME>',sys.argv[0],1)
        print(help, file=sys.stderr)
        sys.exit()  
        

class DocumentRetrieval:
    
    def __init__(self, d_file, q_file, s_file):
        self.stop_list_file = s_file
        self.q_file = q_file
        self.doc_file = d_file        
        
        
    def load_stop_list(self):
        stop_list = set()
        with open(self.stop_list_file, 'r') as input_stop_list:
            skip = re.compile('\S')
            for word in input_stop_list:
                if skip.search(word):
                    stop_list.add(word.strip('\n'))
        
        return stop_list
        

    def build_inv_index(self, stop_list = None, inv_index_file = None):
        re_word = re.compile(r'[A-Za-z]+')
        stemmer = PorterStemmer()
        inv_index = {}
        for doc in ReadDocuments(self.doc_file):
            for line in doc.lines:
                for word in re_word.findall(line.lower()):
                    word = stemmer.stem(word)
                    if word in stop_list:
                        continue
                    if word not in inv_index:    
                        inv_index[word] = {}
                    if doc.docid not in inv_index[word]:
                        inv_index[word][doc.docid] = 1
                    else:
                        inv_index[word][doc.docid] += 1
                                  
        with open(inv_index_file, 'w') as out_handle:
            for first_key in inv_index.keys():
                for items in sorted(inv_index[first_key].items(), key = lambda item:str(item[0])):
                    out_handle.write('{0}\t{1}\t{2}\n'.format(first_key, items[0], items[1]))
                
        out_handle.close()  


    def load_inv_index(self, _file):
        df = {}
        docs_vector = {}
        D = 0
        with open(_file, 'r') as inv_index:
            for line in inv_index:
                items = re.split('\s+', line)
                if items[0] not in df:
                    df.setdefault(items[0], {items[1]:items[2]})
                else:
                    df[items[0]][items[1]] = items[2]                  
                    
                if items[1] in docs_vector:
                    docs_vector[items[1]].setdefault(items[0], int(items[2]))
                else:
                    D += 1
                    docs_vector.setdefault(items[1], {items[0] : int(items[2])})
        inv_index.close()
            
        for docid in docs_vector:
            for word in docs_vector[docid]:
                docs_vector[docid][word] *= np.log10(D/float(len(df[word])))
        
        self.D, self.df, self.docs_vector = D, df, docs_vector


    def cal_query_vector(self, queries_vector):
        for q_id in queries_vector:
            remove = []
            for word in queries_vector[q_id]:
                if word in self.df:
                    queries_vector[q_id][word] *= np.log10(self.D/float(len(self.df[word])))
                else:
                    remove.append(word)
                    
            for word in remove:
                del queries_vector[q_id][word]
         
        return queries_vector
        
                
    def all_query(self, q_file, stop_list):
        re_word = re.compile(r'[A-Za-z]+')
        stemmer = PorterStemmer()
        dict_words = dict([])
        for query in ReadDocuments(q_file):
            dict_words[query.docid] = {}
            for line in query.lines:
                for word in re_word.findall(line.lower()):
                    word = stemmer.stem(word)
                    if word in stop_list:
                        continue
                    if word in dict_words[query.docid]:
                        dict_words[query.docid][word] += 1
                    else:
                        dict_words[query.docid].setdefault(word, 1)
        
        return self.cal_query_vector(dict_words)
        
        
    def similarity(self, queries_vector):
        dict_sim = {}
        for q_instance in queries_vector:
             dict_sim[q_instance] = {}
             for d_instance in self.docs_vector:
                 int_sec = set(queries_vector[q_instance].keys()).intersection(set(self.docs_vector[d_instance].keys()))
                 if int_sec:
                     d_d = np.sqrt(sum([x**2 for x in self.docs_vector[d_instance].values()]))
                     d_q = sum([queries_vector[q_instance][i] * self.docs_vector[d_instance][i] for i in int_sec])
                     dict_sim[q_instance][d_instance] = d_q / float(d_d)
        return dict_sim
        
        
    def ouput(self, dict_sim, o_file, k = 6):
        with open(o_file, 'w') as output:
            queries = sorted(dict_sim.keys())
            for key in queries:
                result = sorted(dict_sim[key].items(), key = lambda item:str(item[1]), reverse=True)
                k = len(result) if len(result) < k else k
                for i in range(k):
                    output.write('{0} {1}\n'.format(int(key), int(result[i][0])))
        output.close()       
             
                   
def main(config):
    dr = DocumentRetrieval(config.documents, config.queries, config.stop_list)
    stpl = dr.load_stop_list()
    if not os.path.exists(config.inverted_index):
        dr.build_inv_index(stop_list = stpl, inv_index_file = config.inverted_index)
        
    dr.load_inv_index(config.inverted_index)
    q_vec = dr.all_query(q_file = config.queries, stop_list = config.stop_list)
    sim = dr.similarity(q_vec)  
    dr.ouput(dict_sim = sim, o_file = config.output, k = config.top_k)
    
    
if __name__ == '__main__':
    config = CommandLine()
    main(config)
    



