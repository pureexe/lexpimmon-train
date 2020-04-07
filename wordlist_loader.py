from keyboard_layout import KeyboardLayout

class WordListLoader():
    def __init__(self, lower_bound = 2, upper_bound = 12):
        self.__word_list = {}
        self.__features = {}
        for i in range(lower_bound,upper_bound+1):
            self.__word_list[i] = []
            self.__features[i] = []
        kb = KeyboardLayout()
        kb_size = kb.shape()
        max_len = 0
        # read text file
        with open('data/thai-wordlist.txt','r',encoding='utf-8') as f:
            while True:
                word = f.readline().strip()
                if(word == ''): break
                try:                    
                    km = kb.to_position(word, normalize = True)                    
                    word_len = len(word)
                    if word_len > upper_bound or word_len < lower_bound:
                        continue
                    self.__word_list[word_len].append(word)
                    self.__features[word_len].append(km)
                except:
                    pass
        
    def get_features(self, word_len = 0, idx = None):
        if idx is None:
            return self.__features[word_len]
        else:
            return self.__features[word_len][idx]
    
    def get_labels(self,word_len):
        return list(range(len(self.__features[word_len])))

    def get_class_label(self,word_len, idx):
        return self.__word_list[word_len][idx]
