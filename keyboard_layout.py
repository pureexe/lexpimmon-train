import numpy as np
import random

class KeyboardLayout():
    def __init__(self):
        self.__char_layout = [
            #non shift layout
            [
                ['ๅ','/','-','ภ','ถ','ุ','ึ','ค','ต','จ','ข','ช'],
                ['ๆ','ไ','ำ','พ','ะ','ั','ี','ร','น','ย','บ','ล'],
                ['ฟ','ห','ก','ด','เ','้','่','า','ส','ว','ง','ฃ'],
                ['','ผ','ป','แ','อ','ิ','ื','ท','ม','ใ','ฝ','']
            ],
            #shift layout
            [
                ['+','๑','๒','๓','๔','ู','฿','๕','๖','๗','๘','๙'],
                ['๐','"','ฎ','ฑ','ธ','ํ','๊','ณ','ฯ','ญ','ฐ',','],
                ['ฤ','ฆ','ฏ','โ','ฌ','็','๋','ษ','ศ','ซ','.','ฅ'],
                ['','(',')','ฉ','ฮ','ฺ','์','?','ฒ','ฬ','ฦ','']
            ]
        ]
        self.__shape = [
            len(self.__char_layout),
            len(self.__char_layout[0]),
            len(self.__char_layout[0][0])
        ]
        self.__char_lookup = {}
        for i,layer in enumerate(self.__char_layout):
            for j,row in enumerate(layer):
                for k,character in enumerate(row):
                    if character == '':
                        continue
                    self.__char_lookup[character] = np.array([i,j,k])
    def shape(self):
        return self.__shape

    def position(self,character, normalize = False):
        if not character in self.__char_lookup:
            raise RuntimeError("Don't support '{}' character".format(character))
        position = self.__char_lookup[character]
        if normalize:
            return self.normalize_key(position)
        else:
            return position

    def normalize(self,position):
      position = position.copy()
      for i,key in enumerate(position):
        position[i] = self.normalize_key(key)
      return position

    def normalize_key(self,position):
      for i in range(3):
        #normalize to -1 to 1 (but not exact 1 since position alway less than shape)
        val = position[i] / self.__shape[i] * 2 - 1 
        #move to middle of key
        val += 1 / (self.__shape[i])
        position[i] = val
      return position

    def add_noise(self,position):
      position = position.copy()
      for i,key in enumerate(position):
        position[i] = self.add_noise_key(key)
      return position

    def add_noise_key(self,position):
      position = position.copy()
      for i in range(1,len(self.__shape)): #assume channel 1 has no noise
        half_size = 1 / (self.__shape[i])
        half_size -= 1e-4 #prevent fall in the middle
        move = random.uniform(-half_size, half_size)
        position[i] += move
      return position

    def to_position(self,char_sequence, normalize = False):
        output = np.zeros([len(char_sequence),len(self.__shape)],dtype=np.float64)
        for i,c in enumerate(char_sequence):
          output[i,:] = self.position(c,normalize = normalize)
        return output