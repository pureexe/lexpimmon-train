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
                    self.__char_lookup[character] = [i,j,k]
    def shape(self):
        return self.__shape

    def position(self,character, normalize = False):
        if not character in self.__char_lookup:
            raise RuntimeError("Don't support '{}' character".format(character))
        position = self.__char_lookup[character]
        output = []
        if normalize:
            for i in range(3):
                val = position[i] / self.__shape[i] * 2 - 1
                output.append(val)
        else:
            output = position
        return output
    
    def to_position(self,char_sequence, normalize = False):
        output = []
        for c in char_sequence:
            output.append(self.position(c,normalize = normalize))
        return output

