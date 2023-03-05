import torch.utils.data as data
import os
import torch
import random
import pandas as pd
import numpy as np

ppg_dir = '/Users/yeseullee/Documents/ECE271B/PPGBPDatabase/Data File/0_subject'
label_path = '/Users/yeseullee/Documents/ECE271B/PPGBPDatabase/Data File/PPG-BP dataset.csv'

## Choose High Quality Data ##

# organized data from https://www.nature.com/articles/sdata201820/tables/2

quality_dict = dict()
# key: subject number; value[0]: subject ID; value[1]:quality of segment 1; value[2]:quality of segment 2; value[3]:quality of segment 3
quality_dict[1] = [2, 0.98, 0.96, 0.92]
quality_dict[2] = [3, 0.69, 0.8, 0.81]
quality_dict[3] = [6, 0.58, 0.59, 0.64]
quality_dict[4] = [8, 0.96, 0.85, 0.87]
quality_dict[5] = [9, 0.65, 0.67, 0.87]
quality_dict[6] = [10, 0.59, 0.64, 0.34]
quality_dict[7] = [11, 0.74, 0.67, -0.16]
quality_dict[8] = [12, 0.23, 0.73, 0.41]
quality_dict[9] = [13, 0.76, 0.84, 1.06]
quality_dict[10] = [14, 0.77, 0.72, 0.15]

quality_dict[11] = [15, 1.23, 0.77, 0.3]
quality_dict[12] = [16, 0.64, 0.66, 0.83]
quality_dict[13] = [17, 0.69, 0.9, 0.9]
quality_dict[14] = [18, 0.87, 0.59, 1.05]
quality_dict[15] = [19, 0.78, 0.19, 0.16]
quality_dict[16] = [21, 0.65, 0.74, 0.75]
quality_dict[17] = [22, 0.73, 0.44, 0.39]
quality_dict[18] = [23, 0.7, 0.6, 0.74]
quality_dict[19] = [24, 0.74, 0.74, 0.75]
quality_dict[20] = [25, 1.38, 0.15, 0.78]

quality_dict[21] = [26, 1.31, 0.47, 0.83]
quality_dict[22] = [27, 1.86, 1.33, 1.26]
quality_dict[23] = [29, 0.73, 0.6, -0.05]
quality_dict[24] = [30, 0.86, 0.8, 0.76]
quality_dict[25] = [31, 0.68, 0.81, 0.7]
quality_dict[26] = [32, 1.06, 1.12, 1.23]
quality_dict[27] = [34, 0.9, 0.79, 0.74]
quality_dict[28] = [35, 0.94, 1.15, 1.18]
quality_dict[29] = [38, 0.88, 0.66, 0.94]
quality_dict[30] = [40, 0.21, 1.2, 0.9]

quality_dict[31] = [41, 0.85, 0.78, 0.73]
quality_dict[32] = [43, 0.52, 0.28, 0.53]
quality_dict[33] = [45, 0.68, 0.76, 0.67]
quality_dict[34] = [47, 0.76, 0.75, 0.73]
quality_dict[35] = [48, 0.74, 0.79, 0.63]
quality_dict[36] = [50, 0.79, 0.72, 0.73]
quality_dict[37] = [51, 0.6, 0.2, 0.49]
quality_dict[38] = [52, 0.95, 0.67, 0.93]
quality_dict[39] = [53, 0.81, 0.91, 0.9]
quality_dict[40] = [54, 1.12, 1.06, 1.11]

quality_dict[41] = [55, 0.63, 0.99, 1.1]
quality_dict[42] = [56, 0.26, 0.4, 0.47]
quality_dict[43] = [57, 0.69, 0.59, 0.65]
quality_dict[44] = [58, 0.97, 0.62, 0.75]
quality_dict[45] = [60, 0.37, 1.64, 0.51]
quality_dict[46] = [61, 0.87, 0.84, 0.93]
quality_dict[47] = [62, 0.57, 0.96, 0.69]
quality_dict[48] = [63, 1.01, 0.92, 0.92]
quality_dict[49] = [64, 0.19, 0.65, 0.39]
quality_dict[50] = [65, 0.78, 0.8, 0.72]

quality_dict[51] = [66, 0.79, 0.79, 0.94]
quality_dict[52] = [67, 0.7, 0.7, 0.99]
quality_dict[53] = [83, 0.81, 0.9, 0.79]
quality_dict[54] = [84, 0.91, 0.3, 0.65]
quality_dict[55] = [85, 0.68, 0.79, 0.63]
quality_dict[56] = [86, 0.66, 0.7, 0.72]
quality_dict[57] = [87, 0.97, 0.96, 0.95]
quality_dict[58] = [88, 0.81, 0.52, 0.65]
quality_dict[59] = [89, 0.39, 0.58, 0.14]
quality_dict[60] = [90, 0.87, 0.97, 1]

quality_dict[61] = [91, 1.05, 0.77, 0.84]
quality_dict[62] = [92, 0.9, 1.1, 1.15]
quality_dict[63] = [93, 0.97, 0.99, 0.46]
quality_dict[64] = [95, 1.31, 0.89, 0.87]
quality_dict[65] = [96, 0.75, 0.89, 0.81]
quality_dict[66] = [97, 0.56, 0.42, 0.76]
quality_dict[67] = [98, 0.88, 0.98, 0.86]
quality_dict[68] = [99, 0.88, 0.72, 0.79]
quality_dict[69] = [100, 0.58, 0.66, 0.16]
quality_dict[70] = [103, 0.37, 0.4, 0.44]

quality_dict[71] = [104, 0.88, 0.23, 0.85]
quality_dict[72] = [105, 0.54, 0.97, 0.8]
quality_dict[73] = [106, 0.82, 0.9, 1.16]
quality_dict[74] = [107, 0.89, 0.58, 0.66]
quality_dict[75] = [108, 0.71, 0.69, 0.64]
quality_dict[76] = [110, 0.9, 0.83, 0.88]
quality_dict[77] = [111, 0.9, 0.85, 0.76]
quality_dict[78] = [112, 0.61, 0.55, 0.57]
quality_dict[79] = [113, 0.35, 0.51, 0.78]
quality_dict[80] = [114, 0.5, 0.58, 0.67]

quality_dict[81] = [115, 0.74, 0.06, 1.03]
quality_dict[82] = [116, 0.78, 0.86, 0.93]
quality_dict[83] = [119, 0.55, 0.59, -0.07]
quality_dict[84] = [120, 0.79, 0.67, 0.77]
quality_dict[85] = [122, 0.93, 0.87, 0.5]
quality_dict[86] = [123, 0.89, 0.97, 1.3]
quality_dict[87] = [124, 0.93, 1.23, 1.19]
quality_dict[88] = [125, 0.84, -0.47, 0.86]
quality_dict[89] = [126, 0.44, -0.01, 0.54]
quality_dict[90] = [127, 0.53, 0.83, 0.75]

quality_dict[91] = [128, 0.87, 0.86, 0.9]
quality_dict[92] = [130, 0.91, 0.97, 1]
quality_dict[93] = [131, 0.86, 0.85, 0.75]
quality_dict[94] = [134, 0.71, 0.21, 0.86]
quality_dict[95] = [135, 0.68, 0.72, 0.67]
quality_dict[96] = [136, 1.73, 0.56, 0.8]
quality_dict[97] = [137, 0.28, 0.58, 0.79]
quality_dict[98] = [138, 0.74, 0.48, 0.59]
quality_dict[99] = [139, 1.35, 0.69, 0.63]
quality_dict[100] = [140, 0.41, 0.85, 0.71]

quality_dict[101] = [141, 1.14, 0.96, 0.86]
quality_dict[102] = [142, 0.8, 0.82, 0.83]
quality_dict[103] = [144, 0.76, 0.66, -0.01]
quality_dict[104] = [145, 1.11, 1.1, 1.1]
quality_dict[105] = [146, 0.84, 0.84, 1]
quality_dict[106] = [148, 1.03, 1.03, 1.06]
quality_dict[107] = [149, 0.52, 0.58, 0.49]
quality_dict[108] = [150, 0.75, 0.66, 0.49]
quality_dict[109] = [151, 0.81, 0.29, 0.85]
quality_dict[110] = [152, 1.05, 0.9, 1.22]

quality_dict[111] = [153, 0.93, 1.15, 0.79]
quality_dict[112] = [154, 0.82, 0.7, 0.75]
quality_dict[113] = [155, 0.75, 0.94, 0.71]
quality_dict[114] = [156, 0.75, 0.76, 0.7]
quality_dict[115] = [157, 0.58, 0.68, 0.44]
quality_dict[116] = [158, 0.66, 0.86, 0.05]
quality_dict[117] = [160, 0.54, 0.66, 0.59]
quality_dict[118] = [161, 0.89, 0.83, 0.86]
quality_dict[119] = [162, 1.07, 1, 0.97]
quality_dict[120] = [163, 0.79, 0.42, 0.61]

quality_dict[121] = [164, 0.94, 0.85, 0.71]
quality_dict[122] = [165, 1.07, 0.96, 1.02]
quality_dict[123] = [166, 0.71, 0.93, 0.77]
quality_dict[124] = [167, 0.57, 0.76, 0.57]
quality_dict[125] = [169, 1.11, 0.82, 0.91]
quality_dict[126] = [170, 0.77, 0.85, 0.95]
quality_dict[127] = [171, 0.69, 0.46, 0.48]
quality_dict[128] = [172, 0.45, 0.56, 0.53]
quality_dict[129] = [173, 0.89, 0.84, 0.97]
quality_dict[130] = [174, 1.14, 1.03, 1.17]

quality_dict[131] = [175, 0.69, 0.62, 0.71]
quality_dict[132] = [176, 0.75, 0.73, 0.68]
quality_dict[133] = [178, 0.38, 0.68, 0.55]
quality_dict[134] = [179, 2.34, 0.83, 0.78]
quality_dict[135] = [180, 0.8, 0.64, 0.84]
quality_dict[136] = [182, 0.72, 0.93, 0.9]
quality_dict[137] = [183, 0.35, 0.25, 0.36]
quality_dict[138] = [184, 0.49, 0.93, 0.87]
quality_dict[139] = [185, 1.03, 1.1, 1.1]
quality_dict[140] = [186, 0.73, 0.69, 0.99]

quality_dict[141] = [188, 0.85, 1.78, 0.71]
quality_dict[142] = [189, 0.9, 0.68, 0.92]
quality_dict[143] = [190, 1.13, 0.8, 0.99]
quality_dict[144] = [191, 1.23, 1.1, 0.85]
quality_dict[145] = [192, 0.85, 0.87, 0.8]
quality_dict[146] = [193, 0.76, 0.53, 0.63]
quality_dict[147] = [195, 0.91, 1.1, 0.49]
quality_dict[148] = [196, 0.88, 0.74, 0.68]
quality_dict[149] = [197, 0.9, 1.06, 1.33]
quality_dict[150] = [198, 1.12, 1.07, 1.02]

quality_dict[151] = [199, 0.81, 0.96, 0.83]
quality_dict[152] = [200, 0.3, 0.86, 1.02]
quality_dict[153] = [201, 0.68, 0.69, 0.8]
quality_dict[154] = [203, 0.92, 1.05, 0.86]
quality_dict[155] = [205, 0.91, 0.82, 0.77]
quality_dict[156] = [206, 0.69, 0.84, 0.67]
quality_dict[157] = [207, 0.75, 0.7, 0.2]
quality_dict[158] = [208, 1.47, 0.95, 0.92]
quality_dict[159] = [209, 0.76, 0.67, 0.7]
quality_dict[160] = [210, 0.72, 0.71, 0.78]

quality_dict[161] = [211, 1.4, 0.73, 0.89]
quality_dict[162] = [212, 0.74, 0.56, 0.59]
quality_dict[163] = [213, 1.16, 0.91, 0.67]
quality_dict[164] = [214, 0.46, 0.72, 0.24]
quality_dict[165] = [215, 0.62, 0.69, 0.81]
quality_dict[166] = [216, 0.33, 0.33, 0.37]
quality_dict[167] = [217, 0.69, 1.26, 0.8]
quality_dict[168] = [218, 0.82, 0.99, 0.89]
quality_dict[169] = [219, 1.02, 1.05, 0.84]
quality_dict[170] = [220, 0.63, 0.65, 0.66]

quality_dict[171] = [221, 0.43, 0.78, 0.6]
quality_dict[172] = [222, 0.92, 0.87, 0.85]
quality_dict[173] = [223, 0.81, 0.08, 0.98]
quality_dict[174] = [224, 0.85, 1.03, 0.75]
quality_dict[175] = [226, 0.84, 1.04, 0.44]
quality_dict[176] = [227, 0.99, 0.88, 0.94]
quality_dict[177] = [228, 1.06, 1, 0.93]
quality_dict[178] = [229, 1, 1.09, 0.98]
quality_dict[179] = [230, 0.59, 0.72, 0.81]
quality_dict[180] = [231, 1.28, 1.46, 0.98]

quality_dict[181] = [232, 0.92, 1.21, 0.87]
quality_dict[182] = [233, 0.89, 0.86, 0.67]
quality_dict[183] = [234, 0.62, 0.81, 1.02]
quality_dict[184] = [235, 0.94, 1.08, 0.97]
quality_dict[185] = [237, 0.79, 1.19, 1.42]
quality_dict[186] = [239, 0.81, 0.8, 0.7]
quality_dict[187] = [240, 0.6, 1.09, 0.93]
quality_dict[188] = [241, 0.68, 0.58, 0.52]
quality_dict[189] = [242, 0.59, 0.7, 0.69]
quality_dict[190] = [243, 1.09, 0.84, 0.91]

quality_dict[191] = [244, 0.73, 0.79, 0.68]
quality_dict[192] = [245, 0.56, 0.54, 7.13]
quality_dict[193] = [246, 1.23, 1.85, 0.63]
quality_dict[194] = [247, 0.8, 0.66, 0.54]
quality_dict[195] = [248, 0.14, 0.65, 0.69]
quality_dict[196] = [250, 0.79, 0.84, 0.77]
quality_dict[197] = [251, 0.99, 0.95, 0.99]
quality_dict[198] = [252, 0.78, 0.38, 10.22]
quality_dict[199] = [253, 0.8, 0.89, 0.92]
quality_dict[200] = [254, 0.51, 0.84, 0.75]

quality_dict[201] = [256, 0.95, 0.72, 1.25]
quality_dict[202] = [257, 0.63, 0.69, 0.87]
quality_dict[203] = [259, 0.59, 0.62, 0.67]
quality_dict[204] = [403, 0.92, 0.92, 0.92]
quality_dict[205] = [404, 1.4, 1, 0.96]
quality_dict[206] = [405, 0.72, 0.79, 0.96]
quality_dict[207] = [406, 0.28, 0.45, 0.54]
quality_dict[208] = [407, 0.84, 0.82, 0.76]
quality_dict[209] = [409, 0.84, 1, 0.89]
quality_dict[210] = [410, 0.94, 0.91, 0.9]

quality_dict[211] = [411, 1.09, 0.92, 1.05]
quality_dict[212] = [412, 0.95, 0.7, 0.99]
quality_dict[213] = [413, 0.74, 0.67, 0.63]
quality_dict[214] = [414, 0.93, 1.34, 1.4]
quality_dict[215] = [415, 1.15, 1.38, 1.19]
quality_dict[216] = [416, 0.96, 0.94, 1.01]
quality_dict[217] = [417, 1.12, 1.32, 1.38]
quality_dict[218] = [418, 0.96, 0.87, 1.06]
quality_dict[219] = [419, 1.13, 1, 0.81]


## Select all segments with positive quality ##

# most of data have three segments, only few of them have some negative quality segments
# -> just throw away those segments
not_choose = []
for subject, segment in quality_dict.items():
  for seg, quality in enumerate(quality_dict[subject]):
    if quality < 0:
      not_choose.append((subject, seg))

#print(not_choose)
#print(len(not_choose))

## Set up Pytorch Dataloader ##

class BPDataset(data.Dataset):
    
    def __init__(self, ppg_dir, label_path, normalize):

        self.ppg_dir = ppg_dir
        self.label_path = label_path
        self.data = []
        self.label = []
        self.subjectid = []
        self.normalize=normalize
        
        # read ppg data
        for subject in range(1, 220):
          subjectid = quality_dict[subject][0]
          for segnum in range(1, 4):
            if (subject, segnum) not in [(7, 3), (23, 3), (83, 3), (88, 2), (89, 2), (103, 3), (180, 1), (180, 2), (180, 3)]:
              ppg_path = os.path.join(ppg_dir, '{}_{}.txt'.format(subjectid, segnum))
              if os.path.exists(ppg_path):
                with open(ppg_path) as f:
                  lines = f.readlines()[0].split('\t')[:-1]
                  if len(lines) != 2100:
                    print(subject, subjectid, segment, len(lines))
                    continue
                  ppg = torch.Tensor([float(x) for x in lines])
                  ppg = ppg.reshape((1,2100))
                  self.data.append(ppg)
                  self.subjectid.append(subjectid)
        
        # read BP labels (Label: 'Stage 1 hypertension' or 'Stage 2 hypertension'-> 2; 'Prehypertension'-> 1; 'Normal'-> 0)
        df = pd.read_csv(label_path, skiprows=1)
        print(set(df["Hypertension"]))
        for subject in range(219):
          if df['Hypertension'][subject] == 'Stage 1 hypertension' or df['Hypertension'][subject] == 'Stage 2 hypertension':
            bp_label = 2
          elif df['Hypertension'][subject] == 'Prehypertension':
            bp_label = 1
          elif df['Hypertension'][subject] == 'Normal':
            bp_label = 0
          
          if subject+1 == 180:
            continue
          elif subject+1 in [7, 23, 83, 88, 89, 103]:
            self.label.extend([bp_label]*2)
          else:
            self.label.extend([bp_label]*3)
        
        self.label = torch.Tensor(self.label)
        self.label = self.label.type(torch.long)
       
    def median_filter(tensor_data):
        x,y = tensor_data.size()
        new_data = torch.zeros(x,y)
        for i in range(x):
            for j in range(y):
                if j == 0:
                    new_data[i][j] = tensor_data[i][j]
                elif j < 11:
                    new_data[i][j] = tensor_data[i][0: 2 * j + 1].median()
                elif 11 <= j < y - 11:
                    new_data[i][j] = tensor_data[i][j - 11: j + 12].median()
                elif y - 11 <= j < y - 1:
                    new_data[i][j] = tensor_data[i][2*j - y : y].median()
                elif j == y - 1:
                    new_data[i][j] = tensor_data[i][j]
        return new_data
      
    def roll_filter(tensor_data):
        x,y = tensor_data.size()
        new_data = torch.zeros(x,y)
        for i in range(x):
            for j in range(y):
                if j == 0:
                    new_data[i][j] = tensor_data[i][j]
                elif j < 11:
                    new_data[i][j] = tensor_data[i][0: 2 * j + 1].mean()
                elif 11 <= j < y - 11:
                    new_data[i][j] = tensor_data[i][j - 11: j + 12].mean()
                elif y - 11 <= j < y - 1:
                    new_data[i][j] = tensor_data[i][2*j - y : y].mean()
                elif j == y - 1:
                    new_data[i][j] = tensor_data[i][j]
        return new_data
        
    def __getitem__(self, index):
        data = self.data[index]
        data = roll_filter(median_filter(data)) #preprocessing
        data=(data-self.normalize['mean'])/self.normalize['std'] #normalization
        label = self.label[index]
        subjectid = self.subjectid[index]
        return data, label, subjectid

    def __len__(self):
        return len(self.data)



