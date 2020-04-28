import numpy  as np

def pad_data(df,key,pad_size = 21,data_size = None):
  if data_size is None:
    data_size = df.shape[0]
  data = np.zeros((data_size,pad_size,3),dtype=np.float32)
  for i in range(data_size):
    index = df.index[i]
    s, _ = df[key][index].shape
    data[i,:s,:3] = df[key][index]
  return data

# convert np.array2string into np.array
# adapt from https://stackoverflow.com/questions/45704999/
def str2numpy(input_str, shape = None):
  data = np.fromstring(input_str.replace('\n','').replace('[','').replace(']','').replace('  ',' '), sep=' ')
  if shape is not None:
    data = data.reshape(shape)
  else:
    #use auto reshape, It's work with only 1d or 2d array
    lines = input_str.strip().split('\n')
    data = data.reshape((len(lines),-1))
  return data