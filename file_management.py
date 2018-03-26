import random
import os
import shutil



def SelectItemsFromDir(dir,num =2):
    temp = random.sample((os.listdir(dir)), num)
    random.shuffle(temp)
    return temp


def MakeDir(path):
    try:
        shutil.rmtree(path)
        os.makedirs(path)
        print "[INFO] Replaced with new file at path: ",path
    except Exception, e:
        os.makedirs(path)
        print "[INFO] New file made. at path: ", path

def Copy(src,dst):
    shutil.copy(src, dst)
    #print "[INFO] Copy of ",src," to ",dst, " is successful"

def CopyAllToOne(list,dst,log=None,label=None):
    if log!=None and label!=None:
        for each in list:
            Copy(each,dst)
            st = label+","+FindFileName(each).replace('.png','')[2:]+","+FindFileName(each)[0]
            log.writeln(st)
    else:
        for each in list:
            Copy(each,dst)

def FindFileName(st):
    st_arr = st.split('/')
    fname = st_arr[len(st_arr)-1]
    return fname


def PeekForClass(list):
    item = list[0]
    return item[0]

def SingleRangeSelector(name_list,st,end):
    ret_list = []
    for each in name_list:
        temp = each.replace('.png','')
        temp = int(temp[2:])
        if temp >= st and temp <= end:
            ret_list.append(each)
    return ret_list

def MultipleRangeSelector(name_list,st,end):
    ret_list = []
    if type(st)!=list or type(end) != list or len(st)!=len(end):
        raise Exception(" [ERR] Ranges are supposed to be of list type")
    else:
        range_len = len(st)
        for i in range(range_len):
            ret_list.append([])
        for each in name_list:
            temp = each.replace('.png','')
            temp = int(temp[2:])
            for i in range(range_len):
                if temp >= st[i] and temp <= end[i]:
                    ret_list[i].append(each)
    return ret_list

def PathMaker(str,list):
    #list of .png items
    temp = []
    for e in list:
        if str[len(str)-1]=="/":
            st = str + e[0] + "/"+ e
        else:
            st = str + "/" + e[0] + "/" + e
        temp.append(st)
    return temp

class Logger:

    def __init__(self,filename="random_name_made"):
        self.fn = filename
        try:
            self.Removefile()
        except:
            pass

    def MakeTruncatefile(self):
        with open(self.fn,'w') as f:
            f.write('')

    def Removefile(self):
        os.remove(self.fn)

    def write(self,str):
        with open(self.fn,"a") as f:
            f.write(str)

    def writeln(self,str):
        with open(self.fn,"a") as f:
            f.write(str+"\n")