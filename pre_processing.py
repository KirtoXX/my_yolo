import numpy as np
from scipy import misc
import anchor_box

anchor_boxs = anchor_box.anchor_box
classs = 3
down_sample = 32
input_shape = 416
output_shape = int(input_shape/down_sample)

def choose_anchor(anchor):
    t = np.ones(5)*anchor - anchor_boxs
    t = np.fabs(t)
    #print(t)
    id = np.argmin(t)
    return id

def get_lable(id):
    f = open('data/lable/'+str(id)+'.txt')
    split_lines = f.readlines()
    result = []
    for line in split_lines:
        c,x1,y1,x2,y2 = line.split()
        x1 = int(x1)
        x2 = int(x2)
        y1 = int(y1)
        y2 = int(y2)
        x = int((x1+x2)/2)
        y = int((y1+y2)/2)
        w = int((x2-x1))
        h = int((y2-y1))
        temp = [c,x,y,w,h]
        result.append(temp)
    return result

def one_hot_lable(name):
    result = []
    if name=='Car':
        result = [1,0,0]
    elif name=='Pedestrian':
        result = [0,1,0]
    elif name=='Truck':
        result = [0,0,1]
    return result

#Emmmmmmmm
def trans_pos(x,y,w,h,c):
    anchor_id = choose_anchor(x/y)
    rate = anchor_boxs[anchor_id]
    result = []
    #anchor example anchor=lenght/height 2/1=1
    #grid_cell int
    px = int(x/down_sample)
    py = int(y/down_sample)
    #position in the grid cell(0,1)
    tx = (x - px*down_sample)/down_sample
    ty = (y - py*down_sample)/down_sample
    #confience
    confidence = 1.
    #size of anchor
    tw = np.log(w/rate)
    th = np.log(h/1.)
    #onehot of class
    tc = one_hot_lable(c)
    #union
    result.append(tx)
    result.append(ty)
    result.append(tw)
    result.append(th)
    result.append(confidence)
    result.extend(tc)

    return px,py,anchor_id,result

def build_data(id,size=[416,416]):
    lable = np.zeros([13,13,5,8])
    img_path = 'data/img/'+str(id)+'.jpg'
    img = misc.imread('data/img/'+str(id)+'.jpg')
    img_h,img_w,_ = img.shape
    #print(img_w,img_h)
    img = misc.imresize(img,size=size)
    #get resize rate
    w_rate,h_rate = np.divide(size,[img_w,img_h])
    #print(w_rate,h_rate)
    #get info of img
    img_info = get_lable(id)
    for info in img_info:
        c,x,y,w,h = info
        #resize x,y,w,h
        #print(c, x, y, w, h)
        x = x*w_rate
        y = y*h_rate
        w = w*w_rate
        h = h*h_rate
        #print(x,y,w,h)
        #
        px,py,anchor_id,temp = trans_pos(x,y,w,h,c)
        #print(px,py,temp)
        lable[px,py,anchor_id,:] = temp
    lable = lable.astype(np.float32)
    return img,lable

def main():
    id = 1479498371963069978
    img,lable = build_data(id)
    print(img.shape,lable.shape)

if __name__ == '__main__':
    main()


