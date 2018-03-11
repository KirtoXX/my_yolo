# my_yolo
Yolov2 implement by keras and TensorFlow </p>
## Detail of Cnns
1.DenseNet121 pre-trained model</p>
2.down-sample:32  </p>
3.input shape:416x416 output shape:13x13 </p>
## Detail of Anchor
1.set 5 anchorbox by hand</p>
2.anchor_size = lenght/weight = [1,2,3,0.5,0.3]</p>
## exchange from output to real bbox
assert output[3,4,3,:] = [0.3,0.4,1.2,1.5,0.9,1,0,0]</p>
anchorbox = 3/1</p>
x = 3x32+0.3x32</p>
y = 3x32+0.4x32</p>
w = 3xe^1.2</p>
h = 1xe^1.5</p>
confidence = 0.9</p>
class = 1</p>
