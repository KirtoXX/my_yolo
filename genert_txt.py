import os

def main():
    # generate_label_1.py
    # encoding:utf-8
    file = open('F:/object-detection-crowdai/labels.txt', 'r')  # 原始labels.txt的地址
    for eachline in file:
        data = eachline.strip().split(',')
        filename = data[4]
        filename = filename[:-4]
        txt_path = 'F:/object-detection-crowdai/Labels' + filename + '.txt'  # 生成的txt标注文件地址
        with open(txt_path, 'a') as txt:
            new_line=data[5]+' '+data[0]+' '+data[1]+' '+data[2]+' '+data[3]
            txt.writelines(new_line)
            txt.write('\n')
    file.close()
    print('generate label success')


if __name__ == '__main__':
    main()