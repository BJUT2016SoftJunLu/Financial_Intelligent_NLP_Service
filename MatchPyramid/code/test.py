#coding=utf-8


def main():

    Y_label = []
    with open("../data/test_data_one",'r') as f:
        for line in f.readlines():
            example = line.split("\t")
            Y_label.append(int(example[-1]))

    predict = []
    with open("../data/test_output",'r') as f:
        for line in f.readlines():
            example = line.split("\t")
            predict.append(int(example[-1]))


    right = 0.0
    for i in range(len(Y_label)):
        if Y_label[i] == predict[i]:
            right += 1.0

    print(right/len(Y_label))

    return


if __name__ == '__main__':
    main()