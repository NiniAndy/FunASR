import numpy as np


stage = 4

# Gen.1 Generate a txt file from the original data file
if stage ==1:
    path = "/ssd/zhuang/code/ASR_baseline/data/aishell/A_data/dev/text"
    save_path = "/ssd/zhuang/CodeStorage/ChineseCorrection/data/dev.txt"

    text_list = []
    with open(path, "r", encoding="utf-8") as f:
        i = 0
        for line in f:
            if len(line) == 0:
                continue
            else:
                text = line.split(" ", 1)[1].strip()
                text_list.append(text)

    with open(save_path, "w", encoding="utf-8") as f:
        for text in text_list:
            f.write(text + "\n")

    print ("Done!")

# Gen.2 Generate from wer
elif stage ==2:
    path = "/ssd/zhuang/code/wenet2024/examples/aishell/paraformer/exp/paraformer/avg_5_maxepoch_100.pt_chunk-1_ctc0.3_reverse0.5/paraformer_greedy_search/wer"
    with open(path, "r", encoding="utf-8") as f:
        output_list = []
        lab_list= []
        rec_list = []
        for line in f:
            line = line.strip()
            if len(line) < 2:
                continue
            if line.startswith("WER"):
                continue
            if line.startswith("utt:"):
                continue
            if line.startswith("==="):
                continue
            if line.startswith("Overall"):
                continue
            if line.startswith("Mandarin"):
                continue
            if line.startswith("Other"):
                continue
            if line.startswith("lab"):
                lab = line.split(" ", 1)[1]
                lab = lab.replace(" ", "")
                lab_list.append(lab)
            if line.startswith("rec"):
                rec = line.split(" ", 1)[1]
                rec = rec.replace(" ", "")
                rec_list.append(rec)

    length = len(lab_list)
    pickup_num =  0.1 * length
    random_index = np.random.permutation(length)
    pickup_index = random_index[:int(pickup_num)]


    save_path = "/ssd/zhuang/CodeStorage/ChineseCorrection/data/paraformer_text/"
    with open(save_path+"dev_lab.txt", "w", encoding="utf-8") as f:
        for text in lab_list:
            f.write(text + "\n")

    with open(save_path+"dev_rec.txt", "w", encoding="utf-8") as f:
        for text in rec_list:
            f.write(text + "\n")

    with open(save_path+"dev_lab.txt", "w", encoding="utf-8") as f:
        for i in pickup_index:
            f.write(lab_list[i] + "\n")

    with open(save_path+"dev_rec.txt", "w", encoding="utf-8") as f:
        for i in pickup_index:
            f.write(rec_list[i] + "\n")

    print ("Done!")



# Gen.3 Mingling two txt files
elif stage == 3:
    path_1_1 = "/ssd/zhuang/CodeStorage/ChineseCorrection/data/paraformer_text/dev_lab.txt"
    path_1_2 = "/ssd/zhuang/CodeStorage/ChineseCorrection/data/paraformer_text/dev_rec.txt"

    path_2_1 = "/ssd/zhuang/CodeStorage/ChineseCorrection/data/qian_text/train_lab.txt"
    path_2_2 = "/ssd/zhuang/CodeStorage/ChineseCorrection/data/qian_text/train_rec.txt"

    save_path = "/ssd/zhuang/CodeStorage/ChineseCorrection/data/mingle_text/"


    mingle_lab_list = []
    mingle_rec_list = []

    with open(path_1_1, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            mingle_lab_list.append(line)

    with open(path_1_2, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            mingle_rec_list.append(line)

    train_list = []
    clean_train_list = []
    predict_list = []
    clean_predict_list = []

    with open(path_2_1, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            train_list.append(line)

    with open(path_2_2, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            predict_list.append(line)

    for i in range(len(train_list)):
        if len(train_list[i]) != len(predict_list[i]):
            continue
        else:
            clean_train_list.append(train_list[i])
            clean_predict_list.append(predict_list[i])


    mingle_lab_list = mingle_lab_list + clean_train_list
    mingle_rec_list = mingle_rec_list + clean_predict_list

    with open(save_path+"dev_lab.txt", "w", encoding="utf-8") as f:
        for text in mingle_lab_list:
            f.write(text + "\n")

    with open(save_path+"dev_rec.txt", "w", encoding="utf-8") as f:
        for text in mingle_rec_list:
            f.write(text + "\n")

    print  ("Done!")

# Gen.4 Check each utterance length
if stage == 4:
    train_path = ["/ssd/zhuang/CodeStorage/ChineseCorrection/data/paraformer_text/dev_lab.txt",
          "/ssd/zhuang/CodeStorage/ChineseCorrection/data/paraformer_text/dev_rec.txt"]


    lab = []
    rec = []

    checked_lab = []
    checked_rec = []


    with open(train_path[0], "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            lab.append(line)

    with open(train_path[1], "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            rec.append(line)



    for i in range(len(lab)):
        if len(lab[i]) != len(rec[i]):
            print (lab[i])
            print (rec[i])
            print ("------------------")
            continue
        else:
            checked_lab.append(lab[i])
            checked_rec.append(rec[i])

    with open(train_path[0], "w", encoding="utf-8") as f:
        for text in checked_lab:
            f.write(text + "\n")

    with open(train_path[1], "w", encoding="utf-8") as f:
        for text in checked_rec:
            f.write(text + "\n")

    print  ("Done!")

