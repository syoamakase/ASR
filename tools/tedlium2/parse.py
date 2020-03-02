import codecs

word = ''
with codecs.open("predict_2.txt", "w", "utf-8") as fout:
    with codecs.open("predict.txt", "r", "utf-8") as f:
        for line in f:
            if (("<eos>" in line) or ("<sos>" in line) or ("<sp>" in line)):
                continue
            else:
                sp_line = line.split('+')

                #print(sp_line[0])
                if '@@' in sp_line[0]:
                    #continue_flag = True
                    word += sp_line[0].strip().replace('@@', '')
                else:
                    word += sp_line[0].strip()
                    #continue_flag = False
                    fout.write(word+"\n")
                    word = ''
