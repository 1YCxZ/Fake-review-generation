import config
import math

WINDOW_SIZE = 5
PUNCTUATION_MARK = ['x']  # 标点
NOUN_MARK = ['n', 'ng', 'nr', 'nrfg', 'nrt', 'ns', 'nt', 'nz']  # 名词
VERB_MARK = ['v', 'vd', 'vg', 'vi', 'vn', 'vq']  # 动词
ADJECTIVE_MARK = ['a', 'ad', 'an', 'ag']  # 形容词
ADVERB_MARK = ['d', 'df', 'dg']  # 副词
ENG_MARK = ['eng']

RESERVED_MARK = NOUN_MARK + VERB_MARK + ADJECTIVE_MARK + ADVERB_MARK + ENG_MARK # 用于发现新词


def text2comment(seg_pos_text):
    """
    经过分词的文档，得到原始用户的每条评论
    """
    comment_list = []  # 保存全部的按照指定标点切分的句子
    all_word = set()  # 全部单词
    for line in seg_pos_text:
        line = line.strip()
        line = line.split(' ')
        cur_comment = []
        for term in line:
            word, flag = term.split('/')
            cur_comment.append(word)
            if flag in RESERVED_MARK:
                all_word.add(word)
        comment_list.append(cur_comment)

    return comment_list, all_word


def find_word_phrase(all_word, seg_list):
    """
    根据点互信息以及信息熵发现词组，主要目的是提升分词效果
    """
    word_count = {k: 0 for k in all_word}  # 记录全部词出现的次数

    all_word_count = 0
    all_bi_gram_count = 0
    for sentence in seg_list:
        all_word_count += len(sentence)
        all_bi_gram_count += len(sentence) - 1
        for idx, word in enumerate(sentence):
            if word in word_count:
                word_count[word] += 1

    bi_gram_count = {}
    bi_gram_lcount = {}
    bi_gram_rcount = {}
    for sentence in seg_list:
        for idx, _ in enumerate(sentence):
            left_word = sentence[idx - 1] if idx != 0 else ''
            right_word = sentence[idx + 2] if idx < len(sentence) - 2 else ''

            first = sentence[idx]
            second = sentence[idx + 1] if idx + 1 < len(sentence) else ''
            if first in word_count and second in word_count:
                if (first, second) in bi_gram_count:
                    bi_gram_count[(first, second)] += 1
                else:
                    bi_gram_count[(first, second)] = 1
                    bi_gram_lcount[(first, second)] = {}
                    bi_gram_rcount[(first, second)] = {}

                if left_word in bi_gram_lcount[(first, second)]:
                    bi_gram_lcount[(first, second)][left_word] += 1
                elif left_word != '':
                    bi_gram_lcount[(first, second)][left_word] = 1

                if right_word in bi_gram_rcount[(first, second)]:
                    bi_gram_rcount[(first, second)][right_word] += 1
                elif right_word != '':
                    bi_gram_rcount[(first, second)][right_word] = 1

    bi_gram_count = dict(filter(lambda x: x[1] >= 5, bi_gram_count.items()))

    bi_gram_le = {}  # 全部bi_gram的左熵
    bi_gram_re = {}  # 全部bi_gram的右熵
    for phrase in bi_gram_count:
        le = 0
        for l_word in bi_gram_lcount[phrase]:
            p_aw_w = bi_gram_lcount[phrase][l_word] / bi_gram_count[phrase]  # P(aW | W)
            le += p_aw_w * math.log2(p_aw_w)
        le = -le
        bi_gram_le[phrase] = le

    for phrase in bi_gram_count:
        re = 0
        for r_word in bi_gram_rcount[phrase]:
            p_wa_w = bi_gram_rcount[phrase][r_word] / bi_gram_count[phrase]  # P(Wa | W)
            re += p_wa_w * math.log2(p_wa_w)
        re = -re
        bi_gram_re[phrase] = re

    PMI = {}
    for phrase in bi_gram_count:
        p_first = word_count[phrase[0]] / all_word_count
        p_second = word_count[phrase[1]] / all_word_count
        p_bi_gram = bi_gram_count[phrase] / all_bi_gram_count
        PMI[phrase] = math.log2(p_bi_gram / (p_first * p_second))

    phrase_score = []
    for phrase in PMI:
        le = bi_gram_le[phrase]
        re = bi_gram_re[phrase]
        score = PMI[phrase] + le + re
        phrase_score.append((phrase, score))

    phrase_score = sorted(phrase_score, key=lambda x: x[1], reverse=True)

    for item in phrase_score:
        print('{}:{}'.format(''.join(item[0]), item[1]))

    print()


if __name__ == '__main__':
    ######
    PRODUCTID = 279619  # 修改商品ID以得到不同商品发现的新词
    ######

    with open('%s/%s.txt' % (config.CLEAN_DATA_FOLD, PRODUCTID), 'r') as f:
        str_text = f.read()
    with open('%s/%s.txt' % (config.SEG_POS_FOLD, PRODUCTID), 'r') as f:
        seg_pos_text = f.readlines()

    # 加载停用词
    stop_word = []
    with open(config.STOP_WORD_FILE, 'r') as f:
        for line in f.readlines():
            stop_word.append(line.strip())

    # 加载words_idf值
    word_idf = {}
    for idx, line in enumerate(open(config.IDF_FILE, 'r')):
        line = line.strip()
        line = line.split(' ')
        word_idf[line[0]] = float(line[1])

    comment_list, all_word = text2comment(seg_pos_text)

    find_word_phrase(all_word, comment_list)

