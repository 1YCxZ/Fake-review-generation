import re
import os
import pickle
import jieba.analyse
import jieba.posseg as pseg
import math
import config


FH_PUNCTUATION = [
    (u"。", u"."), (u"，", u","), (u"！", u"!"), (u"？", u"?"), (u"～", u"~"),
]

keep_p = ['，', '。', '！', '？', '～', '、']
f2h = {}
for item in FH_PUNCTUATION:
    c1 = item[0]
    c2 = item[1]
    f2h[c2] = c1


def convert(content):
    nc = []
    for c in content:
        if c in f2h:
            nc.append(f2h[c])
            continue
        nc.append(c)
    return "".join(nc)


def clean(line):
    if line == "":
        return
    line = convert(line)
    c_content = []
    for char in line:
        if re.search("[\u4e00-\u9fa5]", char):
            c_content.append(char)
        elif re.search("[a-zA-Z0-9]", char):
            c_content.append(char)
        elif char in keep_p:
            c_content.append(char)
        elif char == ' ':  # 很多用户喜欢用空格替代标点
            c_content.append('，')
        else:
            c_content.append('')
    nc_content = []
    c = 0
    for char in c_content:
        if char in keep_p:
            c += 1
        else:
            c = 0
        if c < 2:
            nc_content.append(char)
    result = ''.join(nc_content)
    result = result.strip()
    result = result.lower()  # 所有英文转成小写字母
    return result


def clean_comment(text):
    """
    对原始评论进行清理，删去非法字符，统一标点，删去无用评论
    """
    comment_set = []
    for line in text:
        line = line.lstrip()
        line = line.rstrip()
        line = clean(line)
        if len(line) < 7:  # 过于短的评论需要删除
            continue
        if line and line not in ['该用户没有填写评论。', '用户晒单。']:
            comment_set.append(line)

    return comment_set


def get_seg_pos(text, type='word'):
    """
    获取文档的分词以及词性标注结果，分词的方式可以为按词切分或者按字切分
    """
    if type == 'word':
        seg_pos = []
        for line in text:
            line = line.strip()
            line_cut = pseg.cut(line)
            wordlist = []
            for term in line_cut:
                wordlist.append('%s/%s' % (term.word, term.flag))
            seg_pos.append(' '.join(list(wordlist)))

        return seg_pos


def save_to_pickle(object, path):
    """
    将python对象保存至pickle
    """
    with open(path, 'wb') as f:
        pickle.dump(object, f, pickle.HIGHEST_PROTOCOL)


def caculate_word_idf(doc_folder, stop_word):
    """
    计算所有文档中的每个词的idf
    doc_folder: str, 文件夹路径
    stop_word: list, 停用词list

    return: 所有词的idf值
    """
    num_doc = 0  # 商品数量
    word_IDF = {}  # word-IDF 记录每个word在不同的doc出现过的次数,然后计算IDF
    for _, _, files in os.walk(doc_folder):
        num_doc = len(files)
        for file in files:
            cur_doc_word_set = set()  # 记录当前文档中出现的不同的词
            doc_path = os.path.join(doc_folder, file)
            for _, line in enumerate(open(doc_path, 'r')):
                line = line.strip()
                word_list = line.split(" ")
                word_list = [term.split('/')[0] for term in word_list]
                for w in word_list:
                    # 如果这个词在停用词表中就不添加
                    if w in stop_word:
                        continue
                    cur_doc_word_set.add(w)
            for w in cur_doc_word_set:
                if w in word_IDF:
                    word_IDF[w] += 1
                else:
                    word_IDF[w] = 1
    for w in word_IDF:
        word_IDF[w] = math.log10(num_doc / word_IDF[w])
    return word_IDF


if __name__ == "__main__":

    stop_word = []
    with open(config.STOP_WORD_FILE, 'r') as f:
        for line in f.readlines():
            stop_word.append(line.strip())

    jieba.load_userdict('%s/user_dict.txt' % config.RESOURCES_FOLD)

    # 繁体转为简体
    print('繁体转为简体...')
    for _, _, files in os.walk(config.RAW_DATA_FOLD):
        for file in files:
            doc_path = os.path.join(config.RAW_DATA_FOLD, file)
            os.system('opencc -i %s -o %s -c t2s.json' % (doc_path, doc_path))
    print('\tDone')

    # 清洗文本
    print('开始清洗文本...')
    for _, _, files in os.walk(config.RAW_DATA_FOLD):
        for file in files:
            if 'txt' not in file:
                continue
            doc_path = os.path.join(config.RAW_DATA_FOLD, file)
            with open(doc_path, 'r') as f_r:
                text = f_r.readlines()
            comment_set = clean_comment(text)
            print('---process %s ---' % doc_path)
            with open(os.path.join(config.CLEAN_DATA_FOLD, file), 'w') as f_w:
                for i in comment_set:
                    f_w.write(i+'\n')
    print('\tDone')

    # 对清洗后的文本进行分词，词性标注
    print('开始进行分词，词性标注...')
    for _, _, files in os.walk(config.CLEAN_DATA_FOLD):
        for file in files:
            doc_path = os.path.join(config.CLEAN_DATA_FOLD, file)
            with open(doc_path, 'r') as f_r:
                text = f_r.readlines()
            print('---process %s ---' % doc_path)
            seg_pos = get_seg_pos(text)
            with open(os.path.join(config.SEG_POS_FOLD, file), 'w') as f_w:
                for i in seg_pos:
                    f_w.write(i+'\n')
    print('\tDone')

    # 计算除去停用词的每个词的idf值
    word_idf = caculate_word_idf(config.SEG_POS_FOLD, stop_word)
    with open(config.IDF_FILE, 'w', encoding='utf-8') as f:
        for word in word_idf:
            f.write('%s %s\n' % (word, word_idf[word]))