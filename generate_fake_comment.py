from comment_generate_utils import *
import config

# 利用抽取的方法从原始评论中抽取含有观点表达的短句，并随机组合一些含有观点表达的短句组成一句仿真评论。
#########
PRODUCTID = 2227311  # 修改productid
#########

with open('%s/%s.txt' % (config.CLEAN_DATA_FOLD, PRODUCTID), 'r') as f:
    str_text = f.read()
with open('%s/%s.txt' % (config.SEG_POS_FOLD, PRODUCTID), 'r') as f:
    seg_pos_text = f.readlines()

# 加载IDF表
word_idf = {}
with open(config.IDF_FILE, 'r') as f:
    for line in f.readlines():
        word, idf = line.strip().split(' ')
        word_idf[word] = float(idf)

# 加载停用词
stop_word = []
with open(config.STOP_WORD_FILE, 'r') as f:
    for line in f.readlines():
        stop_word.append(line.strip())

# 加载正向情感词典
pos_adj_word = []
with open(config.POS_ADJ_WORD_FILE, 'r') as f:
    for line in f.readlines():
        pos_adj_word.append(line.strip())

seg_list, pos_list, seg_comment_list = text2seg_pos(seg_pos_text, pattern='[。！？，～]')
raw_aspect_list = get_candidate_aspect(seg_list, pos_list, pos_adj_word, stop_word, word_idf)

# 构建候选集合
N = NSDict(seg_list, pos_list, raw_aspect_list)
ns_dict = N.build_nsdict()
# 候选集合排序
P = PairPattSort(ns_dict)
pair_score = P.sort_pair()

# 得到正确的观点表达候选
pair_useful = {}
baseline = 0.1 * len(pair_score)
for i, item in enumerate(pair_score):
    if i <= baseline:
        aspect, opinion = item[0].split('\t')
        if aspect in pair_useful:
            pair_useful[aspect].append(opinion)
        else:
            pair_useful[aspect] = [opinion]

# 从原始评论中抽取观点表达
aspect_express = get_aspect_express(seg_comment_list, pair_useful)

# 字符匹配合并aspect
merged_aspect_express, opinion_set = merge_aspect_express(aspect_express, pair_useful)

# 生成假评论
generated_raw_comments = generate_comments(merged_aspect_express)

results = fake_comment_filter(generated_raw_comments, opinion_set)

with open('%s/%s_generated_comments.txt' % (config.RESULTS_DATASET_FOLD, PRODUCTID), 'w') as f:
    for c in results:
        f.write(c + '\n')
