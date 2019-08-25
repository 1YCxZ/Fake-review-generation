RAW_DATA_FOLD = 'raw_review'  # 原始评论保存目录
CLEAN_DATA_FOLD = 'clean_review'  # 经过清洗的文本的保存目录
SEG_POS_FOLD = 'seg_pos'  # 所有文本的分词结果
RESOURCES_FOLD = 'resources'  # 所有资源的存放目录，例如用户词表
STOP_WORD_FILE = '%s/stopword.txt' % RESOURCES_FOLD   # 停用词表
POS_ADJ_WORD_FILE = '%s/HowNetPOSWord.txt' % RESOURCES_FOLD  # 第三方情感词典词表
IDF_FILE = '%s/idf.txt' % RESOURCES_FOLD  # 所有词的IDF值
RESULTS_DATASET_FOLD = 'results'  # 存放抽取式方法生成的仿真评论
