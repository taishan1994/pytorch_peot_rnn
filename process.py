import re
import os
import json
import opencc
import numpy as np

from config import Config


def _parseRawData(author=None, constrain=None, src='./data/tang/', category="poet.tang"):
    """
    code from https://github.com/justdark/pytorch-poetry-gen/blob/master/dataHandler.py
    处理json文件，返回诗歌内容
    @param: author： 作者名字
    @param: constrain: 长度限制
    @param: src: json 文件存放路径
    @param: category: 类别，有poet.song 和 poet.tang

    在此基础上，新增了将繁体转简体，以及存储文件的功能
    返回 data：list
        ['床前明月光，疑是地上霜，举头望明月，低头思故乡。',
         '一去二三里，烟村四五家，亭台六七座，八九十支花。',
        .........
        ]
    """

    def sentenceParse(para):
        # para 形如 "-181-村橋路不端，數里就迴湍。積壤連涇脉，高林上笋竿。早嘗甘蔗淡，
        # 生摘琵琶酸。（「琵琶」，嚴壽澄校《張祜詩集》云：疑「枇杷」之誤。）
        # 好是去塵俗，煙花長一欄。"
        result, number = re.subn(u"（.*）", "", para)
        result, number = re.subn(u"{.*}", "", result)
        result, number = re.subn(u"《.*》", "", result)
        result, number = re.subn(u"《.*》", "", result)
        result, number = re.subn(u"[\]\[]", "", result)
        r = ""
        for s in result:
            if s not in set('0123456789-'):
                r += s
        r, number = re.subn(u"。。", u"。", r)
        return r

    def handleJson(file):
        # print file
        rst = []
        data = json.loads(open(file).read())
        for poetry in data:
            pdata = ""
            if (author is not None and poetry.get("author") != author):
                continue
            p = poetry.get("paragraphs")
            flag = False
            for s in p:
                sp = re.split(u"[，！。]", s)
                for tr in sp:
                    if constrain is not None and len(tr) != constrain and len(tr) != 0:
                        flag = True
                        break
                    if flag:
                        break
            if flag:
                continue
            for sentence in poetry.get("paragraphs"):
                pdata += sentence
            pdata = sentenceParse(pdata)
            if pdata != "":
                rst.append(pdata)
        return rst

    data = []
    cc = opencc.OpenCC('t2s')
    for filename in os.listdir(src):
        if filename.startswith(category):
            peots = handleJson(src + filename)
            for peot in peots:
                peot = cc.convert(peot)
                data.append(peot)
    with open('./data/peot.txt', 'w') as fp:
        fp.write("\n".join(data))
    return data


def pad_sequences(sequences,
                  maxlen=None,
                  dtype='int32',
                  padding='pre',
                  truncating='pre',
                  value=0.):
    """
    code from keras
    Pads each sequence to the same length (length of the longest sequence).
    If maxlen is provided, any sequence longer
    than maxlen is truncated to maxlen.
    Truncation happens off either the beginning (default) or
    the end of the sequence.
    Supports post-padding and pre-padding (default).
    Arguments:
        sequences: list of lists where each element is a sequence
        maxlen: int, maximum length
        dtype: type to cast the resulting sequence.
        padding: 'pre' or 'post', pad either before or after each sequence.
        truncating: 'pre' or 'post', remove values from sequences larger than
            maxlen either in the beginning or in the end of the sequence
        value: float, value to pad the sequences to the desired value.
    Returns:
        x: numpy array with dimensions (number_of_sequences, maxlen)
    Raises:
        ValueError: in case of invalid values for `truncating` or `padding`,
            or in case of invalid shape for a `sequences` entry.
    """
    if not hasattr(sequences, '__len__'):
        raise ValueError('`sequences` must be iterable.')
    lengths = []
    for x in sequences:
        if not hasattr(x, '__len__'):
            raise ValueError('`sequences` must be a list of iterables. '
                             'Found non-iterable: ' + str(x))
        lengths.append(len(x))

    num_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)

    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:  # pylint: disable=g-explicit-length-test
            sample_shape = np.asarray(s).shape[1:]
            break

    x = (np.ones((num_samples, maxlen) + sample_shape) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if not len(s):  # pylint: disable=g-explicit-length-test
            continue  # empty list/array was found
        if truncating == 'pre':
            trunc = s[-maxlen:]  # pylint: disable=invalid-unary-operand-type
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" not understood' % truncating)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError(
                'Shape of sample %s of sequence at position %s is different from '
                'expected shape %s'
                % (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x


def get_data(config):
    if os.path.exists(config.pickle_path):
        data = np.load(config.pickle_path, allow_pickle=True)
        data, word2idx, idx2word = data['data'], data['word2idx'].item(), data['idx2word'].item()
        return data, word2idx, idx2word

    with open(config.data_path, 'r') as fp:
        data = fp.read().split('\n')
        words = {word for sentence in data for word in sentence}
        word2idx = {_word: _ix + 4 for _ix, _word in enumerate(words)}
        # PAD:0 UNK:1 SOP:2（开始标识符） EOP:3（终止标识符）
        word2idx['PAD'] = 0
        word2idx['UNK'] = 1
        word2idx['SOP'] = 2
        word2idx['EOP'] = 3
        idx2word = {_ix: _word for _word, _ix in word2idx.items()}

        # 为每首诗歌加上起始符和终止符
        for i in range(len(data)):
            data[i] = ["SOP"] + list(data[i]) + ["EOP"]

        # 将每首诗歌保存的内容由‘字’变成‘数’
        # 形如[春,江,花,月,夜]变成[1,2,3,4,5]
        new_data = [[word2idx[_word] for _word in _sentence]
                    for _sentence in data]

        # 诗歌长度不够opt.maxlen的在前面补空格，超过的，删除末尾的
        pad_data = pad_sequences(new_data,
                                 maxlen=config.max_len,
                                 padding='post',
                                 truncating='post',
                                 value=0)
        # 保存成二进制文件
        np.savez_compressed(config.pickle_path,
                            data=pad_data,
                            word2idx=word2idx,
                            idx2word=idx2word)
        return pad_data, word2idx, idx2word


if __name__ == '__main__':
    # 1、读取json文件，提取古诗并简体化，最后存储
    # data = _parseRawData()
    # print(data[:10])

    # 2、获取映射
    config = Config()
    data, word2idx, idx2word = get_data('./data/peot.txt', config)
    print(data[0])
    print(word2idx)
    print(idx2word)
