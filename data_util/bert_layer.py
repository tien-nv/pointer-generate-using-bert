import torch
from multiprocess import Pool
from pytorch_pretrained_bert import BertTokenizer
from data_util import config


class BertData():
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        self.sep_vid = self.tokenizer.vocab['[SEP]']
        self.cls_vid = self.tokenizer.vocab['[CLS]']
        self.pad_vid = self.tokenizer.vocab['[PAD]']
        self.min_src_ntokens = 5 #số kí tự tối thiểu của 1 word nếu lớn hơn sẽ bị loại bỏ
        self.max_src_ntokens = 400 #
        self.min_nsents = 3
        self.max_nsents = 100

    def preprocess(self, src):
        """
        This is Tien'document.
        src là một mảng 2 chiều trong đó hàng là một câu, cột là các từ trong 1 câu
        original_src_txt là một mảng 1 chiều trong đó mỗi phần tử là 1 câu
        idxs là một 1 mảng 1 chiều trong đó mỗi phần tử là vị trí các từ trong 1 câu

        trả về:
        src_subtoken_idxs:  là một list các id của các word
        segment_id: là một list các số đánh dấu câu lẻ hay chẵn
        cls_id: là một list các số đánh dấu vị trí các thẻ CLS
        """

        original_src_txt = [' '.join(s) for s in src] #đây là list các câu

        # #chỉ chấp nhận các từ có độ dài kí tự lớn hơn 5
        # idxs = [i for i, s in enumerate(src) if (len(s) > self.min_src_ntokens)] #min_src_ntoken là 5

        idxs = [i for i,s in enumerate(src)]

        #loại bỏ các từ có độ dài kí tự lớn hơn max_src_ntokens
        src = [src[i][:self.max_src_ntokens] for i in idxs] #max_src_ntokens là 400
        
        #một document lấy tối đa là max_nsents câu
        src = src[:self.max_nsents]

        #gộp hàng của ma trận 2 chiều src thành 1 câu
        src_txt = [' '.join(sent) for sent in src]
        # text = [' '.join(ex['src_txt'][i].split()[:self.args.max_src_ntokens]) for i in idxs]
        # text = [_clean(t) for t in text]
        text = ' [SEP] [CLS] '.join(src_txt)
        src_subtokens = self.tokenizer.tokenize(text)
        src_subtokens = src_subtokens[:510]
        src_subtokens = ['[CLS]'] + src_subtokens + ['[SEP]']

        src_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(src_subtokens)
        _segs = [-1] + [i for i, t in enumerate(src_subtoken_idxs) if t == self.sep_vid]
        segs = [_segs[i] - _segs[i - 1] for i in range(1, len(_segs))]
        segments_ids = []
        for i, s in enumerate(segs):
            if (i % 2 == 0):
                segments_ids += s * [0]
            else:
                segments_ids += s * [1]
        cls_ids = [i for i, t in enumerate(src_subtoken_idxs) if t == self.cls_vid]
        sep_ids = [i for i, t in enumerate(src_subtoken_idxs) if t == self.sep_vid]
        return src_subtoken_idxs, segments_ids, cls_ids, sep_ids, len(cls_ids), len(sep_ids)
