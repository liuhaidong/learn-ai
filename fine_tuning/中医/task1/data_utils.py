from tqdm import tqdm
import json
import torch
import os
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split

def TCM_SD_Data_Loader(tokenizer,type):
    herbs = ['冬瓜皮', '沉香', '茜草炭', '浮小麦', '炙甘草', '炒白扁豆', '砂仁', '合欢花', '北刘寄奴', '炒六神曲', '炒决明子', '益母草', '酒苁蓉', '炒僵蚕', '稀莶草', '秦艽', '黄酒', '瞿麦', '白鲜皮', '熟地黄', '扁蓄', '诃子肉', '煅牡蛎', '鸡血藤', '党参', '瓜蒌', '莲子', '酒五味子', '金钱草', '法半夏', '北败酱草', '花椒', '吴茱萸(粉)', '桑白皮', '茯神', '桂枝', '降香', '制远志', '琥珀', '佛手', '麦芽', '水红花子', '金银花', '马鞭草', '半枝莲', '炮姜', '生酸枣仁', '盐补骨脂', '炒瓜蒌子', '珍珠母', '乌药', '茵陈', '地肤子', '酸枣仁', '槟榔', '大青叶', '人参片', '麸煨肉豆蔻', '蛤蚧', '路路通', '蝉蜕', '马勃', '香橼', '络石藤', '狗脊', '蜈蚣', '制川乌', '白扁豆花', '麻黄', '射干', '厚朴', '蜂蜜', '柏子仁', '炒谷芽', '蜜百合', '石菖蒲', '白薇', '续断', '炒川楝子', '黄连片', '绵萆薢', '鹿角胶', '翻白草', '羚羊角粉', '天麻', '山慈菇', '菊花', '炒芥子', '墨旱莲', '蜜枇杷叶', '川芎', '酒大黄', '焦山楂', '红曲', '山药', '牡蛎', '海藻', '夏枯草', '白前', '白芍', '茯苓皮', '煅自然铜', '附片', '土茯苓', '制何首乌', '炒莱菔子', '黄芩', '蒲黄', '紫石英', '透骨草', '绞股蓝', '泽泻', '甘松', '炒酸枣仁', '儿茶', '马齿苋', '太子参', '薏苡仁', '萹蓄', '青蒿', '苏木', '桑叶', '连翘', '穿山龙', '忍冬藤', '苦参', '炒茺蔚子', '防己', '益母草炭', '莲须', '猫眼草', '麸炒芡实', '炒牛蒡子', '龟甲胶', '蜜槐角', '柿蒂', '龙骨', '泽兰', '桔梗', '青葙子', '冰片', '大枣', '侧柏叶', '三七粉', '醋乳香', '川牛膝', '全蝎', '合欢皮', '首乌藤', '醋鳖甲', '炒蔓荆子', '烫骨碎补', '紫苏叶', '盐沙苑子', '南沙参', '石见穿', '胆南星', '焦白术', '酒黄芩', '白术', '鬼箭羽', '玫瑰花', '干姜', '牡丹皮', '白花蛇舌草', '酒当归', '火麻仁', '炒桃仁', '醋鸡内金', '磁石', '醋龟甲', '白茅根', '肉桂', '白及', '油松节', '炒苍耳子', '化橘红', '佩兰', '芦根', '紫草', '酒萸肉', '丹参', '柴胡', '制巴戟天', '木蝴蝶', '炒紫苏子', '浮萍', '栀子', '甘草片', '木香', '丝瓜络', '炒麦芽', '板蓝根', '车前草', '炒王不留行', '朱砂', '醋三棱', '辛夷', '土鳖虫', '煅龙骨', '炒白芍', '炒白果仁', '芒硝', '赭石', '西洋参', '桑枝', '红景天', '锁阳', '淫羊藿', '酒乌梢蛇', '制草乌', '肉苁蓉片', '麸炒枳壳', '炒苦杏仁', '炙黄芪', '黄连', '重楼', '细辛', '蜜旋覆花', '醋没药', '玉竹', '蛤壳', '草豆蔻', '炙淫羊藿', '广藿香', '麸炒枳实', '鱼腥草', '鹿角霜', '通草', '烫水蛭', '水牛角', '烫狗脊', '盐续断', '盐益智仁', '常山', '百部', '阿胶', '藁本片', '制吴茱萸', '豆蔻', '酒女贞子', '片姜黄', '蜜款冬花', '龙胆', '寒水石', '莲子心', '荷叶', '防风', '炒蒺藜', '川贝母', '虎杖', '海桐皮', '甘草', '赤石脂', '麻黄根', '郁金', '海风藤', '青皮', '地龙', '地榆', '石韦', '焦栀子', '盐杜仲', '清半夏', '盐知母', '薤白', '茜草', '荆芥炭', '百合', '龙齿', '石决明', '炒葶苈子', '知母', '赤小豆', '麸炒白术', '酒仙茅', '淡竹叶', '大黄', '海螵蛸', '仙鹤草', '白芷', '麸炒薏苡仁', '青风藤', '前胡', '升麻', '海浮石', '制天南星', '麸炒山药', '蒲公英', '豨莶草', '当归', '醋莪术', '薄荷', '红参片', '生地黄', '苦地丁', '炒槐米', '蜜桑白皮', '盐小茴香', '麸炒苍术', '姜半夏', '钟乳石', '桑椹', '瓜蒌皮', '葛根', '桑螵蛸', '浙贝片', '菟丝子', '醋延胡索', '艾叶', '五加皮', '炒冬瓜子', '瓦楞子', '盐黄柏', '醋五灵脂', '石膏', '醋山甲', '檀香', '皂角刺', '红花', '野菊花', '木瓜', '蜜麻黄', '槲寄生', '密蒙花', '蜜百部', '蜜紫菀', '茯苓', '海金沙', '麦冬', '猪苓', '天竺黄', '石斛', '枸杞子', '徐长卿', '醋香附', '麸神曲', '黄芪', '郁李仁', '枯矾', '盐车前子', '伸筋草', '草果仁', '山楂', '炒稻芽', '威灵仙', '淡豆豉', '蛇莓', '丁香', '盐荔枝核', '绵马贯众', '黄柏', '独活', '覆盆子', '龙眼肉', '老鹳草', '乌梅', '紫苏梗', '制白附子', '大腹皮', '竹茹', '天花粉', '乌梅炭', '滑石粉', '冬葵子', '灯心草', '六月雪', '牛膝', '陈皮', '荆芥', '炒甘草', '北沙参', '地骷髅', '地骨皮', '赤芍', '玄参', '桑葚', '酒黄精', '羌活', '钩藤', '天冬']

    #找到所有的标签
    if type ==1:
        syndromes = ['气虚血瘀证', '痰瘀互结证', '气阴两虚证', '气滞血瘀证', '肝阳上亢证', '阴虚阳亢证', '痰热蕴结证', '痰湿痹阻证', '阳虚水停证', '肝肾阴虚证','胸痹心痛病', '心衰病', '眩晕病', '心悸病']   #使用bert进行预测疾病和证型联合时标签
        syndromes = syndromes + herbs
    elif type == 2:
        syndromes = ['胸痹心痛病', '心衰病', '眩晕病', '心悸病']        #单使用bert进行预测疾病时标签
    else:
        syndromes = ['气虚血瘀证', '痰瘀互结证', '气阴两虚证', '气滞血瘀证', '肝阳上亢证', '阴虚阳亢证', '痰热蕴结证', '痰湿痹阻证', '阳虚水停证', '肝肾阴虚证']  #单使用bert进行预测证型时标签

    
    id2syndrome_dict = {}
    syndrome2id_dict = {}

    y = 0
    for i in range(len(syndromes)):
        id2syndrome_dict[i] = syndromes[i]
        syndrome2id_dict[syndromes[i]] = i

    def get_InputTensor(path):

        contents = []
        with open(path, 'r', encoding='utf-8') as file:
            file = json.load(file)
            for line in file:
                contents.append(line)

        labels = []
        input_ids = []
        attention_masks = []
        true_splitNumbers = []
        sentences = []

        for content in tqdm(contents, desc='Loading data',total=len(contents)):
            if not content.get('证型'):
                continue
            sentence = content['症状'] + content['中医望闻切诊']
            sentences.append(sentence)
            labele_sentence = [0]*(len(syndromes))
            # 证型标签
            if type == 1 or type == 3:
                for label in content['证型'].split('|'):
                    id = syndrome2id_dict[label]
                    labele_sentence[id] = 1
            if type == 1:
                prescription = [item.strip(" '") for item in content['处方'].strip(" []").split(",")]
                for herb in prescription:    
                    id = syndrome2id_dict[herb]  
                    labele_sentence[id] = 1  
            #疾病标签
            if type == 1 or type == 2:
                labele_sentence[syndrome2id_dict[content['疾病']]] = 1
            
            labels.append(torch.tensor(labele_sentence))

        # 输入数据集种的文本
        input_ids_sen = []
        attention_masks_sen = []
        for sentence in tqdm(sentences,desc='Loading clinic text {} sentence'.format(path)):
            sentencei = sentence[:512]
            encoded_dicti = tokenizer(
                sentencei,  # 输入文本
                add_special_tokens=True,  # 添加 '[CLS]' 和 '[SEP]'
                max_length=100,  # 填充 & 截断长度
                padding='max_length',
                return_attention_mask=True,  # 返回 attn. masks.
                return_tensors='pt',  # 返回 pytorch tensors 格式的数据
                truncation=True
            )
            input_idsi = encoded_dicti['input_ids'][0].reshape(1,-1)
            attention_maski = encoded_dicti['attention_mask'][0].reshape(1,-1)
            input_ids_sen.append(input_idsi)
            attention_masks_sen.append(attention_maski)
        input_ids = torch.cat(input_ids_sen, dim=0)
        attention_masks = torch.cat(attention_masks_sen, dim=0)
        labels = torch.stack(labels, dim=0)
        return input_ids, attention_masks, labels

    input_ids_train, attention_masks_train, labels_train = get_InputTensor(
        '/home/liuhaidong/workspace/tianchi/TCM-Syndrome-and-Disease-Differentiation-and-Prescription-Recommendation/baseline/task1/dataset/TCM-TBOSD-train.json')
    # input_ids_test, attention_masks_test, labels_test = get_InputTensor(
    #     '/home/liuhaidong/workspace/tianchi/TCM-Syndrome-and-Disease-Differentiation-and-Prescription-Recommendation/baseline/task1/dataset/TCM-TBOSD-test-A.json')
    # input_ids_val, attention_masks_val, labels_val = get_InputTensor(
    #     '/home/liuhaidong/workspace/tianchi/TCM-Syndrome-and-Disease-Differentiation-and-Prescription-Recommendation/baseline/task1/dataset/TCM-TBOSD-test-A.json')

        # 将输入数据合并为 TensorDataset 对象
    
    # Split into train (80%), validation (10%), test (10%)
    train_size = 0.8
    val_size = 0.1
    test_size = 0.1

    # First split
    input_ids_train, input_ids_temp, attention_masks_train, attention_masks_temp, labels_train, labels_temp = train_test_split(
        input_ids_train, 
        attention_masks_train, 
        labels_train, 
        test_size=(val_size + test_size), 
        random_state=42
    )

    # Adjust test size for second split
    test_size_adjusted = test_size / (val_size + test_size)

    # Second split
    input_ids_val, input_ids_test, attention_masks_val, attention_masks_test, labels_val, labels_test = train_test_split(
        input_ids_temp,
        attention_masks_temp,
        labels_temp,
        test_size=test_size_adjusted,
        random_state=42
    )

    train_dataset = TensorDataset(input_ids_train, attention_masks_train, labels_train)
    test_dataset = TensorDataset(input_ids_test, attention_masks_test, labels_test)
    val_dataset = TensorDataset(input_ids_val, attention_masks_val, labels_val)
    
    # 为训练和验证集创建 Dataloader，对训练样本随机洗牌
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,  # 训练样本
        sampler=RandomSampler(train_dataset),  # 随机小批量
        batch_size=2,  # 以小批量进行训练
        drop_last=True,
    )

    # 测试集不需要随机化，这里顺序读取就好
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,  # 验证样本
        sampler=SequentialSampler(test_dataset),  # 顺序选取小批量
        batch_size=2,
        drop_last=True,
    )

    # 验证集不需要随机化，这里顺序读取就好
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,  # 验证样本
        sampler=SequentialSampler(val_dataset),  # 顺序选取小批量
        batch_size=2,
        drop_last=True
    )


    return train_dataloader, test_dataloader, val_dataloader, id2syndrome_dict


# from transformers import BertTokenizer
# from transformers import BertConfig, BertModel, AdamW
# tokenizer = BertTokenizer.from_pretrained('/home/liuhaidong/workspace/tianchi/TCM-Syndrome-and-Disease-Differentiation-and-Prescription-Recommendation/baseline/task1/ZY_BERT/chinese-bert-wwm-ext')
# TCM_SD_Data_Loader(tokenizer,1)