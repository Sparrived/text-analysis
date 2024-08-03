import jieba.posseg as pseg
import spacy
import pandas as pd
from pathlib import Path
import pickle
import time

# 路径设置
PATH_MAPPING = {
    "DATASETS": Path("datasets"),
    "OUTPUTS": Path("outputs")
}
# 停用词
STOPWORDS = set(["的", "了", "在", "是", "话题"])
nlp = spacy.load("zh_core_web_sm")

class DataStruct:
    words_with_pos : list[list[tuple[str, str]]] = []
    ents : list[list[tuple[str, str]]] = []
    filtered_words : list[list[tuple[str, str]]] = []
    vectorized_sentence : list[list[list[int]]] = []


    def __init__(self, name, path, textlist) -> None:
        self.name = name
        self.path = path
        self.textlist = textlist
        self.words_with_pos = []
        self.ents = []
        self.filtered_words = []
        self.vectorized_sentence = []
        self.preprocessing()

    def preprocessing(self):
        if self.textlist:
            for sentence in self.textlist:
                # 分词与词性标注
                words_with_pos = pseg.cut(sentence)
                words_with_pos = [(word, flag) for word, flag in words_with_pos if flag != "x"]
                self.words_with_pos.append([(word, flag) for word, flag in words_with_pos if flag != "x"])
                # 命名实体识别
                doc = nlp(sentence)
                self.ents.append([(ent.text, ent.label_) for ent in doc.ents])
                # 去除停用词
                filtered_words = [(word, flag) for word, flag in words_with_pos if word not in STOPWORDS]
                self.filtered_words.append(filtered_words)
                # 向量化（使用spacy）
                vectorized_sentence = doc.vector
                self.vectorized_sentence.append(vectorized_sentence)
        else:
            print("数据不存在！")

# 设置excel表格中的列名
COLUMNNAMES = ["评论内容", "博文内容"]
data_structs : list[DataStruct] = []
from concurrent.futures import ProcessPoolExecutor, as_completed

def process_excel_file(file_path: Path):
    result = []
    try:
        df = pd.read_excel(file_path, engine='openpyxl')
        for column in COLUMNNAMES:
            if column in df.columns:
                data_struct = DataStruct(f"[{column}]{file_path.name}", file_path, list(df[column]))
                result.append(data_struct)
                print(data_struct.name, "done!")
    except Exception as e:
        print(e)
    return result

def process_txt_file(file_path: Path):
    result = []
    try:
        with open(file_path, 'r', encoding='utf8') as f:
            lines = f.readlines()
            data_struct = DataStruct(f"[评论内容]{file_path.name}", file_path, lines)
            result.append(data_struct)
            print(data_struct.name, "done!")
    except Exception as e:
        print(e)
    return result

if __name__ == "__main__":
    print("开始处理文件")
    excel_files = list(PATH_MAPPING['DATASETS'].rglob('*.xlsx'))
    txt_files = list(PATH_MAPPING['DATASETS'].rglob('*.txt'))
    with ProcessPoolExecutor() as executor:
        futures = []
        for file in excel_files:
            futures.append(executor.submit(process_excel_file, file))
        for file in txt_files:
            futures.append(executor.submit(process_txt_file, file))
        for future in as_completed(futures):
            result = future.result()
            data_structs.extend(result)
    pkl_path = PATH_MAPPING['DATASETS'] / "pickle"
    pkl_name =  pkl_path / (str(time.time()) + ".pkl")
    pkl_path.mkdir(exist_ok=True)
    with open(pkl_name, 'wb') as f:
        pickle.dump(data_structs, f)
    with open(pkl_path / "tmp", 'w') as f:
        f.write(str(pkl_name))