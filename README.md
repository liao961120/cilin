Cilin
=================


## Install

```bash
pip install cilin
```

## Usage

```python
from cilin import Cilin

C = Cilin(trad=False)  # trad=True to convert results to traditional Chinese

# Category label
>>> C.get_tag('A')
'人'
>>> C.get_tag('Aa')
'人-泛称'
>>> C.get_tag('Aa01')
'人 人民 众人 角色'

# Terms under a certain category
>>> C.get_members('Ca01B')
{'己巳', '回历', '戊辰', '丙寅', '太阴历', '子丑寅卯', '庚午', '地支', '乙丑', '夏历', '甲午', '旧历', '天干', '巳', '子午卯酉', '阴历', '戊寅', '戊戌', '伊斯兰教历', '丁卯', '农历', '庚子', '庚申', '癸', '辛未', '辛亥'}
>>> C.get_members('Ca01B01=')
{'旧历', '农历', '太阴历', '阴历', '夏历'}

# Divide terms into categories at different levels (1-5)
>>> C.category_split(level=1)
{
    'A': {'荷兰人', '巡视员', '人夫', '农妇', '王子', ...},
    'B': {'赌具', '湖', '敞篷车', '灰黄色', '圆桌', ...},
    ...
    'L': {'费心', '失敬', '劳神', '恭贺', '借问', ...}
}
```


## Acknowledgent

同義詞詞林資料：<https://github.com/yaleimeng/Final_word_Similarity/tree/master/cilin/V3>