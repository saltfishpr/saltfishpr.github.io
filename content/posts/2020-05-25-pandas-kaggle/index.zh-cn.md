---
title: Pandas 入门
date: 2020-05-25T00:00:00+08:00
author: Salt Fish
authorLink: https://github.com/saltfishpr
description: Kaggle 上的 Pandas 入门教程
tags: ["python", "data science"]
categories: ["Programming"]
---

<!--more-->

## 读取数据集

```python
import pandas as pd
reviews = pd.read_csv("~/Programming/datasets/winemag-data-130k-v2.csv")
```

## 分组和排序

对分数分组求和

```python
reviews.groupby("points").points.count()
```

Output:

```
points
80       397
81       692
82      1836
...
98        77
99        33
100       19
Name: points, dtype: int64
```

获得相同评分葡萄酒的最低价格

```python
reviews.groupby("points").price.min()
```

Output:

```
points
80      5.0
81      5.0
82      4.0
...
98     50.0
99     44.0
100    80.0
Name: price, dtype: float64
```

从每个酒庄中选择第一瓶葡萄酒的名称

```python
reviews.groupby('winery').apply(lambda df: df.title.iloc[0])
```

Output:

```
winery
1+1=3                                     1+1=3 NV Rosé Sparkling (Cava)
10 Knots                            10 Knots 2010 Viognier (Paso Robles)
100 Percent Wine              100 Percent Wine 2015 Moscato (California)
1000 Stories           1000 Stories 2013 Bourbon Barrel Aged Zinfande...
1070 Green                  1070 Green 2011 Sauvignon Blanc (Rutherford)
...
Órale                       Órale 2011 Cabronita Red (Santa Ynez Valley)
Öko                    Öko 2013 Made With Organically Grown Grapes Ma...
Ökonomierat Rebholz    Ökonomierat Rebholz 2007 Von Rotliegenden Spät...
àMaurice               àMaurice 2013 Fred Estate Syrah (Walla Walla V...
Štoka                                    Štoka 2009 Izbrani Teran (Kras)
Length: 16757, dtype: object
```

根据国家和省份挑选最好的葡萄酒

```python
reviews.groupby(["country", "province"]).apply(lambda df:df.loc[df.points.idxmax()])
```

Output:

```
                            Unnamed: 0  ...                winery
country   province                      ...
Argentina Mendoza Province       82754  ...  Bodega Catena Zapata
          Other                  78303  ...                Colomé
Armenia   Armenia                66146  ...              Van Ardi
Australia Australia Other        37882  ...       Marquis Philips
          New South Wales        85337  ...            De Bortoli
                                ...  ...                   ...
Uruguay   Juanico                 9133  ...        Familia Deicas
          Montevideo             15750  ...                 Bouza
          Progreso               93103  ...                Pisano
          San Jose               39898  ...        Castillo Viejo
          Uruguay                39361  ...               Narbona
[425 rows x 14 columns]
```

`agg()` 方法可以在一个 dataframe 上运行一些一维的函数

```python
reviews.groupby(['country']).price.agg([len, min, max])
```

Output:

```
                            len   min     max
country
Argentina                3800.0   4.0   230.0
Armenia                     2.0  14.0    15.0
Australia                2329.0   5.0   850.0
...
US                      54504.0   4.0  2013.0
Ukraine                    14.0   6.0    13.0
Uruguay                   109.0  10.0   130.0
```

多索引：

使用 `groupby()` 可能会产生多索引

```python
countries_reviewed = reviews.groupby(['country', 'province']).description.agg([len])
countries_reviewed
```

Output:

```
                             len
country   province
Argentina Mendoza Province  3264
          Other              536
Armenia   Armenia              2
Australia Australia Other    245
          New South Wales     85
                          ...
Uruguay   Juanico             12
          Montevideo          11
          Progreso            11
          San Jose             3
          Uruguay             24
[425 rows x 1 columns]
```

多索引转回常规索引

```python
countries_reviewed.reset_index()
```

Output:

```
       country          province   len
0    Argentina  Mendoza Province  3264
1    Argentina             Other   536
2      Armenia           Armenia     2
3    Australia   Australia Other   245
4    Australia   New South Wales    85
..         ...               ...   ...
420    Uruguay           Juanico    12
421    Uruguay        Montevideo    11
422    Uruguay          Progreso    11
423    Uruguay          San Jose     3
424    Uruguay           Uruguay    24
[425 rows x 3 columns]
```

使用 `sort_values()` 对数据排序（默认为升序排序）

```python
countries_reviewed = countries_reviewed.reset_index()
countries_reviewed.sort_values(by='len')
```

Output:

```
          country               province    len
179        Greece  Muscat of Kefallonian      1
192        Greece          Sterea Ellada      1
194        Greece                 Thraki      1
354  South Africa             Paardeberg      1
40         Brazil       Serra do Sudeste      1
..            ...                    ...    ...
409            US                 Oregon   5373
227         Italy                Tuscany   5897
118        France               Bordeaux   5941
415            US             Washington   8639
392            US             California  36247
[425 rows x 3 columns]
```

降序排序

```python
countries_reviewed.sort_values(by='len', ascending=False)
```

Output:

```
          country         province    len
392            US       California  36247
415            US       Washington   8639
118        France         Bordeaux   5941
227         Italy          Tuscany   5897
409            US           Oregon   5373
..            ...              ...    ...
101       Croatia              Krk      1
247   New Zealand        Gladstone      1
357  South Africa  Piekenierskloof      1
63          Chile          Coelemu      1
149        Greece           Beotia      1
[425 rows x 3 columns]
```

重新按照索引排序

```python
countries_reviewed.sort_index()
```

Output:

```
       country          province   len
0    Argentina  Mendoza Province  3264
1    Argentina             Other   536
2      Armenia           Armenia     2
3    Australia   Australia Other   245
4    Australia   New South Wales    85
..         ...               ...   ...
420    Uruguay           Juanico    12
421    Uruguay        Montevideo    11
422    Uruguay          Progreso    11
423    Uruguay          San Jose     3
424    Uruguay           Uruguay    24
[425 rows x 3 columns]
```

同时使用多个列排序

```python
countries_reviewed.sort_values(by=['country', 'len'])
```

Output:

```
       country          province   len
1    Argentina             Other   536
0    Argentina  Mendoza Province  3264
2      Armenia           Armenia     2
6    Australia          Tasmania    42
4    Australia   New South Wales    85
..         ...               ...   ...
421    Uruguay        Montevideo    11
422    Uruguay          Progreso    11
420    Uruguay           Juanico    12
424    Uruguay           Uruguay    24
419    Uruguay         Canelones    43
[425 rows x 3 columns]
```

## 数据类型和缺失值

获取 price 列的数据类型

```python
reviews.price.dtype
```

Output:

```
dtype('float64')
```

查看 dataframe 中每一列的数据类型

```python
reviews.dtypes
```

Output:

```
Unnamed: 0                 int64
country                   object
description               object
designation               object
points                     int64
price                    float64
province                  object
region_1                  object
region_2                  object
taster_name               object
taster_twitter_handle     object
title                     object
variety                   object
winery                    object
dtype: object
```

使用 `astype()` 转换数据类型

```python
reviews.points.astype('float64')
```

Output:

```
0         87.0
1         87.0
2         87.0
3         87.0
4         87.0
          ...
129966    90.0
129967    90.0
129968    90.0
129969    90.0
129970    90.0
Name: points, Length: 129971, dtype: float64
```

缺少值的条目被赋予值 NaN，缩写为 "Not a Number"

查看 country 为 NaN 的行

```python
reviews[pd.isnull(reviews.country)]
```

Output:

```
        Unnamed: 0 country  ...             variety                           winery
913            913     NaN  ...             Chinuri               Gotsa Family Wines
3131          3131     NaN  ...           Red Blend                Barton & Guestier
4243          4243     NaN  ...            Ojaleshi  Kakhetia Traditional Winemaking
9509          9509     NaN  ...         White Blend                         Tsililis
9750          9750     NaN  ...          Chardonnay                         Ross-idi
            ...     ...  ...                 ...                              ...
124176      124176     NaN  ...           Red Blend                Les Frères Dutruy
129407      129407     NaN  ...  Cabernet Sauvignon                      El Capricho
129408      129408     NaN  ...         Tempranillo                      El Capricho
129590      129590     NaN  ...           Red Blend                        Büyülübağ
129900      129900     NaN  ...              Merlot                           Psagot
[63 rows x 14 columns]
```

使用 `fillna()` 填充缺失的行

```python
reviews.region_2.fillna("Unknown)
```

Output:

```
0                   Unknown
1                   Unknown
2         Willamette Valley
3                   Unknown
4         Willamette Valley
                ...
129966              Unknown
129967         Oregon Other
129968              Unknown
129969              Unknown
129970              Unknown
Name: region_2, Length: 129971, dtype: object
```

使用 `replace(old_val, new_val)` 方法替换非空值

```python
reviews.taster_twitter_handle.replace("@kerinokeefe", "@kerino")
```

Output:

```
0             @kerino
1          @vossroger
2          @paulgwine
3                 NaN
4          @paulgwine
             ...
129966            NaN
129967     @paulgwine
129968     @vossroger
129969     @vossroger
129970     @vossroger
Name: taster_twitter_handle, Length: 129971, dtype: object
```

## 重命名和合并

重命名一列

```python
reviews.rename(columns={'points': 'score'})
```

Output:

```
        Unnamed: 0  ...                                    winery
0                0  ...                                   Nicosia
1                1  ...                       Quinta dos Avidagos
2                2  ...                                 Rainstorm
3                3  ...                                St. Julian
4                4  ...                              Sweet Cheeks
            ...  ...                                       ...
129966      129966  ...  Dr. H. Thanisch (Erben Müller-Burggraef)
129967      129967  ...                                  Citation
129968      129968  ...                           Domaine Gresser
129969      129969  ...                      Domaine Marcel Deiss
129970      129970  ...                          Domaine Schoffit
[129971 rows x 14 columns]
```

重命名可以选择 index 或者 colum 参数

```python
reviews.rename(index={0: 'firstEntry', 1: 'secondEntry'})
```

Output:

```
             Unnamed: 0  ...                                    winery
firstEntry            0  ...                                   Nicosia
secondEntry           1  ...                       Quinta dos Avidagos
2                     2  ...                                 Rainstorm
3                     3  ...                                St. Julian
4                     4  ...                              Sweet Cheeks
                 ...  ...                                       ...
129966           129966  ...  Dr. H. Thanisch (Erben Müller-Burggraef)
129967           129967  ...                                  Citation
129968           129968  ...                           Domaine Gresser
129969           129969  ...                      Domaine Marcel Deiss
129970           129970  ...                          Domaine Schoffit
[129971 rows x 14 columns]
```

因为很少重命名索引值，通常情况下 `set_index()` 更好用

给行索引和列索引添加名称属性

```python
reviews.rename_axis("wines", axis='rows').rename_axis("fields", axis='columns')
```

Output:

```
fields  Unnamed: 0  ...                                    winery
wines               ...
0                0  ...                                   Nicosia
1                1  ...                       Quinta dos Avidagos
2                2  ...                                 Rainstorm
3                3  ...                                St. Julian
4                4  ...                              Sweet Cheeks
            ...  ...                                       ...
129966      129966  ...  Dr. H. Thanisch (Erben Müller-Burggraef)
129967      129967  ...                                  Citation
129968      129968  ...                           Domaine Gresser
129969      129969  ...                      Domaine Marcel Deiss
129970      129970  ...                          Domaine Schoffit
[129971 rows x 14 columns]
```

组合有三个核心方法`concat()`, `join()` 和 `merge()`

concat(): 给定一个元素列表，此函数将沿着一个 axis 将这些元素混合在一起

```python
canadian_youtube
british_youtube
pd.concat([canadian_youtube, british_youtube])
```

Output:

```
          video_id  ...                                        description
0      n1WpP7iowLc  ...  Eminem's new track Walk on Water ft. Beyoncé i...
1      0dBIkQ4Mz1M  ...  STill got a lot of packages. Probably will las...
2      5qpjK5DgCt4  ...  WATCH MY PREVIOUS VIDEO ▶ \n\nSUBSCRIBE ► http...
3      d380meD0W0M  ...  I know it's been a while since we did this sho...
4      2Vv-BfVoq4g  ...  🎧: https://ad.gt/yt-perfect\n💰: https://atlant...
            ...  ...                                                ...
40876  sGolxsMSGfQ  ...  🚨 NEW MERCH! http://amzn.to/annoyingorange 🚨➤ ...
40877  8HNuRNi8t70  ...  ► Retrouvez vos programmes préférés : https://...
40878  GWlKEM3m2EE  ...  Find out more about Kingdom Hearts 3: https://...
40879  lbMKLzQ4cNQ  ...  Peter Navarro isn’t talking so tough now. Ana ...
40880  POTgw38-m58  ...  藝人：李妍瑾、玉兔、班傑、LaLa、小優、少少專家：陳筱屏(律師)、Wendy(心理師)、羅...
[40881 rows x 16 columns]

          video_id  ...                                        description
0      Jw1Y-zhQURU  ...  Click here to continue the story and make your...
1      3s1rvMFUweQ  ...  Musical guest Taylor Swift performs …Ready for...
2      n1WpP7iowLc  ...  Eminem's new track Walk on Water ft. Beyoncé i...
3      PUTEiSjKwJU  ...  Salford drew 4-4 against the Class of 92 and F...
4      rHwDegptbI4  ...  Dashcam captures truck's near miss with child ...
            ...  ...                                                ...
38911  l884wKofd54  ...  NEW SONG - MOVE TO MIAMI feat. Pitbull (Click ...
38912  IP8k2xkhOdI  ...  THE OFFICIAL UP WITH IT MUSIC VIDEO!Get my new...
38913  Il-an3K9pjg  ...  Get 2002 by Anne-Marie HERE ▶ http://ad.gt/200...
38914  -DRsfNObKIQ  ...  Eleni Foureira represented Cyprus at the first...
38915  4YFo4bdMO8Q  ...  Debut album 'Light of Mine' out now: http://ky...
[38916 rows x 16 columns]

          video_id  ...                                        description
0      n1WpP7iowLc  ...  Eminem's new track Walk on Water ft. Beyoncé i...
1      0dBIkQ4Mz1M  ...  STill got a lot of packages. Probably will las...
2      5qpjK5DgCt4  ...  WATCH MY PREVIOUS VIDEO ▶ \n\nSUBSCRIBE ► http...
3      d380meD0W0M  ...  I know it's been a while since we did this sho...
4      2Vv-BfVoq4g  ...  🎧: https://ad.gt/yt-perfect\n💰: https://atlant...
            ...  ...                                                ...
38911  l884wKofd54  ...  NEW SONG - MOVE TO MIAMI feat. Pitbull (Click ...
38912  IP8k2xkhOdI  ...  THE OFFICIAL UP WITH IT MUSIC VIDEO!Get my new...
38913  Il-an3K9pjg  ...  Get 2002 by Anne-Marie HERE ▶ http://ad.gt/200...
38914  -DRsfNObKIQ  ...  Eleni Foureira represented Cyprus at the first...
38915  4YFo4bdMO8Q  ...  Debut album 'Light of Mine' out now: http://ky...
[79797 rows x 16 columns]
```

join(): 可以组合具有共同索引的不同 dataframe 对象

```python
left = canadian_youtube.set_index(['title', 'trending_date'])
right = british_youtube.set_index(['title', 'trending_date'])

left.join(right, lsuffix='_CAN', rsuffix='_UK')
```

Output:

```
                                                                 video_id_CAN  ... description_UK
title                                              trending_date               ...
!! THIS VIDEO IS NOTHING BUT PAIN !! | Getting ... 18.04.01       PNn8sECd7io  ...            NaN
#1 Fortnite World Rank - 2,323 Solo Wins!          18.09.03       DvPW66IFhMI  ...            NaN
#1 Fortnite World Rank - 2,330 Solo Wins!          18.10.03       EXEaMjFeiEk  ...            NaN
#1 MOST ANTICIPATED VIDEO (Timber Frame House R... 17.20.12       bYvQmusLaxw  ...            NaN
                                                   17.21.12       bYvQmusLaxw  ...            NaN
                                                                       ...  ...            ...
😲She Is So Nervous But BLOWS The ROOF After Tak... 18.02.05       WttN1Z0XF4k  ...            NaN
                                                   18.29.04       WttN1Z0XF4k  ...            NaN
                                                   18.30.04       WttN1Z0XF4k  ...            NaN
🚨 BREAKING NEWS 🔴 Raja Live all Slot Channels W... 18.07.05       Wt9Gkpmbt44  ...            NaN
🚨Active Shooter at YouTube Headquarters - LIVE ... 18.04.04       Az72jrKbANA  ...            NaN
[40900 rows x 28 columns]
```

## 其他

显示所有列

```python
pd.set_option("display.max_columns", None)
```

<!--
```python

```

Output:

```

```
 -->
