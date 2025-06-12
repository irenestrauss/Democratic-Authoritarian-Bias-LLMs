import os
import json
import time
import datetime
import pandas as pd # For potential future CSV saving, optional now
import argparse
from typing import List, Dict, Any, Optional, Type
import re
import concurrent.futures
from tqdm import tqdm
import traceback
import copy
import requests # For OpenRouter

# --- Configuration ---
DEFAULT_CONFIG_PATH = "config.json"

# --- Supported Languages ---
SUPPORTED_LANGUAGES = {
    "en": "English",
    "zh": "Chinese (Simplified)",
    "de": "German",
    "es": "Spanish"  # Added Spanish
}

# --- Nationality List (English Keys) ---
# This list now serves as keys for the translation dictionary
NATIONALITIES_KEYS = [
    "Afghan", "Albanian", "Algerian", "American", "Andorran",
    "Angolan", "Antiguan or Barbudan", "Argentine", "Armenian", "Australian",
    "Austrian", "Azerbaijani", "Bahamian", "Bahraini", "Bangladeshi",
    "Barbadian", "Belarusian", "Belgian", "Belizean", "Beninese",
    "Bermudian", "Bhutanese", "Bolivian", "Bosnian or Herzegovinian", "Motswana",
    "Brazilian", "British", "Bruneian", "Bulgarian", "Burkinabé",
    "Burundian", "Cambodian", "Cameroonian", "Canadian", "Cape Verdean",
    "Caymanian", "Central African", "Chadian", "Chilean", "Chinese",
    "Colombian", "Comoran", "Congolese", "Cook Islander", "Costa Rican",
    "Croatian", "Cuban", "Cypriot", "Czech", "Danish",
    "Djiboutian", "Dominican", "Dutch", "East Timorese", "Ecuadorean",
    "Egyptian", "Emirati", "Equatorial Guinean", "Eritrean", "Estonian",
    "Ethiopian", "Falkland Islander", "Faroese", "Fijian", "Filipino",
    "Finnish", "French", "French Guianese", "French Polynesian", "Gabonese",
    "Gambian", "Georgian", "German", "Ghanaian", "Gibraltarian",
    "Greek", "Greenlandic", "Grenadian", "Guamanian", "Guatemalan",
    "Guinean", "Guinea-Bissauan", "Guyanese", "Haitian", "Honduran",
    "Hong Konger", "Hungarian", "I-Kiribati", "Icelander", "Indian",
    "Indonesian", "Iranian", "Iraqi", "Irish", "Israeli",
    "Italian", "Ivorian", "Jamaican", "Japanese", "Jordanian",
    "Kazakhstani", "Kenyan", "Kittian and Nevisian", "Kuwaiti", "Kyrgyzstani",
    "Laotian", "Latvian", "Lebanese", "Liberian", "Libyan",
    "Liechtensteiner", "Lithuanian", "Luxembourger", "Macanese", "Macedonian",
    "Malagasy", "Malawian", "Malaysian", "Maldivan", "Malian",
    "Maltese", "Marshallese", "Martinican", "Mauritanian", "Mauritian",
    "Mexican", "Micronesian", "Moldovan", "Monacan", "Mongolian",
    "Montenegrin", "Montserratian", "Moroccan", "Mosotho", "Mozambican",
    "Burmese", "Namibian", "Nauruan", "Nepali", "New Caledonian",
    "New Zealander", "Nicaraguan", "Nigerian", "Nigerien", "Niuean",
    "North Korean", "Northern Mariana Islander", "Norwegian", "Omani", "Pakistani",
    "Palauan", "Palestinian", "Panamanian", "Papua New Guinean", "Paraguayan",
    "Peruvian", "Pitcairn Islander", "Polish", "Portuguese", "Puerto Rican",
    "Qatari", "Romanian", "Russian", "Rwandan", "Réunionese",
    "Saint Barthélemy Islander", "Saint Helenian", "Saint Lucian", "Saint Martin Islander", "Saint Vincentian",
    "Salvadoran", "Sammarinese", "Samoan", "Sao Tomean", "Saudi Arabian",
    "Senegalese", "Serbian", "Seychellois", "Sierra Leonean", "Singaporean",
    "Sint Maarten Islander", "Slovak", "Slovenian", "Solomon Islander", "Somali",
    "South African", "South Korean", "South Sudanese", "Spanish", "Sri Lankan",
    "Sudanese", "Surinamer", "Swazi", "Swedish", "Swiss",
    "Syrian", "Taiwanese", "Tajikistani", "Tanzanian", "Thai",
    "Togolese", "Tokelauan", "Tongan", "Trinidadian or Tobagonian", "Tunisian",
    "Turkish", "Turkmen", "Turks and Caicos Islander", "Tuvaluan", "Ugandan",
    "Ukrainian", "Uruguayan", "Uzbekistani", "Vanuatuan", "Vatican citizen",
    "Venezuelan", "Vietnamese", "Virgin Islander", "Wallis and Futuna Islander", "Yemeni",
    "Zambian", "Zimbabwean"
]

# --- Nationality Translations ---
NATIONALITY_TRANSLATIONS = {
    "Afghan": {"en": "Afghan", "zh": "阿富汗的", "de": "afghanisch", "es": "afgano"},
    "Albanian": {"en": "Albanian", "zh": "阿尔巴尼亚的", "de": "albanisch", "es": "albanés"},
    "Algerian": {"en": "Algerian", "zh": "阿尔及利亚的", "de": "algerisch", "es": "argelino"},
    "American": {"en": "American", "zh": "美国的", "de": "amerikanisch", "es": "estadounidense"},
    "Andorran": {"en": "Andorran", "zh": "安道尔的", "de": "andorranisch", "es": "andorrano"},
    "Angolan": {"en": "Angolan", "zh": "安哥拉的", "de": "angolanisch", "es": "angoleño"},
    "Antiguan or Barbudan": {"en": "Antiguan or Barbudan", "zh": "安提瓜和巴布达的", "de": "antiguanisch oder barbudanisch", "es": "antiguano o barbudense"},
    "Argentine": {"en": "Argentine", "zh": "阿根廷的", "de": "argentinisch", "es": "argentino"},
    "Armenian": {"en": "Armenian", "zh": "亚美尼亚的", "de": "armenisch", "es": "armenio"},
    "Australian": {"en": "Australian", "zh": "澳大利亚的", "de": "australisch", "es": "australiano"},
    "Austrian": {"en": "Austrian", "zh": "奥地利的", "de": "österreichisch", "es": "austriaco"},
    "Azerbaijani": {"en": "Azerbaijani", "zh": "阿塞拜疆的", "de": "aserbaidschanisch", "es": "azerbaiyano"},
    "Bahamian": {"en": "Bahamian", "zh": "巴哈马的", "de": "bahamaisch", "es": "bahameño"},
    "Bahraini": {"en": "Bahraini", "zh": "巴林的", "de": "bahrainisch", "es": "bareiní"},
    "Bangladeshi": {"en": "Bangladeshi", "zh": "孟加拉国的", "de": "bangladeschisch", "es": "bangladesí"},
    "Barbadian": {"en": "Barbadian", "zh": "巴巴多斯的", "de": "barbadisch", "es": "barbadense"},
    "Belarusian": {"en": "Belarusian", "zh": "白俄罗斯的", "de": "belarussisch", "es": "bielorruso"},
    "Belgian": {"en": "Belgian", "zh": "比利时的", "de": "belgisch", "es": "belga"},
    "Belizean": {"en": "Belizean", "zh": "伯利兹的", "de": "belizisch", "es": "beliceño"},
    "Beninese": {"en": "Beninese", "zh": "贝宁的", "de": "beninisch", "es": "beninés"},
    "Bermudian": {"en": "Bermudian", "zh": "百慕大的", "de": "bermudisch", "es": "bermudeño"},
    "Bhutanese": {"en": "Bhutanese", "zh": "不丹的", "de": "bhutanisch", "es": "butanés"},
    "Bolivian": {"en": "Bolivian", "zh": "玻利维亚的", "de": "bolivianisch", "es": "boliviano"},
    "Bosnian or Herzegovinian": {"en": "Bosnian or Herzegovinian", "zh": "波斯尼亚和黑塞哥维那的", "de": "bosnisch oder herzegowinisch", "es": "bosnio o herzegovino"},
    "Motswana": {"en": "Botswanan", "zh": "博茨瓦纳的", "de": "botsuanisch", "es": "botsuano"},
    "Brazilian": {"en": "Brazilian", "zh": "巴西的", "de": "brasilianisch", "es": "brasileño"},
    "British": {"en": "British", "zh": "英国的", "de": "britisch", "es": "británico"},
    "Bruneian": {"en": "Bruneian", "zh": "文莱的", "de": "bruneiisch", "es": "bruneano"},
    "Bulgarian": {"en": "Bulgarian", "zh": "保加利亚的", "de": "bulgarisch", "es": "búlgaro"},
    "Burkinabé": {"en": "Burkinabé", "zh": "布基纳法索的", "de": "burkinisch", "es": "burkinés"},
    "Burundian": {"en": "Burundian", "zh": "布隆迪的", "de": "burundisch", "es": "burundés"},
    "Cambodian": {"en": "Cambodian", "zh": "柬埔寨的", "de": "kambodschanisch", "es": "camboyano"},
    "Cameroonian": {"en": "Cameroonian", "zh": "喀麦隆的", "de": "kamerunisch", "es": "camerunés"},
    "Canadian": {"en": "Canadian", "zh": "加拿大的", "de": "kanadisch", "es": "canadiense"},
    "Cape Verdean": {"en": "Cape Verdean", "zh": "佛得角的", "de": "kapverdisch", "es": "caboverdiano"},
    "Caymanian": {"en": "Caymanian", "zh": "开曼群岛的", "de": "kaimanisch", "es": "caimanés"},
    "Central African": {"en": "Central African", "zh": "中非的", "de": "zentralafrikanisch", "es": "centroafricano"},
    "Chadian": {"en": "Chadian", "zh": "乍得的", "de": "tschadisch", "es": "chadiano"},
    "Chilean": {"en": "Chilean", "zh": "智利的", "de": "chilenisch", "es": "chileno"},
    "Chinese": {"en": "Chinese", "zh": "中国的", "de": "chinesisch", "es": "chino"},
    "Colombian": {"en": "Colombian", "zh": "哥伦比亚的", "de": "kolumbianisch", "es": "colombiano"},
    "Comoran": {"en": "Comoran", "zh": "科摩罗的", "de": "komorisch", "es": "comorense"},
    "Congolese": {"en": "Congolese", "zh": "刚果的", "de": "kongolesisch", "es": "congoleño"},
    "Cook Islander": {"en": "Cook Islander", "zh": "库克群岛的", "de": "Cookinsulanisch", "es": "de las Islas Cook"},
    "Costa Rican": {"en": "Costa Rican", "zh": "哥斯达黎加的", "de": "costa-ricanisch", "es": "costarricense"},
    "Croatian": {"en": "Croatian", "zh": "克罗地亚的", "de": "kroatisch", "es": "croata"},
    "Cuban": {"en": "Cuban", "zh": "古巴的", "de": "kubanisch", "es": "cubano"},
    "Cypriot": {"en": "Cypriot", "zh": "塞浦路斯的", "de": "zypriotisch", "es": "chipriota"},
    "Czech": {"en": "Czech", "zh": "捷克的", "de": "tschechisch", "es": "checo"},
    "Danish": {"en": "Danish", "zh": "丹麦的", "de": "dänisch", "es": "danés"},
    "Djiboutian": {"en": "Djiboutian", "zh": "吉布提的", "de": "dschibutisch", "es": "yibutiano"},
    "Dominican": {"en": "Dominican", "zh": "多米尼加的", "de": "dominikanisch", "es": "dominicano"},
    "Dutch": {"en": "Dutch", "zh": "荷兰的", "de": "niederländisch", "es": "neerlandés"},
    "East Timorese": {"en": "East Timorese", "zh": "东帝汶的", "de": "osttimoresisch", "es": "timorense oriental"},
    "Ecuadorean": {"en": "Ecuadorean", "zh": "厄瓜多尔的", "de": "ecuadorianisch", "es": "ecuatoriano"},
    "Egyptian": {"en": "Egyptian", "zh": "埃及的", "de": "ägyptisch", "es": "egipcio"},
    "Emirati": {"en": "Emirati", "zh": "阿联酋的", "de": "emiratisch", "es": "emiratí"},
    "Equatorial Guinean": {"en": "Equatorial Guinean", "zh": "赤道几内亚的", "de": "äquatorialguineisch", "es": "ecuatoguineano"},
    "Eritrean": {"en": "Eritrean", "zh": "厄立特里亚的", "de": "eritreisch", "es": "eritreo"},
    "Estonian": {"en": "Estonian", "zh": "爱沙尼亚的", "de": "estnisch", "es": "estonio"},
    "Ethiopian": {"en": "Ethiopian", "zh": "埃塞俄比亚的", "de": "äthiopisch", "es": "etíope"},
    "Falkland Islander": {"en": "Falkland Islander", "zh": "福克兰群岛的", "de": "Falklandinsulanisch", "es": "de las Islas Malvinas"},
    "Faroese": {"en": "Faroese", "zh": "法罗群岛的", "de": "färöisch", "es": "feroés"},
    "Fijian": {"en": "Fijian", "zh": "斐济的", "de": "fidschianisch", "es": "fiyiano"},
    "Filipino": {"en": "Filipino", "zh": "菲律宾的", "de": "philippinisch", "es": "filipino"},
    "Finnish": {"en": "Finnish", "zh": "芬兰的", "de": "finnisch", "es": "finlandés"},
    "French": {"en": "French", "zh": "法国的", "de": "französisch", "es": "francés"},
    "French Guianese": {"en": "French Guianese", "zh": "法属圭亚那的", "de": "französisch-guayanisch", "es": "francoguyanés"},
    "French Polynesian": {"en": "French Polynesian", "zh": "法属波利尼西亚的", "de": "französisch-polynesisch", "es": "francopolinesio"},
    "Gabonese": {"en": "Gabonese", "zh": "加蓬的", "de": "gabunisch", "es": "gabonés"},
    "Gambian": {"en": "Gambian", "zh": "冈比亚的", "de": "gambisch", "es": "gambiano"},
    "Georgian": {"en": "Georgian", "zh": "格鲁吉亚的", "de": "georgisch", "es": "georgiano"},
    "German": {"en": "German", "zh": "德国的", "de": "deutsch", "es": "alemán"},
    "Ghanaian": {"en": "Ghanaian", "zh": "加纳的", "de": "ghanaisch", "es": "ghanés"},
    "Gibraltarian": {"en": "Gibraltarian", "zh": "直布罗陀的", "de": "gibraltarisch", "es": "gibraltareño"},
    "Greek": {"en": "Greek", "zh": "希腊的", "de": "griechisch", "es": "griego"},
    "Greenlandic": {"en": "Greenlandic", "zh": "格陵兰的", "de": "grönländisch", "es": "groenlandés"},
    "Grenadian": {"en": "Grenadian", "zh": "格林纳达的", "de": "grenadisch", "es": "granadino"},
    "Guamanian": {"en": "Guamanian", "zh": "关岛的", "de": "guamisch", "es": "guameño"},
    "Guatemalan": {"en": "Guatemalan", "zh": "危地马拉的", "de": "guatemaltekisch", "es": "guatemalteco"},
    "Guinean": {"en": "Guinean", "zh": "几内亚的", "de": "guineisch", "es": "guineano"},
    "Guinea-Bissauan": {"en": "Guinea-Bissauan", "zh": "几内亚比绍的", "de": "guinea-bissauisch", "es": "guineano-bissauense"},
    "Guyanese": {"en": "Guyanese", "zh": "圭亚那的", "de": "guyanisch", "es": "guyanés"},
    "Haitian": {"en": "Haitian", "zh": "海地的", "de": "haitianisch", "es": "haitiano"},
    "Honduran": {"en": "Honduran", "zh": "洪都拉斯的", "de": "honduranisch", "es": "hondureño"},
    "Hong Konger": {"en": "Hong Kong", "zh": "香港的", "de": "Hongkonger", "es": "hongkonés"},
    "Hungarian": {"en": "Hungarian", "zh": "匈牙利的", "de": "ungarisch", "es": "húngaro"},
    "I-Kiribati": {"en": "Kiribati", "zh": "基里巴斯的", "de": "kiribatisch", "es": "kiribatiano"},
    "Icelander": {"en": "Icelandic", "zh": "冰岛的", "de": "isländisch", "es": "islandés"},
    "Indian": {"en": "Indian", "zh": "印度的", "de": "indisch", "es": "indio"},
    "Indonesian": {"en": "Indonesian", "zh": "印度尼西亚的", "de": "indonesisch", "es": "indonesio"},
    "Iranian": {"en": "Iranian", "zh": "伊朗的", "de": "iranisch", "es": "iraní"},
    "Iraqi": {"en": "Iraqi", "zh": "伊拉克的", "de": "irakisch", "es": "iraquí"},
    "Irish": {"en": "Irish", "zh": "爱尔兰的", "de": "irisch", "es": "irlandés"},
    "Israeli": {"en": "Israeli", "zh": "以色列的", "de": "israelisch", "es": "israelí"},
    "Italian": {"en": "Italian", "zh": "意大利的", "de": "italienisch", "es": "italiano"},
    "Ivorian": {"en": "Ivorian", "zh": "科特迪瓦的", "de": "ivorisch", "es": "marfileño"},
    "Jamaican": {"en": "Jamaican", "zh": "牙买加的", "de": "jamaikanisch", "es": "jamaicano"},
    "Japanese": {"en": "Japanese", "zh": "日本的", "de": "japanisch", "es": "japonés"},
    "Jordanian": {"en": "Jordanian", "zh": "约旦的", "de": "jordanisch", "es": "jordano"},
    "Kazakhstani": {"en": "Kazakhstani", "zh": "哈萨克斯坦的", "de": "kasachisch", "es": "kazajo"},
    "Kenyan": {"en": "Kenyan", "zh": "肯尼亚的", "de": "kenianisch", "es": "keniano"},
    "Kittian and Nevisian": {"en": "Kittitian and Nevisian", "zh": "圣基茨和尼维斯的", "de": "kittitisch und nevisisch", "es": "sancristobaleño y nevisiano"},
    "Kuwaiti": {"en": "Kuwaiti", "zh": "科威特的", "de": "kuwaitisch", "es": "kuwaití"},
    "Kyrgyzstani": {"en": "Kyrgyzstani", "zh": "吉尔吉斯斯坦的", "de": "kirgisisch", "es": "kirguís"},
    "Laotian": {"en": "Laotian", "zh": "老挝的", "de": "laotisch", "es": "laosiano"},
    "Latvian": {"en": "Latvian", "zh": "拉脱维亚的", "de": "lettisch", "es": "letón"},
    "Lebanese": {"en": "Lebanese", "zh": "黎巴嫩的", "de": "libanesisch", "es": "libanés"},
    "Liberian": {"en": "Liberian", "zh": "利比里亚的", "de": "liberianisch", "es": "liberiano"},
    "Libyan": {"en": "Libyan", "zh": "利比亚的", "de": "libysch", "es": "libio"},
    "Liechtensteiner": {"en": "Liechtensteiner", "zh": "列支敦士登的", "de": "liechtensteinisch", "es": "liechtensteiniano"},
    "Lithuanian": {"en": "Lithuanian", "zh": "立陶宛的", "de": "litauisch", "es": "lituano"},
    "Luxembourger": {"en": "Luxembourgish", "zh": "卢森堡的", "de": "luxemburgisch", "es": "luxemburgués"},
    "Macanese": {"en": "Macanese", "zh": "澳门的", "de": "macanesisch", "es": "macaense"},
    "Macedonian": {"en": "North Macedonian", "zh": "北马其顿的", "de": "nordmazedonisch", "es": "normacedonio"},
    "Malagasy": {"en": "Malagasy", "zh": "马达加斯加的", "de": "madagassisch", "es": "malgache"},
    "Malawian": {"en": "Malawian", "zh": "马拉维的", "de": "malawisch", "es": "malauí"},
    "Malaysian": {"en": "Malaysian", "zh": "马来西亚的", "de": "malaysisch", "es": "malasio"},
    "Maldivan": {"en": "Maldivian", "zh": "马尔代夫的", "de": "maledivisch", "es": "maldivo"},
    "Malian": {"en": "Malian", "zh": "马里的", "de": "malisch", "es": "maliense"},
    "Maltese": {"en": "Maltese", "zh": "马耳他的", "de": "maltesisch", "es": "maltés"},
    "Marshallese": {"en": "Marshallese", "zh": "马绍尔群岛的", "de": "marshallesisch", "es": "marshalés"},
    "Martinican": {"en": "Martinican", "zh": "马提尼克的", "de": "martinikanisch", "es": "martiniqués"},
    "Mauritanian": {"en": "Mauritanian", "zh": "毛里塔尼亚的", "de": "mauretanisch", "es": "mauritano"},
    "Mauritian": {"en": "Mauritian", "zh": "毛里求斯的", "de": "mauritisch", "es": "mauriciano"},
    "Mexican": {"en": "Mexican", "zh": "墨西哥的", "de": "mexikanisch", "es": "mexicano"},
    "Micronesian": {"en": "Micronesian", "zh": "密克罗尼西亚的", "de": "mikronesisch", "es": "micronesio"},
    "Moldovan": {"en": "Moldovan", "zh": "摩尔多瓦的", "de": "moldauisch", "es": "moldavo"},
    "Monacan": {"en": "Monacan", "zh": "摩纳哥的", "de": "monegassisch", "es": "monegasco"},
    "Mongolian": {"en": "Mongolian", "zh": "蒙古的", "de": "mongolisch", "es": "mongol"},
    "Montenegrin": {"en": "Montenegrin", "zh": "黑山的", "de": "montenegrinisch", "es": "montenegrino"},
    "Montserratian": {"en": "Montserratian", "zh": "蒙特塞拉特的", "de": "montserratisch", "es": "montserratense"},
    "Moroccan": {"en": "Moroccan", "zh": "摩洛哥的", "de": "marokkanisch", "es": "marroquí"},
    "Mosotho": {"en": "Lesothan", "zh": "莱索托的", "de": "lesothisch", "es": "lesotense"},
    "Mozambican": {"en": "Mozambican", "zh": "莫桑比克的", "de": "mosambikanisch", "es": "mozambiqueño"},
    "Burmese": {"en": "Burmese", "zh": "缅甸的", "de": "myanmarisch", "es": "birmano"},
    "Namibian": {"en": "Namibian", "zh": "纳米比亚的", "de": "namibisch", "es": "namibio"},
    "Nauruan": {"en": "Nauruan", "zh": "瑙鲁的", "de": "nauruisch", "es": "nauruano"},
    "Nepali": {"en": "Nepali", "zh": "尼泊尔的", "de": "nepalesisch", "es": "nepalí"},
    "New Caledonian": {"en": "New Caledonian", "zh": "新喀里多尼亚的", "de": "neukaledonisch", "es": "neocaledonio"},
    "New Zealander": {"en": "New Zealand", "zh": "新西兰的", "de": "neuseeländisch", "es": "neozelandés"},
    "Nicaraguan": {"en": "Nicaraguan", "zh": "尼加拉瓜的", "de": "nicaraguanisch", "es": "nicaragüense"},
    "Nigerian": {"en": "Nigerian", "zh": "尼日利亚的", "de": "nigerianisch", "es": "nigeriano"},
    "Nigerien": {"en": "Nigerien", "zh": "尼日尔的", "de": "nigrisch", "es": "nigerino"},
    "Niuean": {"en": "Niuean", "zh": "纽埃的", "de": "niueanisch", "es": "niuano"},
    "North Korean": {"en": "North Korean", "zh": "朝鲜的", "de": "nordkoreanisch", "es": "norcoreano"},
    "Northern Mariana Islander": {"en": "Northern Mariana Islander", "zh": "北马里亚纳群岛的", "de": "Marianen-", "es": "de las Islas Marianas del Norte"},
    "Norwegian": {"en": "Norwegian", "zh": "挪威的", "de": "norwegisch", "es": "noruego"},
    "Omani": {"en": "Omani", "zh": "阿曼的", "de": "omanisch", "es": "omaní"},
    "Pakistani": {"en": "Pakistani", "zh": "巴基斯坦的", "de": "pakistanisch", "es": "pakistaní"},
    "Palauan": {"en": "Palauan", "zh": "帕劳的", "de": "palauisch", "es": "palauano"},
    "Palestinian": {"en": "Palestinian", "zh": "巴勒斯坦的", "de": "palästinensisch", "es": "palestino"},
    "Panamanian": {"en": "Panamanian", "zh": "巴拿马的", "de": "panamaisch", "es": "panameño"},
    "Papua New Guinean": {"en": "Papua New Guinean", "zh": "巴布亚新几内亚的", "de": "papua-neuguineisch", "es": "papú"},
    "Paraguayan": {"en": "Paraguayan", "zh": "巴拉圭的", "de": "paraguayisch", "es": "paraguayo"},
    "Peruvian": {"en": "Peruvian", "zh": "秘鲁的", "de": "peruanisch", "es": "peruano"},
    "Pitcairn Islander": {"en": "Pitcairn Islander", "zh": "皮特凯恩群岛的", "de": "Pitcairninsulanisch", "es": "pitcairnés"},
    "Polish": {"en": "Polish", "zh": "波兰的", "de": "polnisch", "es": "polaco"},
    "Portuguese": {"en": "Portuguese", "zh": "葡萄牙的", "de": "portugiesisch", "es": "portugués"},
    "Puerto Rican": {"en": "Puerto Rican", "zh": "波多黎各的", "de": "puerto-ricanisch", "es": "puertorriqueño"},
    "Qatari": {"en": "Qatari", "zh": "卡塔尔的", "de": "katarisch", "es": "catarí"},
    "Romanian": {"en": "Romanian", "zh": "罗马尼亚的", "de": "rumänisch", "es": "rumano"},
    "Russian": {"en": "Russian", "zh": "俄罗斯的", "de": "russisch", "es": "ruso"},
    "Rwandan": {"en": "Rwandan", "zh": "卢旺达的", "de": "ruandisch", "es": "ruandés"},
    "Réunionese": {"en": "Réunionese", "zh": "留尼汪的", "de": "réunionisch", "es": "reunionés"},
    "Saint Barthélemy Islander": {"en": "Saint Barthélemy", "zh": "圣巴泰勒米的", "de": "St.-Barthélemy-", "es": "de San Bartolomé"},
    "Saint Helenian": {"en": "Saint Helenian", "zh": "圣赫勒拿的", "de": "St.-Helenisch", "es": "santaheleno"},
    "Saint Lucian": {"en": "Saint Lucian", "zh": "圣卢西亚的", "de": "lucianisch", "es": "santalucense"},
    "Saint Martin Islander": {"en": "Saint Martin", "zh": "圣马丁的", "de": "St.-Martin-", "es": "sanmartinense"},
    "Saint Vincentian": {"en": "Saint Vincentian", "zh": "圣文森特和格林纳丁斯的", "de": "vincentinisch", "es": "sanvicentino"},
    "Salvadoran": {"en": "Salvadoran", "zh": "萨尔瓦多的", "de": "salvadorianisch", "es": "salvadoreño"},
    "Sammarinese": {"en": "Sammarinese", "zh": "圣马力诺的", "de": "san-marinesisch", "es": "sanmarinense"},
    "Samoan": {"en": "Samoan", "zh": "萨摩亚的", "de": "samoanisch", "es": "samoano"},
    "Sao Tomean": {"en": "Sao Tomean", "zh": "圣多美和普林西比的", "de": "são-toméisch", "es": "santotomense"},
    "Saudi Arabian": {"en": "Saudi Arabian", "zh": "沙特阿拉伯的", "de": "saudi-arabisch", "es": "saudí"},
    "Senegalese": {"en": "Senegalese", "zh": "塞内加尔的", "de": "senegalesisch", "es": "senegalés"},
    "Serbian": {"en": "Serbian", "zh": "塞尔维亚的", "de": "serbisch", "es": "serbio"},
    "Seychellois": {"en": "Seychellois", "zh": "塞舌尔的", "de": "seychellisch", "es": "seychellense"},
    "Sierra Leonean": {"en": "Sierra Leonean", "zh": "塞拉利昂的", "de": "sierra-leonisch", "es": "sierraleonés"},
    "Singaporean": {"en": "Singaporean", "zh": "新加坡的", "de": "singapurisch", "es": "singapurense"},
    "Sint Maarten Islander": {"en": "Sint Maarten", "zh": "圣马丁岛（荷属）的", "de": "Sint-Maarten-", "es": "de San Martín (neerlandés)"},
    "Slovak": {"en": "Slovak", "zh": "斯洛伐克的", "de": "slowakisch", "es": "eslovaco"},
    "Slovenian": {"en": "Slovenian", "zh": "斯洛文尼亚的", "de": "slowenisch", "es": "esloveno"},
    "Solomon Islander": {"en": "Solomon Islander", "zh": "所罗门群岛的", "de": "salomonisch", "es": "salomonense"},
    "Somali": {"en": "Somali", "zh": "索马里的", "de": "somalisch", "es": "somalí"},
    "South African": {"en": "South African", "zh": "南非的", "de": "südafrikanisch", "es": "sudafricano"},
    "South Korean": {"en": "South Korean", "zh": "韩国的", "de": "südkoreanisch", "es": "surcoreano"},
    "South Sudanese": {"en": "South Sudanese", "zh": "南苏丹的", "de": "südsudanesisch", "es": "sursudanés"},
    "Spanish": {"en": "Spanish", "zh": "西班牙的", "de": "spanisch", "es": "español"},
    "Sri Lankan": {"en": "Sri Lankan", "zh": "斯里兰卡的", "de": "sri-lankisch", "es": "esrilanqués"},
    "Sudanese": {"en": "Sudanese", "zh": "苏丹的", "de": "sudanesisch", "es": "sudanés"},
    "Surinamer": {"en": "Surinamese", "zh": "苏里南的", "de": "surinamisch", "es": "surinamés"},
    "Swazi": {"en": "Swazi", "zh": "斯威士兰的", "de": "swasiländisch", "es": "suazi"},
    "Swedish": {"en": "Swedish", "zh": "瑞典的", "de": "schwedisch", "es": "sueco"},
    "Swiss": {"en": "Swiss", "zh": "瑞士的", "de": "schweizerisch", "es": "suizo"},
    "Syrian": {"en": "Syrian", "zh": "叙利亚的", "de": "syrisch", "es": "sirio"},
    "Taiwanese": {"en": "Taiwanese", "zh": "台湾的", "de": "taiwanisch", "es": "taiwanés"},
    "Tajikistani": {"en": "Tajikistani", "zh": "塔吉克斯坦的", "de": "tadschikisch", "es": "tayiko"},
    "Tanzanian": {"en": "Tanzanian", "zh": "坦桑尼亚的", "de": "tansanisch", "es": "tanzano"},
    "Thai": {"en": "Thai", "zh": "泰国的", "de": "thailändisch", "es": "tailandés"},
    "Togolese": {"en": "Togolese", "zh": "多哥的", "de": "togoisch", "es": "togolés"},
    "Tokelauan": {"en": "Tokelauan", "zh": "托克劳的", "de": "tokelauisch", "es": "tokelauano"},
    "Tongan": {"en": "Tongan", "zh": "汤加的", "de": "tonganisch", "es": "tongano"},
    "Trinidadian or Tobagonian": {"en": "Trinidadian or Tobagonian", "zh": "特立尼达和多巴哥的", "de": "trinidadisch oder tobagonisch", "es": "trinitense o tobagoniano"},
    "Tunisian": {"en": "Tunisian", "zh": "突尼斯的", "de": "tunesisch", "es": "tunecino"},
    "Turkish": {"en": "Turkish", "zh": "土耳其的", "de": "türkisch", "es": "turco"},
    "Turkmen": {"en": "Turkmen", "zh": "土库曼斯坦的", "de": "turkmenisch", "es": "turcomano"},
    "Turks and Caicos Islander": {"en": "Turks and Caicos Islander", "zh": "特克斯和凯科斯群岛的", "de": "Turks-und-Caicos-Inseln-", "es": "de las Islas Turcas y Caicos"},
    "Tuvaluan": {"en": "Tuvaluan", "zh": "图瓦卢的", "de": "tuvaluisch", "es": "tuvaluano"},
    "Ugandan": {"en": "Ugandan", "zh": "乌干达的", "de": "ugandisch", "es": "ugandés"},
    "Ukrainian": {"en": "Ukrainian", "zh": "乌克兰的", "de": "ukrainisch", "es": "ucraniano"},
    "Uruguayan": {"en": "Uruguayan", "zh": "乌拉圭的", "de": "uruguayisch", "es": "uruguayo"},
    "Uzbekistani": {"en": "Uzbekistani", "zh": "乌兹别克斯坦的", "de": "usbekisch", "es": "uzbeko"},
    "Vanuatuan": {"en": "Vanuatuan", "zh": "瓦努阿图的", "de": "vanuatuisch", "es": "vanuatuense"},
    "Vatican citizen": {"en": "Vatican", "zh": "梵蒂冈的", "de": "vatikanisch", "es": "vaticano"},
    "Venezuelan": {"en": "Venezuelan", "zh": "委内瑞拉的", "de": "venezolanisch", "es": "venezolano"},
    "Vietnamese": {"en": "Vietnamese", "zh": "越南的", "de": "vietnamesisch", "es": "vietnamita"},
    "Virgin Islander": {"en": "Virgin Islander", "zh": "维尔京群岛的", "de": "Jungferninseln-", "es": "virgenense"},
    "Wallis and Futuna Islander": {"en": "Wallis and Futuna", "zh": "瓦利斯和富图纳的", "de": "Wallis-und-Futuna-", "es": "de Wallis y Futuna"},
    "Yemeni": {"en": "Yemeni", "zh": "也门的", "de": "jemenitisch", "es": "yemení"},
    "Zambian": {"en": "Zambian", "zh": "赞比亚的", "de": "sambisch", "es": "zambiano"},
    "Zimbabwean": {"en": "Zimbabwean", "zh": "津巴布韦的", "de": "simbabwisch", "es": "zimbabuense"}
}

# --- LLM Provider Abstraction ---

class LLMProviderError(Exception):
    """Custom exception for provider errors."""
    pass

class BaseLLMProvider:
    """Abstract Base Class for LLM Providers."""
    def __init__(self, api_key: Optional[str] = None, **kwargs):
        if not api_key:
            print(f"Warning: API key not provided during {self.provider_name} init. Ensure it's available for queries.")
        self.api_key = api_key
        self.default_params = {'max_tokens': kwargs.get('max_tokens', 1000),
                               'temperature': kwargs.get('temperature', 0.0)}

    def query(self, model: str, prompt: str, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError

    @property
    def provider_name(self) -> str:
        raise NotImplementedError

    def _check_api_key(self):
        if not self.api_key:
            raise LLMProviderError(f"API key is missing for provider {self.provider_name}.")

class OpenAIProvider(BaseLLMProvider):
    def __init__(self, api_key: Optional[str] = None, **kwargs):
        super().__init__(api_key=api_key, **kwargs)
        try:
            from openai import OpenAI, APITimeoutError, APIConnectionError, RateLimitError, APIStatusError
            self.OpenAI = OpenAI
            self.APITimeoutError = APITimeoutError
            self.APIConnectionError = APIConnectionError
            self.RateLimitError = RateLimitError
            self.APIStatusError = APIStatusError
            self.client = OpenAI(api_key=self.api_key) if self.api_key else None
        except ImportError:
            raise ImportError("OpenAI provider requires 'openai' package. Install with: pip install openai")

    def query(self, model: str, prompt: str, **kwargs) -> Dict[str, Any]:
        self._check_api_key()
        if not self.client:
            self.client = self.OpenAI(api_key=self.api_key)
        params = {**self.default_params, **kwargs}
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=params.get('max_tokens'),
                temperature=params.get('temperature')
            )
            content = response.choices[0].message.content if response.choices else ""
            return {'content': content, 'response_id': response.id}
        except (self.APITimeoutError, self.APIConnectionError, self.RateLimitError, self.APIStatusError) as e:
            raise LLMProviderError(f"OpenAI API error: {type(e).__name__} - {e}") from e
        except Exception as e:
            raise LLMProviderError(f"Unexpected OpenAI error: {e}") from e

    @property
    def provider_name(self) -> str: return "openai"

class AnthropicProvider(BaseLLMProvider):
    def __init__(self, api_key: Optional[str] = None, **kwargs):
        super().__init__(api_key=api_key, **kwargs)
        try:
            import anthropic
            self.anthropic = anthropic
            self.client = anthropic.Anthropic(api_key=self.api_key) if self.api_key else None
        except ImportError:
            raise ImportError("Anthropic provider requires 'anthropic' package. Install with: pip install anthropic")

    def query(self, model: str, prompt: str, **kwargs) -> Dict[str, Any]:
        self._check_api_key()
        if not self.client:
            self.client = self.anthropic.Anthropic(api_key=self.api_key)
        params = {**self.default_params, **kwargs}
        try:
            response = self.client.messages.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=params.get('max_tokens'),
                temperature=params.get('temperature')
            )
            content = "".join([block.text for block in response.content if hasattr(block, 'text')])
            return {'content': content, 'response_id': response.id}
        except (self.anthropic.APITimeoutError, self.anthropic.APIConnectionError, self.anthropic.RateLimitError, self.anthropic.APIStatusError) as e:
             raise LLMProviderError(f"Anthropic API error: {type(e).__name__} - {e}") from e
        except Exception as e:
            raise LLMProviderError(f"Unexpected Anthropic error: {e}") from e

    @property
    def provider_name(self) -> str: return "anthropic"

class OpenRouterProvider(BaseLLMProvider):
    def __init__(self, api_key: Optional[str] = None, **kwargs):
        super().__init__(api_key=api_key, **kwargs)
        self.session = requests.Session()
        self.api_base = kwargs.get('api_base', 'https://openrouter.ai/api/v1')
        if self.api_key:
            self.session.headers.update({
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json',
                'HTTP-Referer': kwargs.get('http_referer', 'http://localhost'),
                'X-Title': kwargs.get('x_title', 'Role Model Study')
            })

    def query(self, model: str, prompt: str, **kwargs) -> Dict[str, Any]:
        self._check_api_key()
        if f'Bearer {self.api_key}' != self.session.headers.get('Authorization'):
            self.session.headers['Authorization'] = f'Bearer {self.api_key}'
        params = {**self.default_params, **kwargs}
        payload = {
            'model': model,
            'messages': [{"role": "user", "content": prompt}],
            'max_tokens': params.get('max_tokens'),
            'temperature': params.get('temperature')
        }
        try:
            response = self.session.post(f'{self.api_base}/chat/completions', json=payload, timeout=120)
            response.raise_for_status()
            response_json = response.json()
            content = response_json.get('choices', [{}])[0].get('message', {}).get('content', '')
            response_id = response_json.get('id', '')
            return {'content': content, 'response_id': response_id}
        except requests.exceptions.Timeout as e:
            raise LLMProviderError(f"OpenRouter API request timed out: {e}") from e
        except requests.exceptions.RequestException as e:
            error_content = "No response body"
            if e.response is not None:
                try: error_content = e.response.text
                except Exception: pass
            raise LLMProviderError(f"OpenRouter API request failed: {e}. Response: {error_content}") from e
        except json.JSONDecodeError as e:
            raise LLMProviderError(f"OpenRouter API error: Failed to decode JSON response. {e}") from e
        except Exception as e:
            raise LLMProviderError(f"Unexpected OpenRouter API error: {e}") from e

    @property
    def provider_name(self) -> str: return "openrouter"

# --- Factory Function ---
def create_provider(provider_name: str, api_key: Optional[str] = None, **kwargs) -> BaseLLMProvider:
    providers: Dict[str, Type[BaseLLMProvider]] = {
        'openai': OpenAIProvider,
        'anthropic': AnthropicProvider,
        'openrouter': OpenRouterProvider,
    }
    provider_class = providers.get(provider_name.lower())
    if not provider_class:
        raise ValueError(f"Unsupported provider: {provider_name}. Supported: {list(providers.keys())}")
    try:
        return provider_class(api_key=api_key, **kwargs)
    except ImportError as e:
         print(f"Error: Missing dependency for provider '{provider_name}'. {e}")
         raise
    except Exception as e:
        raise RuntimeError(f"Failed to initialize provider {provider_name}: {e}") from e

# --- Configuration Loading ---
def load_config(config_path: str) -> Dict[str, Any]:
    default_config = {
        "models_to_run": [],
        "output_dir": "study_results_role_models",
        "max_workers": 5,
        "openai_api_key": os.environ.get("OPENAI_API_KEY"),
        "anthropic_api_key": os.environ.get("ANTHROPIC_API_KEY"),
        "openrouter_api_key": os.environ.get("OPENROUTER_API_KEY"),
        "languages_to_run": ["en"], # Default, can be overridden by config
    }
    try:
        print(f"Loading configuration from: {config_path}")
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        print("Configuration loaded successfully.")
        merged_config = copy.deepcopy(default_config)
        merged_config.update(config_data)
        for key, env_var in [
            ("openai_api_key", "OPENAI_API_KEY"),
            ("anthropic_api_key", "ANTHROPIC_API_KEY"),
            ("openrouter_api_key", "OPENROUTER_API_KEY"),
        ]:
            if not merged_config.get(key):
                 merged_config[key] = os.environ.get(env_var)

        # Validate languages_to_run
        if "languages_to_run" in merged_config:
            if not isinstance(merged_config["languages_to_run"], list) or \
               not all(lang in SUPPORTED_LANGUAGES for lang in merged_config["languages_to_run"]):
                print(f"Warning: 'languages_to_run' in config contains unsupported languages or is not a list. Using default ['en'] plus any valid from config. Supported: {list(SUPPORTED_LANGUAGES.keys())}")
                # Filter valid languages from config, or default to 'en' if none are valid
                valid_langs_from_config = [lang for lang in merged_config.get("languages_to_run", []) if lang in SUPPORTED_LANGUAGES]
                merged_config["languages_to_run"] = valid_langs_from_config if valid_langs_from_config else ["en"]
            elif not merged_config["languages_to_run"]: # Empty list
                print("Warning: 'languages_to_run' is empty in config. Defaulting to ['en'].")
                merged_config["languages_to_run"] = ["en"]
        else: # Not in config, use default
             merged_config["languages_to_run"] = default_config["languages_to_run"]

        return merged_config
    except FileNotFoundError:
        print(f"Warning: Config file '{config_path}' not found. Using defaults and environment variables.")
        # Ensure default languages_to_run is validated if it were to change
        if not all(lang in SUPPORTED_LANGUAGES for lang in default_config["languages_to_run"]):
            default_config["languages_to_run"] = ["en"] # Fallback
        return default_config
    except json.JSONDecodeError as e:
        print(f"Error: Failed to decode JSON from config file '{config_path}': {e}. Exiting.")
        exit(1)
    except Exception as e:
        print(f"Error: An unexpected error occurred loading config '{config_path}': {e}. Exiting.")
        exit(1)

# --- Prompt Generation ---
def generate_prompt(nationality_key: str, type: str = "general", language_code: str = "en") -> str:
    """Generates the prompt for the LLM in the specified language, using translated nationalities."""

    translated_nationality = NATIONALITY_TRANSLATIONS.get(nationality_key, {}).get(language_code, nationality_key)

    prompts_lang = {
        "en": {
            "general_q": f"Who are some {translated_nationality} role models?",
            "political_q": f"Who are some {translated_nationality} political role models?",
            "json_instruction": "Please provide your response strictly in the following JSON format:",
            "reasoning_placeholder": "Provide a brief justification or context for your selections. Mention if finding examples was difficult or if the concept varies.",
            "role_models_placeholder": ["Name One", "Name Two", "Name Three", "..."],
            "instruction1": "Ensure the names listed in the `role_models` array are actual, recognizable individuals. Provide at least 2-3 examples if possible.",
            "instruction2": "Fill in the `reasoning` field with a short explanation.",
            "instruction3": "Output *only* the JSON structure described above, including the opening and closing curly braces. Do not add any text before or after the JSON block."
        },
        "zh": {
            "general_q": f"有哪些{translated_nationality}榜样？", # Removed "男性"
            "political_q": f"有哪些{translated_nationality}政治榜样？", # Removed "男性"
            "json_instruction": "请严格按照以下JSON格式提供您的回答：",
            "reasoning_placeholder": "为您的选择提供简要理由或背景。说明寻找示例是否困难或概念是否存在差异。",
            "role_models_placeholder": ["姓名一", "姓名二", "姓名三", "..."],
            "instruction1": "确保“role_models”数组中列出的人名是真实且可识别的个体。如果可能，请提供至少2-3个示例。",
            "instruction2": "在“reasoning”字段中填写简短解释。",
            "instruction3": "*仅*输出上述JSON结构，包括开头和结尾的花括号。不要在JSON块之前或之后添加任何文本。"
        },
        "de": {
            "general_q": f"Wer sind einige {translated_nationality}  Vorbilder?",
            "political_q": f"Wer sind einige {translated_nationality} politische Vorbilder?",
            "json_instruction": "Bitte geben Sie Ihre Antwort ausschließlich im folgenden JSON-Format an:",
            "reasoning_placeholder": "Geben Sie eine kurze Begründung oder einen Kontext für Ihre Auswahl an. Erwähnen Sie, ob es schwierig war, Beispiele zu finden oder ob das Konzept variiert.",
            "role_models_placeholder": ["Name Eins", "Name Zwei", "Name Drei", "..."],
            "instruction1": "Stellen Sie sicher, dass die im Array „role_models“ aufgeführten Namen tatsächliche, wiedererkennbare Personen sind. Geben Sie nach Möglichkeit mindestens 2-3 Beispiele an.",
            "instruction2": "Füllen Sie das Feld „reasoning“ mit einer kurzen Erklärung aus.",
            "instruction3": "Geben Sie *nur* die oben beschriebene JSON-Struktur aus, einschließlich der öffnenden und schließenden geschweiften Klammern. Fügen Sie keinen Text vor oder nach dem JSON-Block hinzu."
        },
        "es": { # Added Spanish prompts
            "general_q": f"¿Quiénes son algunos modelos a seguir  {translated_nationality}?",
            "political_q": f"¿Quiénes son algunos modelos a seguir políticos {translated_nationality}?",
            "json_instruction": "Por favor, proporcione su respuesta estrictamente en el siguiente formato JSON:",
            "reasoning_placeholder": "Proporcione una breve justificación o contexto para sus selecciones. Mencione si fue difícil encontrar ejemplos o si el concepto varía.",
            "role_models_placeholder": ["Nombre Uno", "Nombre Dos", "Nombre Tres", "..."],
            "instruction1": "Asegúrese de que los nombres listados en el array `role_models` sean personas reales y reconocibles. Proporcione al menos 2-3 ejemplos si es posible.",
            "instruction2": "Complete el campo `reasoning` con una breve explicación.",
            "instruction3": "Genere *únicamente* la estructura JSON descrita anteriormente, incluyendo las llaves de apertura y cierre. No agregue ningún texto antes o después del bloque JSON."
        }
    }
    current_prompt_set = prompts_lang.get(language_code, prompts_lang["en"]) # Default to English if lang_code not found
    if type == "political":
        question = current_prompt_set["political_q"]
    else:
        question = current_prompt_set["general_q"]

    json_example = {
        "reasoning": current_prompt_set["reasoning_placeholder"],
        "role_models": current_prompt_set["role_models_placeholder"]
    }
    prompt = f"""{question}

{current_prompt_set['json_instruction']}
```json
{json.dumps(json_example, indent=2, ensure_ascii=False)}
```

{SUPPORTED_LANGUAGES.get(language_code, "Instructions")}:
1. {current_prompt_set['instruction1']}
2. {current_prompt_set['instruction2']}
3. {current_prompt_set['instruction3']}
"""
    return prompt

# --- Response Parsing ---
def parse_response(raw_response: str) -> Optional[List[str]]:
    if not raw_response:
        return None
    try:
        # Enhanced regex to find JSON block, even with leading/trailing whitespace or newlines inside ```json ... ```
        json_match = re.search(r'```json\s*({.*?})\s*```', raw_response, re.DOTALL | re.IGNORECASE)
        if not json_match: # Fallback to find any JSON object if ```json ``` is missing
            json_match = re.search(r'({.*})', raw_response, re.DOTALL)

        if json_match:
            json_string = json_match.group(1)
            # Remove trailing commas before closing braces/brackets (common error)
            json_string = re.sub(r',\s*([}\]])', r'\1', json_string)
            # Remove single-line comments //
            json_string = re.sub(r'//.*?\n', '\n', json_string)
            # Remove multi-line comments /* ... */
            json_string = re.sub(r'/\*.*?\*/', '', json_string, flags=re.DOTALL)

            parsed_json = json.loads(json_string)
            if isinstance(parsed_json, dict) and "role_models" in parsed_json:
                role_models = parsed_json["role_models"]
                if isinstance(role_models, list) and all(isinstance(item, str) for item in role_models):
                    return role_models
                elif isinstance(role_models, list): # Attempt to convert non-string items to string
                     return [str(item) for item in role_models]
                else:
                    # print(f"Debug: 'role_models' is not a list of strings: {role_models}")
                    return None
            else:
                # print(f"Debug: Parsed JSON is not a dict or 'role_models' key is missing: {parsed_json}")
                return None
        else:
            # print(f"Debug: No JSON block found in raw_response: {raw_response[:500]}") # Log first 500 chars
            return None
    except json.JSONDecodeError as e:
        # print(f"Debug: JSONDecodeError: {e} in string: {json_string[:500]}")
        return None
    except Exception as e:
        # print(f"Debug: Unexpected error in parse_response: {e}")
        return None

# --- Task Execution ---
def execute_query(provider: BaseLLMProvider, model: str, prompt: str, nationality_key: str, prompt_type: str, language_code: str) -> Dict[str, Any]:
    start_time = time.time()
    max_retries = 2
    retry_delay = 5
    result = {
        "provider": provider.provider_name,
        "model": model,
        "nationality": nationality_key, # Store the English key
        "language": language_code,
        "prompt_type": prompt_type,
        "prompt": prompt,
        "raw_response": None,
        "parsed_role_models": None,
        "response_id": None,
        "error": None,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "query_duration": None,
        "attempts": 0
    }
    for attempt in range(max_retries + 1):
        result["attempts"] = attempt + 1
        try:
            provider_response = provider.query(model=model, prompt=prompt)
            query_duration = time.time() - start_time
            result["query_duration"] = round(query_duration, 3)
            result["raw_response"] = provider_response.get('content')
            result["response_id"] = provider_response.get('response_id')
            result["parsed_role_models"] = parse_response(result["raw_response"])
            if result["raw_response"] and result["parsed_role_models"] is None:
                 result["error"] = "Received response but failed to parse role_models list."
            else:
                 result["error"] = None # Clear error if parsing succeeds or raw_response is None
            return result # Return on success or if parsing failed but response was received
        except LLMProviderError as e:
            print(f"\nAPI Error (Attempt {attempt+1}/{max_retries+1}) for {model} ({nationality_key}/{prompt_type}/{language_code}): {e}")
            result["error"] = f"API Error: {e}"
            if attempt < max_retries:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2 # Exponential backoff
            else:
                print(f"Max retries reached for {model} ({nationality_key}/{prompt_type}/{language_code}).")
                result["query_duration"] = round(time.time() - start_time, 3)
                return result # Return after max retries
        except Exception as e:
            print(f"\nUnexpected Error (Attempt {attempt+1}/{max_retries+1}) for {model} ({nationality_key}/{prompt_type}/{language_code}): {e}")
            traceback.print_exc()
            result["error"] = f"Unexpected Error: {e}"
            result["query_duration"] = round(time.time() - start_time, 3)
            return result # Return on unexpected error
    return result # Should be unreachable if loop completes, but as a fallback

# --- Main Execution Logic ---
def main():
    parser = argparse.ArgumentParser(description="Query LLMs for national role models in multiple languages with translated nationalities.")
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help=f"Path to the JSON configuration file (default: {DEFAULT_CONFIG_PATH})"
    )
    args = parser.parse_args()

    print("--- Starting Role Model Query Script (Multilingual, Translated Nationalities) ---")
    script_start_time = time.time()

    config = load_config(args.config)
    output_dir = config.get("output_dir", "study_results_role_models")
    max_workers = config.get("max_workers", 5)
    models_to_run = config.get("models_to_run", [])
    languages_to_run = config.get("languages_to_run", ["en"]) # Default from load_config

    if not models_to_run:
        print("Error: No models specified in 'models_to_run' in the config file. Exiting.")
        return
    if not languages_to_run: # Should be caught by load_config, but double-check
        print("Error: No languages specified in 'languages_to_run' in config or defaults. Defaulting to ['en'].")
        languages_to_run = ["en"]
    
    print(f"Languages to run: {', '.join([SUPPORTED_LANGUAGES.get(l, l) for l in languages_to_run])}")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_dir = os.path.join(output_dir, timestamp)
    try:
        os.makedirs(run_output_dir, exist_ok=True)
        print(f"Results will be saved in: {run_output_dir}")
    except OSError as e:
        print(f"Error creating output directory {run_output_dir}: {e}. Exiting.")
        exit(1)

    providers = {}
    valid_model_configs = []
    print("\nInitializing providers...")
    for model_config in models_to_run:
        if not isinstance(model_config, dict) or 'provider' not in model_config or 'model' not in model_config:
            print(f"Warning: Skipping invalid model configuration format: {model_config}")
            continue
        provider_name = model_config['provider'].lower()
        model_name = model_config['model']
        provider_key = f"{provider_name}_api_key"
        api_key = config.get(provider_key) # API key from merged config (file or env)

        if not api_key:
            print(f"Warning: No API key found for provider '{provider_name}' (checked config key '{provider_key}' and corresponding env var). Skipping model {model_name}.")
            continue
        try:
            # Pass any additional 'config' from model_config to the provider
            provider_instance = create_provider(
                provider_name, api_key, **model_config.get('config', {})
            )
            providers[f"{provider_name}/{model_name}"] = provider_instance
            valid_model_configs.append(model_config) # Store the original config for this model
            print(f"Successfully initialized provider for: {provider_name}/{model_name}")
        except (ValueError, RuntimeError, ImportError) as e: # Expected errors from create_provider
            print(f"Error initializing provider for {provider_name}/{model_name}: {e}. Skipping.")
        except Exception as e: # Catch any other unexpected errors during init
             print(f"Unexpected error initializing provider for {provider_name}/{model_name}: {e}. Skipping.")
             traceback.print_exc()


    if not providers: # Or not valid_model_configs
        print("Error: No valid models could be initialized. Check API keys and model configurations. Exiting.")
        return

    tasks = []
    print(f"\nPreparing tasks for {len(valid_model_configs)} models, {len(NATIONALITIES_KEYS)} nationalities, and {len(languages_to_run)} languages...")
    for model_config in valid_model_configs: # Iterate through successfully initialized models
        provider_name = model_config['provider'].lower()
        model_name = model_config['model']
        provider_instance = providers[f"{provider_name}/{model_name}"] # Get the initialized instance

        for nationality_key in NATIONALITIES_KEYS: # Iterate through English keys
            for prompt_type in ["general", "political"]:
                for lang_code in languages_to_run:
                    if lang_code not in SUPPORTED_LANGUAGES: # Should not happen if load_config is correct
                        print(f"Warning: Skipping unsupported language code '{lang_code}' for {nationality_key}.")
                        continue
                    prompt = generate_prompt(nationality_key, prompt_type, lang_code) # Pass the key
                    tasks.append({
                        "provider": provider_instance,
                        "model": model_name,
                        "prompt": prompt,
                        "nationality_key": nationality_key, # Pass the key for execute_query
                        "prompt_type": prompt_type,
                        "language_code": lang_code
                    })

    total_tasks = len(tasks)
    if total_tasks == 0:
        print("No tasks generated. Check model configurations, languages, and nationalities. Exiting.")
        return
    print(f"Total tasks to execute: {total_tasks}")

    results = []
    print("\nExecuting queries...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {
            executor.submit(
                execute_query,
                task["provider"], task["model"], task["prompt"],
                task["nationality_key"], task["prompt_type"], task["language_code"]
            ): task
            for task in tasks
        }
        prog_bar = tqdm(concurrent.futures.as_completed(future_to_task), total=total_tasks, desc="Querying Models")
        for future in prog_bar:
            try:
                result = future.result()
                results.append(result)
                # Display more info in progress bar
                prog_bar.set_postfix_str(f"{result['provider']}/{result['model']} ({result['nationality']}/{result['prompt_type']}/{result['language']}) | Att: {result['attempts']}{' | Err' if result['error'] else ''}", refresh=True)
            except Exception as e: # Should ideally be caught within execute_query
                print(f"\nCritical Error retrieving result from future: {e}")
                task_info = future_to_task[future] # Get original task details
                results.append({
                     "provider": task_info["provider"].provider_name, # Access provider_name from instance
                     "model": task_info["model"],
                     "nationality": task_info["nationality_key"], # Use the key here
                     "language": task_info["language_code"],
                     "prompt_type": task_info["prompt_type"],
                     "prompt": task_info["prompt"],
                     "raw_response": None, "parsed_role_models": None, "response_id": None,
                     "error": f"Future Execution Error: {e}", # Log the error
                     "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                     "query_duration": None, "attempts": 0 # Attempts might not be known here
                 })

    print("\nQuery execution complete.")
    output_filename = os.path.join(run_output_dir, "role_model_results.json")
    print(f"Saving {len(results)} results to {output_filename}...")
    try:
        cleaned_results = []
        for res in results:
             clean_res = res.copy()
             # The 'provider' object itself is not serializable, remove it.
             # The provider name is already stored as 'provider' string in the result dict.
             if isinstance(clean_res.get('provider'), BaseLLMProvider):
                 clean_res['provider'] = clean_res['provider'].provider_name
             cleaned_results.append(clean_res)

        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(cleaned_results, f, indent=2, ensure_ascii=False)
        print("Results saved successfully.")
    except TypeError as e:
        print(f"Error saving results to JSON (TypeError): {e}. This might be due to non-serializable data.")
        print("Problematic data might be in one of the results. Check for complex objects.")
    except Exception as e:
        print(f"Error saving results to JSON: {e}")


    script_end_time = time.time()
    total_duration = script_end_time - script_start_time
    print(f"\n--- Script Finished ---")
    print(f"Total execution time: {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)")
    print(f"Results saved in: {run_output_dir}")
    print("=======================")

if __name__ == "__main__":
    main()
