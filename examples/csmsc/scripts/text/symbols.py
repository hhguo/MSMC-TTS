""" from https://github.com/keithito/tacotron """

'''
Defines the set of symbols used in text input to the model.

The default is a set of ASCII characters that works well for English or text that has been run through Unidecode. For other data, you can modify _characters. See TRAINING_DATA.md for details. '''

# Unspoken symbols
_pad = '<PAD>'

unspoken_symbols = [_pad, 'sil', 'sp1']

spoken_symbols = [
    'a', 'ai', 'an', 'ang', 'ao', 'b', 'c', 'ch', 'd', 'e', 'ei', 'en', 'eng',
    'er', 'f', 'g', 'h', 'i', 'ia', 'ian', 'iang', 'iao', 'ie', 'ii', 'iii',
    'in', 'ing', 'io', 'iong', 'iou', 'iyl', 'j', 'k', 'l', 'm', 'n', 'ng',
    'o', 'ong', 'ou', 'p', 'pl', 'q', 'r', 's', 'sh', 't', 'u', 'ua', 'uai',
    'uan', 'uang', 'uei', 'uen', 'ueng', 'uo', 'v', 'van', 've', 'vn', 'x',
    'z', 'zh'
]

# Export all symbols:
symbols = unspoken_symbols + spoken_symbols
