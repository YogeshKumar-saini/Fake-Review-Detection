import pandas as pd
import os
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, LabelEncoder
from langdetect import detect
import string
import re
import nltk
from symspellpy import SymSpell, Verbosity
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
import inflect
from nltk.stem import WordNetLemmatizer, SnowballStemmer
import contractions

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

df = pd.read_csv('/home/yogesh/mlops/Mlop Projects/Fake Review Detection/data/processed/translated_output.csv')

category_encoder = OrdinalEncoder()
df["category"] = category_encoder.fit_transform(df[["category"]])

label_encoder = LabelEncoder()
df["label"] = label_encoder.fit_transform(df["label"])

output_path = "/home/yogesh/mlops/Mlop Projects/Fake Review Detection/data/processed/encoded_dataset.csv"
df.to_csv(output_path, index=False)
print(f"Encoded dataset saved: {output_path}")

def func(text):
    try:
        if isinstance(text, str) and text.strip():
            return detect(text)
        else:
            return "unknown"
    except:
        return "unknown"  

temp = df['text_'].apply(func)

print("Languages found ",temp.unique())
print("Total instances where it was not english",temp[temp != 'en'].count())

import pandas as pd
import os
from deep_translator import GoogleTranslator

output_file = "/home/yogesh/mlops/Mlop Projects/Fake Review Detection/data/processed/translated_output.csv" 

if os.path.exists(output_file):
    df_translated = pd.read_csv(output_file)
    translated_indices = set(df_translated.index)  
else:
    df_translated = pd.DataFrame()
    translated_indices = set()

def deep_translate(text):
    try:
        if not isinstance(text, str) or not text.strip():
            return "Invalid or Empty Text"
        return GoogleTranslator(source='auto', target='en').translate(text)
    except Exception as e:
        raise ValueError(f"Deep Translator Error: {e}")

for i in range(len(df)):
    if i in translated_indices:
        continue  
    df.loc[i, 'deep_translated_text'] = deep_translate(df.loc[i, 'text_'])
    df.iloc[[i]].to_csv(output_file, mode='a', header=not os.path.exists(output_file), index=False)
    print(f"Processed row {i+1}/{len(df)}: {df.loc[i, 'deep_translated_text']}")  

print("Translation completed or interrupted, progress saved!")

for i in range(len(df)):
    if i in translated_indices:
        continue  
    df.loc[i, 'deep_translated_text'] = deep_translate(df.loc[i, 'text_'])
    df.iloc[[i]].to_csv(output_file, mode='a', header=not os.path.exists(output_file), index=False)
    print(f"Processed row {i+1}/{len(df)}: {df.loc[i, 'deep_translated_text']}")  

trans_df = pd.read_csv("/home/yogesh/mlops/Mlop Projects/Fake Review Detection/data/processed/translated_output.csv")
def func(text):
    try:
        if isinstance(text, str) and text.strip():
            return detect(text)
        else:
            return "unknown"
    except:
        return "unknown"  

temp = trans_df['deep_translated_text'].apply(func)

print("Languages found ",temp.unique())
print("Total instances where it was not english",temp[temp != 'en'].count())

def convert_numbers(text: str) -> str:
    """Convert numeric digits to their word representation (e.g., '4' -> 'four')."""
    return re.sub(r'\b\d+\b', lambda x: inflect_engine.number_to_words(x.group()), text)

def get_wordnet_pos(word: str):
    """Map NLTK POS tags to WordNet POS for better lemmatization."""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

inflect_engine = inflect.engine()
lemmatizer = WordNetLemmatizer()
stemmer = SnowballStemmer("english")
stop_words = set(stopwords.words("english"))

sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
dictionary_path = "/home/yogesh/mlops/Mlop Projects/Fake Review Detection/Assets/frequency_dictionary_en_82_765.txt"
if not sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1):
    raise FileNotFoundError(f"SymSpell dictionary file not found at {dictionary_path}")

def remove_emojis(text: str) -> str:
    """Remove all non-ASCII characters (including most emojis)."""
    return re.sub(r'[^\x00-\x7F]+', '', text)

def convert_numbers(text: str) -> str:
    """Convert numeric digits to their word representation (e.g., '4' -> 'four')."""
    return re.sub(r'\b\d+\b', lambda x: inflect_engine.number_to_words(x.group()), text)

def correct_spelling(word: str) -> str:
    """Correct spelling using SymSpell, returning the closest suggestion if available."""
    suggestions = sym_spell.lookup(word, Verbosity.CLOSEST, max_edit_distance=2)
    return suggestions[0].term if suggestions else word

def get_wordnet_pos(word: str):
    """Map NLTK POS tags to WordNet POS for better lemmatization."""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

def advanced_preprocess_text(
    text: str,
    remove_stopwords: bool = False,
    use_stemming: bool = False,
    use_lemmatization: bool = True,
    use_spell_correction: bool = False,
    expand_contractions_flag: bool = True
) -> str:
    """Preprocess text by removing emojis, expanding contractions, converting numbers, etc."""
    if pd.isna(text) or not isinstance(text, str):
        return ""
    text = remove_emojis(text)
    if expand_contractions_flag:
        text = contractions.fix(text)
    text = re.sub(r'<[^>]+>', '', text)
    text = text.lower()
    text = convert_numbers(text)

    text_clean = text.translate(str.maketrans("", "", string.punctuation))
    words = word_tokenize(text_clean)

    if use_spell_correction:
        words = [correct_spelling(word) for word in words]

    if remove_stopwords:
        words = [word for word in words if word not in stop_words]

    if use_stemming:
        words = [stemmer.stem(word) for word in words]
    elif use_lemmatization:
        words = [lemmatizer.lemmatize(word, get_wordnet_pos(word)) for word in words]

    return " ".join(words)

def advanced_preprocess_texts(
    df: pd.DataFrame,
    remove_stopwords: bool = False,
    use_stemming: bool = False,
    use_lemmatization: bool = True,
    use_spell_correction: bool = False,
    expand_contractions_flag: bool = True,
    filename: str = "processed_output"
) -> pd.DataFrame:
    """Apply the advanced preprocessing pipeline to the 'deep_translated_text' column of the DataFrame."""
    df = df.copy()
    df["processed_text"] = df["deep_translated_text"].apply(
        lambda x: advanced_preprocess_text(
            x,
            remove_stopwords,
            use_stemming,
            use_lemmatization,
            use_spell_correction,
            expand_contractions_flag
        )
    )
    os.makedirs("/home/yogesh/mlops/Mlop Projects/Fake Review Detection/data/processed", exist_ok=True)
    filepath = f"/home/yogesh/mlops/Mlop Projects/Fake Review Detection/data/processed/{filename}.csv"
    df.to_csv(filepath, index=False)
    print(f"Processed file saved: {filepath}")
    return df

df = pd.read_csv("/home/yogesh/mlops/Mlop Projects/Fake Review Detection/data/processed/encoded_dataset.csv")

configurations = [
    {
        "remove_stopwords": False,
        "use_stemming": False,
        "use_lemmatization": True,
        "use_spell_correction": False,
        "expand_contractions_flag": True,
        "filename": "preprocessed_lemmatization"
    },
    {
        "remove_stopwords": True,
        "use_stemming": False,
        "use_lemmatization": False,
        "use_spell_correction": False,
        "expand_contractions_flag": True,
        "filename": "preprocessed_no_stopwords_no_lemmatization"
    },
    {
        "remove_stopwords": True,
        "use_stemming": False,
        "use_lemmatization": True,
        "use_spell_correction": False,
        "expand_contractions_flag": True,
        "filename": "preprocessed_no_stopwords"
    },
    {
        "remove_stopwords": True,
        "use_stemming": True,
        "use_lemmatization": False,
        "use_spell_correction": False,
        "expand_contractions_flag": True,
        "filename": "preprocessed_stemming_no_stopwords"
    },
    {
        "remove_stopwords": False,
        "use_stemming": True,
        "use_lemmatization": False,
        "use_spell_correction": False,
        "expand_contractions_flag": True,
        "filename": "preprocessed_stemming"
    }
]

for config in configurations:
    advanced_preprocess_texts(
        df,
        remove_stopwords=config["remove_stopwords"],
        use_stemming=config["use_stemming"],
        use_lemmatization=config["use_lemmatization"],
        use_spell_correction=config["use_spell_correction"],
        expand_contractions_flag=config["expand_contractions_flag"],
        filename=config["filename"]
    )

print("All preprocessed datasets saved.")
