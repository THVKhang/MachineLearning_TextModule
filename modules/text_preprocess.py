import re
import string
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def setup_nltk():
    for pkg in ['punkt', 'punkt_tab', 'stopwords']:
        try:
            nltk.data.find(f'tokenizers/{pkg}' if 'punkt' in pkg else f'corpora/{pkg}')
        except LookupError:
            nltk.download(pkg)

setup_nltk()
class TextCleaner:
    def __init__(
        self,
        language='english',
        remove_stopwords=True,        
        remove_punctuation=True,     
        remove_numbers=True          
    ):
        self.stop_words = set(stopwords.words(language))
        
        self.remove_stopwords = remove_stopwords
        self.remove_punctuation = remove_punctuation
        self.remove_numbers = remove_numbers

    def clean_text(self, text):
        if not isinstance(text, str):
            return ""
            
        text = text.lower()
        
        if self.remove_punctuation:
            text = text.translate(str.maketrans('', '', string.punctuation))  
        
        if self.remove_numbers:
            text = re.sub(r'\d+', '', text)  
        
        tokens = word_tokenize(text)
        
        if self.remove_stopwords:
            tokens = [
                word for word in tokens 
                if word not in self.stop_words and word.strip() != ''
            ]  
        else:
            tokens = [word for word in tokens if word.strip() != '']

        return " ".join(tokens)
        

    def clean_corpus(self, corpus):
        
        return [self.clean_text(doc) for doc in corpus]
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Tải các gói dữ liệu cần thiết từ nltk (Chỉ tải nếu máy chưa có)
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class TextCleaner:
    def __init__(self, language='english'):
        self.stop_words = set(stopwords.words(language))
        self.punctuation = set(string.punctuation)

    def clean_text(self, text):
        if not isinstance(text, str):
            return ""
            
        # 1. Chuyển thành chữ thường
        text = text.lower()
        
        # 2. Loại bỏ dấu câu và số (tùy chọn theo ngữ cảnh dữ liệu)
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = re.sub(r'\d+', '', text)
        
        # 3. Tokenization
        tokens = word_tokenize(text)
        
        # 4. Loại bỏ stopwords và các khoảng trắng thừa
        cleaned_tokens = [word for word in tokens if word not in self.stop_words and word.strip() != '']
        
        # Nối lại thành chuỗi đã làm sạch
        return " ".join(cleaned_tokens)
        
    def clean_corpus(self, corpus):
        """Áp dụng làm sạch cho một list các văn bản."""
        return [self.clean_text(doc) for doc in corpus]
