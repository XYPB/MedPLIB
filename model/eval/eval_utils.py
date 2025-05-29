from collections import defaultdict
import re
import math

def filter_closed_answers(prediction, answer_type):
    prediction = prediction.lower().strip()
    prediction = re.sub(r'[^a-zA-Z ]', '', prediction)  # Remove non-alphabetic characters
    prediction = prediction.split()
    if answer_type.lower() in ['yes/no', 'closed']:
        # For closed-ended questions, we only consider 'yes' or 'no' as valid answers
        prediction = [word for word in prediction if word in ['yes', 'no']]
        prediction = 'yes' if 'yes' in prediction else 'no' if 'no' in prediction else ''
        return prediction.strip() if prediction else ''  # Default to '' if no valid answer found
    elif answer_type.lower() in ['multiple_choice', 'mcq']:
        # For open-ended questions, we return the prediction as is
        print(prediction)
        prediction = [word for word in prediction if word in ['a', 'b', 'c', 'd', 'e']]  # Remove empty strings
        prediction = ''.join(prediction).strip()  # Join the remaining words
        return prediction if prediction else ''  # Default to '' if no valid answer found
    else:
        # For other types of questions, we return the prediction as is
        return ' '.join(prediction).strip()

def brevity_penalty(candidate, references):
    c = len(candidate)
    ref_lens = (len(reference) for reference in references)
    r = min(ref_lens, key=lambda ref_len: (abs(ref_len - c), ref_len))
    
    if c > r:
        return 1
    else:
        return math.exp(1 - r / c)

def modified_precision(candidate, references, n):
    max_frequency = defaultdict(int)
    min_frequency = defaultdict(int)
    
    candidate_words = split_sentence(candidate, n)
    
    for reference in references:
        reference_words = split_sentence(reference, n)
        for word in candidate_words:
            max_frequency[word] = max(max_frequency[word], reference_words[word])
    for word in candidate_words:
            min_frequency[word] = min(max_frequency[word], candidate_words[word])
    P = sum(min_frequency.values()) / sum(candidate_words.values())
    return P

def split_sentence(sentence, n):
    words = defaultdict(int)
    # tmp_sentence = re.sub("[^a-zA-Z ]", "", sentence)
    tmp_sentence = sentence
    tmp_sentence = tmp_sentence.lower()
    tmp_sentence = tmp_sentence.strip().split()
    length = len(tmp_sentence)
    for i in range(length - n + 1):
        tmp_words = " ".join(tmp_sentence[i: i + n])
        if tmp_words:
            words[tmp_words] += 1
    return words