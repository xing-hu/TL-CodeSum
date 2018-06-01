import nltk
from nltk.translate.bleu_score import SmoothingFunction

def nltk_sentence_bleu(hypothesis, reference, order=4):
    cc = SmoothingFunction()
    return nltk.translate.bleu([reference], hypothesis, smoothing_function=cc.method4)


def nltk_corpus_bleu(hypotheses, references, order=4):
    refs = []
    count = 0
    total_score = 0.0

    cc = SmoothingFunction()

    for hyp, ref in zip(hypotheses, references):
        hyp = hyp.split()
        ref = ref.split()
        refs.append([ref])

        score = nltk.translate.bleu([ref], hyp, smoothing_function=cc.method4)
        total_score += score
        count += 1

    avg_score = total_score / count
    corpus_bleu = nltk.translate.bleu_score.corpus_bleu(refs, hypotheses)
    print('corpus_bleu: %.4f avg_score: %.4f' % (corpus_bleu, avg_score))
    return corpus_bleu, avg_score
