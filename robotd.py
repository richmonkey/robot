from SimpleXMLRPCServer import SimpleXMLRPCServer
from SimpleXMLRPCServer import SimpleXMLRPCRequestHandler
from gensim import corpora, models, similarities
import jieba
import random
import logging
from mysql import Mysql
import config
import copy

dictionary = corpora.Dictionary.load_from_text('data/corpus_wordids.txt')
lsi = models.LsiModel.load('data/model.lsi')

raw_questions = None
index = None

def ask_question(store_id, query):
    logging.debug("store_id:%s query:%s", store_id, query)
    if not query:
        return []

    q = dictionary.doc2bow(list(jieba.cut(query, cut_all=False)))
    sims = index[lsi[q]]
    sims = sorted(enumerate(sims), key=lambda item: -item[1])
    sims = [ item for item in sims if raw_questions[item[0]]['store_id'] == store_id ]
    sims = sims[:3]
    answers = [ raw_questions[item[0]] for item in sims if item[1] > 0.95 ]
    answers = copy.deepcopy(answers)
    for a in answers:
        a.pop('store_id')

    return answers

def load_questions():
    cnf = config.MYSQL
    db = Mysql(*cnf)

    sql = "SELECT id, question, answer, store_id from question"
    r = db.execute(sql)
    result = list(r.fetchall())
    db.close()
    return result

def refresh_questions():
    global raw_questions
    global index

    logging.debug("refresh questions...")

    question_vecs = []
    raw_questions = []
    questions = load_questions()
    for q in questions:
        question = q['question']
        seg = jieba.cut(question, cut_all=False)
        m = [t for t in seg]
        if m:
            question_vecs.append(dictionary.doc2bow(m))
            raw_questions.append(q)

    mm = lsi[question_vecs]
    if not mm:
        index = similarities.MatrixSimilarity(mm, num_features=10)
    else:
        index = similarities.MatrixSimilarity(mm)

if __name__ == "__main__":
    random.seed()
    logger = logging.getLogger('')
    logger.setLevel(logging.DEBUG)
    refresh_questions();

    server = SimpleXMLRPCServer(config.RPC_ADDR, allow_none=True)
    server.register_function(ask_question)
    server.register_function(refresh_questions)
    server.serve_forever()

