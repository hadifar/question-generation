

from pipelines import pipeline

nlp = pipeline('multitask-qa-qg')
text = 'Couples who cohabitate before marriage are far more likely than couples who did not cohabitate before marriage to be married at least 10 years.'
questions = nlp(text)
print(questions)