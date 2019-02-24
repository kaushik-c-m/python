'''
Created on 25-Apr-2018

@author: kaushik
'''
import entityRecognition
#print(entityRecognition.train("Sample_train.json"))
print(entityRecognition.predict(u'Can I make a payment from credit account 456 today'))
print(entityRecognition.predict(u'Can I make a payment from debit account 2345568 with preformat codePreformat1CITI234 today ?'))