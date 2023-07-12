import pandas as pd

def find_s_algorithm(training_data):
    hypothesis=training_data.iloc[0,:-1].tolist()

    for i in range(1,len(training_data)):
        instance=training_data.iloc[i,:-1].tolist()
        label=training_data.iloc[i,-1]
        if label=="Yes":
            for j in range(len(hypothesis)):
                if instance[j]!=hypothesis[j]:
                    hypothesis[j]='?'
    return hypothesis
def list_then_eliminate_algorithm(training_data):
    hypothesis_space=[]
    for i in range(len(training_data)):
        instance=training_data.iloc[i,:-1].tolist()
        label=training_data.iloc[i,-1]
        if label=='Yes':
            hypothesis_space=[h for h in hypothesis_space if all(h[j]==instance[j] or h[j]=='?' for j in range (len(h)))]
            hypothesis_space.append(instance)
    hypothesis=['?' for _ in range(len(training_data.columns)-1)]
    for j in range(len(hypothesis)):
         values=set([h[j] for h in hypothesis_space])
         if len(values)==1:
             hypothesis[j]=values.pop()
    return hypothesis
training_data=pd.read_csv('Book1.csv')
find_s_hypothesis= find_s_algorithm(training_data)
print("find-s:",find_s_hypothesis)
lte_hypothesis=list_then_eliminate_algorithm(training_data)
print("lte:",lte_hypothesis)



