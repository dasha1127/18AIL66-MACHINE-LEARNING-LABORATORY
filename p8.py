import pandas as pd
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination

data = pd.read_csv("heart_disease.csv")
heart_disease = pd.DataFrame(data)

model = BayesianNetwork([
    ('age', 'Lifestyle'),
    ('Gender', 'Lifestyle'),
    ('Family', 'heartdisease'),
    ('diet', 'cholestrol'),
    ('Lifestyle', 'cholestrol'),
    ('cholestrol', 'heartdisease'),
])

model.fit(heart_disease, estimator=MaximumLikelihoodEstimator)

HeartDisease_infer = VariableElimination(model)

print("Enter age: 0 - super senior citizens, 1 - senior citizens, 2 - middle-aged, 3 - youth, 4 - Teen")
print("Enter gender: 0 - Male, 1 - Female")
print("Enter Family history: 0 - No, 1 - Yes")
print("Enter diet: 0 - High, 1 - Medium")
print("Enter Lifestyle: 0 - Athlete, 1 - Active, 2 - Moderate, 3 - Sedentary")
print("Enter cholestrol: 0 - High, 1 - Borderline, 2 - Normal")
q = HeartDisease_infer.query(variables=['heartdisease'], evidence={
    'age': int(input("Enter age: ")),
    'Gender': int(input("Enter gender: ")),
    'Family': int(input("Enter Family history: ")),
    'diet': int(input("Enter diet: ")),
    'Lifestyle': int(input("Enter Lifestyle: ")),
    'cholestrol': int(input("Enter cholestrol: "))
})

print(q.values)  # Retrieve the values of the 'heartdisease' factor
