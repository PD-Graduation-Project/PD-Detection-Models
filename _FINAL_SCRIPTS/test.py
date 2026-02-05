from metadata_predict import predict


age= 50
height= 150
weight= 100
gender= 'male' # 'male', 'female' / 0, 1
appearance_in_kinship= -1 # 0, 1 / 'True', 'False'
appearance_in_first_grade_kinship= -1 # 0, 1 / 'True', 'False'
questions = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]

print(predict([
    age,
    height,
    weight,
    gender,
    appearance_in_kinship,
    appearance_in_first_grade_kinship,
    questions  # list of length 28
])
)