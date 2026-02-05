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


from audio_tabular_predict import predict

print (predict(
    [229.88836845905988,2.071104025335901,7.955661570386569,
    0.0184571699608276,8.027501637281006e-05,0.0100034607734826,
    0.0126182138077345,0.0300103823204479,0.1759911068083561,1.553122422015872,
    0.0901466143338587,0.1143208490382301,0.148635300610465,0.2704398430015762,
    737.3026395212191,1315.6665052161916,2782.494839520613,3648.298588629576,
    29.29205741412645,58.73484719784517,59.23159248987346,85.12483660775213,
    0,100,600
]
))