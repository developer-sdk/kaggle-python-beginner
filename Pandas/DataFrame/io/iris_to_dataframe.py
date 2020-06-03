from sklearn import datasets

# irirs 데이터 로드 
iris = datasets.load_iris()
iris

# iris의 data는 배열, 칼럼명은 feature_names로 확인 
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# data에 종(species)의 이름이 없기 때문에 target 데이터를 이용해서 생성 
df['species'] = [iris.target_names[i] for i in iris.target]
In [000]:df
Out[249]: 
     sepal length (cm)  sepal width (cm)  ...  petal width (cm)    species
0                  5.1               3.5  ...               0.2     setosa
1                  4.9               3.0  ...               0.2     setosa
2                  4.7               3.2  ...               0.2     setosa