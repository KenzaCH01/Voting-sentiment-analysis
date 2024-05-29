# Voting method to extract Valence Polarity

* usage: Python 3.8.8 or below
* pandas, numpy, sklearn
  
### Import Voting 
```python
from vote import Voting
```

### Import data and determine each review's sentiment polarity 
```python
import pandas as pd 

base = pd.read_csv('netflix.csv', sep = ";")
base['note_binary'] = base['note'].apply(lambda x: 1 if x > 3 else 0)
```

### Train the Voting model with the comments and the binary scores
```python
model = Voting()
base = model.fit(base['Comments'], base['note_binary'])
```

|                      | Logistic Regression   | Linear Discriminant Analysis   | SVM      | PLS Regression   | Ridge Classifier   | Final Vote   |
|:---------------------|:----------------------|:-------------------------------|:---------|:-----------------|:-------------------|:-------------|
| 11                   | Negative              | Negative                       | Negative | Negative         | Negative           | Negative     |
| 11 euros             | Positive              | Positive                       | Negative | Negative         | Negative           | Negative     |
| 12                   | Negative              | Negative                       | Negative | Positive         | Negative           | Negative     |
| 13                   | Negative              | Negative                       | Negative | Negative         | Negative           | Negative     |
| abandoning           | Negative              | Negative                       | Negative | Negative         | Negative           | Negative     |
| about                | Positive              | Positive                       | Positive | Negative         | Positive           | Positive     |


### Top features and their sentiment polarity
```python
model.predict()
```
|    | Feature      | Final Vote   |
|---:|:-------------|:-------------|
|  0 | en           | Positive     |
|  1 | films en     | Positive     |
|  2 | fait le      | Negative     |
|  3 | excellent    | Positive     |
|  4 | en france    | Positive     |
|  5 | et mon       | Positive     |
|  6 | faire        | Negative     |
|  7 | en français  | Negative     |


