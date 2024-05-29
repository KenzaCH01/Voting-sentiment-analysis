# Voting method to extract Valence Polarity

### Import Voting (Python 3.8 or below)
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
model.fit(base['commentaire'], base['note_binary'])
```

|                    | Logistic Regression   | Linear Discriminant Analysis   | SVM      | PLS Regression   | Ridge Classifier   | Final Vote   |
|:-------------------|:----------------------|:-------------------------------|:---------|:-----------------|:-------------------|:-------------|
| 11                 | Positive              | Positive                       | Positive | Positive         | Positive           | Positive     |
| 12                 | Negative              | Negative                       | Negative | Negative         | Negative           | Negative     |
| 13                 | Negative              | Negative                       | Negative | Negative         | Negative           | Negative     |
| abandonne          | Negative              | Negative                       | Negative | Negative         | Negative           | Negative     |
| abonnement         | Positive              | Negative                       | Positive | Positive         | Positive           | Positive     |
| abonnement dans    | Positive              | Negative                       | Positive | Positive         | Positive           | Positive     |
| abonnement netflix | Positive              | Positive                       | Positive | Positive         | Positive           | Positive     |


### Top features and their sentiment polarity
```python
model.predict()
```


