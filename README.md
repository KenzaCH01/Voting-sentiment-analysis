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
base
```

### Train the Voting model with the Data and feature sentiment prediction
```python
model = Voting()
model.fit(base['commentaire'], base['note_binary'])
```

### Top features and their sentiment polarity
```python
model.predict()
```


