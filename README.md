# personal_helper_functions
These are a bunch of helper functions I personally use and built (and will continue building) over time. Feel free to use them if they help you

---

# Files:
## training_functions.py:
This file contains my train-test loop implementation for training PyTorch models

---

# How to use them in your own code/environment:
Just run this script in your .ipynb/.py file:
```python
from pathlib import Path
import requests

if Path("some_file.py").is_file():
    print('"some_file.py" Already Exists')
else:
    print("Downloading some_file.py...")
    request = requests.get("https://raw.githubusercontent.com/MarwanMohammed2500/personal_helper_functions/refs/heads/main/some_file.py") # Change some_file.py to the actual file name

    with open("some_file.py", "wb") as f:
        f.write(request.content)
        print("Done!")
```

---

# Required Dependencies:
So far, you just need to install PyTorch, I'll update the list in the future if needed (Possibly will add matplotlib and seaborn alongside Pandas soon)
