# personal_helper_functions
These are a bunch of helper functions I personally use and built (and will continue building) over time. Feel free to use them if they help you

---

# Files:
### training_functions.py:
This file contains my train-test loop implementations for training PyTorch models

### plotting_functions.py:
This contains any helpful and common plottings, functionalized.

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
- PyTorc
- TQDM
# Collaboration:
Feel free to reach out or fork this repo. Collaborations are always welcome!
