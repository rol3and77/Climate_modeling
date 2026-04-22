# Climate Modeling

This is a climate model website created for an undergraduate research project.

This project provides a website that can be executed in the Google Colab environment.

The execution method is as follows.  
A total of four cells are required.

---

## 1. First Cell

Install required libraries and tools:

```
!pip install -q streamlit numpy matplotlib scipy pandas numba
!wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
!dpkg -i cloudflared-linux-amd64.deb
```

---

## 2. Second Cell

Insert the `requirements.txt` file in this cell.

```
%%writefile app.py
# Paste the full Streamlit application code here
```

---

## 3. Third Cell

Run the Streamlit server:

```
import subprocess
import threading

def run_streamlit():
    subprocess.run([
        "streamlit", "run", "app.py",
        "--server.port", "8501"
    ])

thread = threading.Thread(target=run_streamlit, daemon=True)
thread.start()
```

---

## 4. Fourth Cell

Expose the application using Cloudflare tunnel:

```
!cloudflared tunnel --url http://localhost:8501
```

---

## Execution Instructions

Run the cells sequentially from the first to the fourth.

After running the fourth cell, you will see a message:

"Your quick Tunnel has been created! Visit it at (it may take some time to be reachable):"

Click the generated link to access the website.

---

## Notes

- If the website does not run properly, execute all cells again from the beginning in order.
- Alternatively, rerun the fourth cell, wait for a moment, and then access the link again.
