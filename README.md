# Climate_modeling
This is a climate model website created for an undergraduate research project.
I created a website that can be run in the Google Colab environment.
The execution method is as follows.
A total of four cells are required for execution.

---# 1. first cell #--------------------------------------------------------------------------------------
!pip install -q streamlit numpy matplotlib scipy pandas numba
!wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
!dpkg -i cloudflared-linux-amd64.deb
----------------------------------------------------------------------------------------------------------

---# 2. second cell #-------------------------------------------------------------------------------------
Put the app.py file in the second cell.
----------------------------------------------------------------------------------------------------------

---# 3. third cell #--------------------------------------------------------------------------------------
import subprocess
import threading
def run_streamlit():
    subprocess.run(["streamlit", "run", "app.py", "--server.port", "8501"])
thread = threading.Thread(target=run_streamlit, daemon=True)
thread.start()
----------------------------------------------------------------------------------------------------------

---# 4. fourth cell #-------------------------------------------------------------------------------------
!cloudflared tunnel --url http://localhost:8501
----------------------------------------------------------------------------------------------------------

Run the cells sequentially from the first to the fourth.
Click the link below to access the website. 
"Your quick Tunnel has been created! Visit it at (it may take some time to be reachable):"
If the website does not run, try executing the cells again from the beginning in order, or run the fourth cell, wait for a moment, and then click the link.
