## Environment

Python version: 3.11
Install:
py -3.11 -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt

## Remarks
Trigno implementation only works in windows
data_fusion_manager script with listen_for_terminal_input func only works in windows given command: msvcrt.kbhit()

## In .venv\Lib\site-packages\GPyOpt\core\evaluators\batch_local_penalization.py
Change 
minusL = res.fun[0][0] to 
minusL = res.fun if isinstance(res.fun, float) else res.fun[0][0]