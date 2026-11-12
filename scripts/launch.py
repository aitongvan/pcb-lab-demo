"""Start the local app independently of the launch window and open its actual URL."""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.request
import webbrowser

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'outputs'
STATE=OUTPUT/'server.json'

def ready(url):
    try:
        # Local app should not depend on system HTTP proxy configuration.
        opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(url+'/_stcore/health',timeout=1) as response:
            return response.status==200 and response.read().strip()==b'ok'
    except (OSError, ValueError):
        return False

def available_port():
    for port in range(8501,8521):
        with socket.socket() as connection:
            try:
                connection.bind(('127.0.0.1',port))
                return port
            except OSError:
                continue
    raise RuntimeError('Khong tim thay cong trong 8501-8520.')

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--no-browser',action='store_true')
    args=parser.parse_args()
    OUTPUT.mkdir(exist_ok=True)
    if STATE.exists():
        try:
            previous=json.loads(STATE.read_text(encoding='utf-8'))
            if previous.get('project')==str(ROOT) and ready(previous['url']):
                print('Ung dung dang chay: '+previous['url'],flush=True)
                if not args.no_browser: webbrowser.open(previous['url'])
                return
        except (OSError,ValueError,KeyError):
            pass
    port=available_port()
    url=f'http://127.0.0.1:{port}'
    command=[sys.executable,'-m','streamlit','run',str(ROOT/'app.py'),
             '--server.address','127.0.0.1','--server.port',str(port),'--server.headless','true']
    environment=os.environ.copy()
    environment['PYTHONUTF8']='1'
    flags=0
    if os.name=='nt':
        flags=subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
    with (OUTPUT/'streamlit.stdout.log').open('w',encoding='utf-8') as out, (OUTPUT/'streamlit.stderr.log').open('w',encoding='utf-8') as err:
        process=subprocess.Popen(command,cwd=ROOT,env=environment,stdout=out,stderr=err,
                                 creationflags=flags,start_new_session=os.name!='nt')
    print('Dang khoi dong PCB Lab...',flush=True)
    for _ in range(90):
        if ready(url):
            STATE.write_text(json.dumps(dict(project=str(ROOT),url=url,pid=process.pid),indent=2),encoding='utf-8')
            print('Da san sang: '+url,flush=True)
            if not args.no_browser: webbrowser.open(url)
            return
        if process.poll() is not None:
            log=(OUTPUT/'streamlit.stderr.log').read_text(encoding='utf-8',errors='replace')
            raise RuntimeError('Khoi dong that bai. Chi tiet:\n'+log[-6000:])
        time.sleep(1)
    raise RuntimeError('Chua khoi dong xong sau 90 giay. Xem outputs/streamlit.stderr.log.')

if __name__=='__main__':
    try:
        main()
    except Exception as error:
        print(str(error),file=sys.stderr)
        sys.exit(1)
