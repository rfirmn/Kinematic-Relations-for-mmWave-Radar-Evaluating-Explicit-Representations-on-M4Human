"""Save public pinned source text for audit; never execute downloaded code."""
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import urllib.request

COMMIT='e01b8fed6132306c3553ed6680be912c21594dce'


def main():
    root=Path('docs/reference/upstream')
    tree=json.loads((root/'tree_e01b8fed.json').read_text())
    entries=[row['path'] for row in tree['tree'] if row['type']=='blob' and row['path'].endswith(('.py','.yaml','.md','.yml'))]
    def fetch(name):
        data=urllib.request.urlopen(f'https://raw.githubusercontent.com/FanJunqiao/M4Human/{COMMIT}/{name}',timeout=30).read()
        out=root/'pinned'/name
        out.parent.mkdir(parents=True,exist_ok=True)
        out.write_bytes(data)
        return name
    with ThreadPoolExecutor(4) as pool:
        print('downloaded',len(list(pool.map(fetch,entries))))
    request=urllib.request.Request('https://api.github.com/repos/FanJunqiao/M4Human/commits/main',headers={'User-Agent':'relational64-audit'})
    current=json.load(urllib.request.urlopen(request,timeout=30))
    (root/'current_commit.json').write_text(json.dumps({'sha':current['sha'],'date':current['commit']['committer']['date'],'message':current['commit']['message']},indent=2))
    print('current_main',current['sha'])


if __name__=='__main__':
    main()
