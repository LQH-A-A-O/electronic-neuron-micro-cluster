# -*- coding: utf-8 -*-
"""真实文档知识库：把项目自己的 87 份 md 切片建索引。"""
import glob, json, os, re, sys, time
import numpy as np, requests
ROOT = os.path.dirname(os.path.abspath(__file__))
EMB = 'http://127.0.0.1:8044'
OUT = os.path.join(ROOT, 'data', 'doc_index.json')
MAXCH = 700   # 每片最大字符数
def chunks():
    out = []
    for p in sorted(glob.glob(os.path.join(ROOT, '*.md'))):
        name = os.path.basename(p)
        try:
            txt = open(p, encoding='utf-8').read()
        except Exception:
            continue
        # 按 ## 小节切
        parts = re.split(r'^## ', txt, flags=re.M)
        for k, part in enumerate(parts):
            part = part.strip()
            if len(part) < 60:
                continue
            head = part.split(chr(10))[0][:60]
            body = part
            for i in range(0, len(body), MAXCH):
                seg = body[i:i+MAXCH]
                if len(seg) < 60: continue
                out.append(dict(doc=name, sec=head, text=seg))
    return out
def embed_all(chs):
    """逐条编码，失败重试 3 次；仍失败就跳过并记录。"""
    v, keep = [], []
    fail = 0
    for i, c in enumerate(chs):
        ok = False
        for attempt in range(3):
            try:
                r = requests.post(EMB + '/v1/embeddings', timeout=90,
                                  json={'input': c['text'][:1200], 'model': 'x'})
                d = r.json()
                if 'data' in d:
                    v.append(d['data'][0]['embedding'])
                    keep.append(c)
                    ok = True
                    break
            except Exception:
                pass
            time.sleep(0.5)
        if not ok:
            fail += 1
            if fail <= 3:
                print('  跳过切片 %d（%s / %s）' % (i, c['doc'], c['sec'][:20]))
    print('编码完成：成功 %d，失败跳过 %d' % (len(v), fail))
    a = np.array(v, dtype=np.float32)
    n = np.linalg.norm(a, axis=1, keepdims=True); n[n == 0] = 1
    return a / n, keep
if __name__ == '__main__':
    chs = chunks()
    print('切片 %d 个（来自 %d 份文档）' % (len(chs), len(set(c['doc'] for c in chs))))
    sys.stdout.flush()
    V, chs = embed_all(chs)
    np.save(os.path.join(ROOT, 'data', 'doc_vecs.npy'), V)
    json.dump(chs, open(OUT, 'w', encoding='utf-8'), ensure_ascii=False)
    print('索引已存：data/doc_index.json + data/doc_vecs.npy，维度 %d' % V.shape[1])