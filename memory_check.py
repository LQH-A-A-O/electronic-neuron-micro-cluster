# -*- coding: utf-8 -*-
"""`.memory/` 一致性体检：找过时、重复、缺字段、孤儿索引。

这是治「版本漂移」的关键 —— 借鉴 interactive-novel 的 notes-stale 检查。

    python memory_check.py            # 体检
    python memory_check.py --fix      # 自动修（只做安全的：重建索引）
"""
import json, os, sys, time
import memory_store as MS
def check():
    issues = []
    MS.ensure()
    # 1. 索引是否过期
    meta = json.load(open(MS.META, encoding='utf-8')) if os.path.exists(MS.META) else None
    if meta is None:
        issues.append(('ERROR', 'index-missing', '索引不存在，跑 rebuild'))
    else:
        if meta.get('facts_sha') != MS.sha(MS.FACTS):
            issues.append(('ERROR', 'index-stale', 'facts.md 改了但索引没重建'))
        if meta.get('episodes_sha') != MS.sha(MS.EPISODES):
            issues.append(('ERROR', 'index-stale', 'episodes.md 改了但索引没重建'))
    # 2. 同一 key 出现多次（版本冲突）
    facts = MS.parse_facts()
    by = {}
    for f in facts:
        by.setdefault(f['key'], []).append(f)
    for k, lst in by.items():
        vals = {x['value'] for x in lst}
        if len(vals) > 1:
            latest = max(lst, key=lambda x: x['at'] or '')
            issues.append(('WARN', 'fact-conflict',
                           '`%s` 有 %d 个版本 %s，最新是 %s (%s)'
                           % (k, len(vals), sorted(vals), latest['value'], latest['at'])))
        elif len(lst) > 1:
            issues.append(('WARN', 'fact-duplicate', '`%s` 重复了 %d 次' % (k, len(lst))))
    # 3. 缺 measured_at
    for f in facts:
        if not f['at']:
            issues.append(('WARN', 'fact-no-date', '`%s` 缺 measured_at' % f['key']))
    # 4. 孤儿索引
    chs, _ = MS.load()
    if chs is not None:
        src = {(c['kind'], c['text'][:60]) for c in facts + MS.parse_episodes()}
        for c in chs:
            if (c['kind'], c['text'][:60]) not in src:
                issues.append(('WARN', 'index-orphan', '索引里有真源没有的条目：%s' % c['text'][:40]))
    # 5. 空真源
    if not facts:
        issues.append(('WARN', 'facts-empty', 'facts.md 里没有解析出任何事实'))
    return issues
if __name__ == '__main__':
    iss = check()
    err = sum(1 for x in iss if x[0] == 'ERROR')
    warn = sum(1 for x in iss if x[0] == 'WARN')
    print('=== .memory 体检 ===')
    print('  真源：facts.md %d 条事实 / episodes.md %d 条片段'
          % (len(MS.parse_facts()), len(MS.parse_episodes())))
    m = json.load(open(MS.META, encoding='utf-8')) if os.path.exists(MS.META) else {}
    if m:
        print('  索引：%d 条，建于 %s' % (m.get('n', 0), m.get('built_at', '?')))
    print()
    if not iss:
        print('  ✅ 无问题')
    for lvl, code, msg in iss:
        print('  [%-5s] %-16s %s' % (lvl, code, msg))
    print()
    print('合计：%d 个错误，%d 个警告' % (err, warn))
    if '--fix' in sys.argv:
        if any(c == 'index-stale' or c == 'index-missing' for _, c, _ in iss):
            print()
            print('  修：重建索引...')
            MS.rebuild()
        else:
            print('  没有可自动修的项（版本冲突要人工决定保留哪个）')