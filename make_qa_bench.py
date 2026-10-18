# -*- coding: utf-8 -*-
"""构造 120 条知识问答评测集（事实 + 改写查询 + 不可答问题）。

设计原则：
  1. 答案都是独特 token（数字/名字），减少同义表述导致的误判
  2. 查询用同义词改写，字面尽量不与事实重叠
  3. 加 20 条「库里没有」的问题，用来测弃答能力
"""
import json, os, random
ROOT = os.path.dirname(os.path.abspath(__file__))
rnd = random.Random(20261018)

# 实体 x 属性 -> 值；属性带同义问法
SYS = ['订单服务','支付网关','用户中心','消息总线','搜索集群','推荐引擎',
       '风控系统','报表平台','文件存储','调度中心','配置中心','审计日志']
ATTR = [
  ('监听端口', ['在哪个端口上','用哪个口子','端口号是多少','监听哪个端口'], lambda i: str(8000+i*7)),
  ('超时毫秒数', ['等多久超时','超时设多少毫秒','多久算超时','超时时间'], lambda i: str(500+i*13)),
  ('负责团队', ['哪个团队管','谁在负责','归哪个组','负责人是哪个团队'], lambda i: ['基础架构组','交易组','数据组','平台组'][i%4]),
  ('日请求上限', ['一天能扛多少请求','日调用上限','每天最多多少','请求配额'], lambda i: str(100000*(i%9+1))),
  ('部署区域', ['放在哪个region','部署在哪个地方','机房在哪','跑在哪个区域'], lambda i: ['华东1','华北2','华南1','西南3'][i%4]),
]
facts, queries = [], []
fid = 0
for si, s in enumerate(SYS):
    for ai, (attr, syns, mk) in enumerate(ATTR):
        v = mk(si*len(ATTR)+ai)
        key = '%s 的%s是 %s' % (s, attr, v)
        facts.append(dict(key=key, value=v, sys=s, attr=attr))
        j = (si*len(ATTR)+ai) % len(syns)
        # 改写查询：只用同义问法，不提「属性」原名
        qs = ['%s %s？' % (s, syns[j]),
              '%s 这块，%s？' % (s, syns[(j+1)%len(syns)]),
              '想确认一下，%s %s' % (s, syns[(j+2)%len(syns)])]
        queries.append(dict(q=qs[0], gold=v, key=key, answerable=1, form=0))
        if (si+ai) % 2 == 0:
            queries.append(dict(q=qs[1], gold=v, key=key, answerable=1, form=1))
        if (si+ai) % 3 == 0:
            queries.append(dict(q=qs[2], gold=v, key=key, answerable=1, form=2))
# 不可答问题：库里没有的实体
for i in range(20):
    q = '临时服务X%02d 的%s是 %s？' % (i, ATTR[i%len(ATTR)][0], '多少')
    queries.append(dict(q=q, gold=None, key=None, answerable=0, form=3))
rnd.shuffle(queries)
out = os.path.join(ROOT, 'data', 'qa_bench.json')
json.dump(dict(facts=facts, queries=queries), open(out, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('事实 %d 条，查询 %d 条（可答 %d / 不可答 %d）'
      % (len(facts), len(queries), sum(1 for x in queries if x['answerable']),
         sum(1 for x in queries if not x['answerable'])))
print('  查询形式分布：', {f: sum(1 for x in queries if x['form']==f) for f in range(4)})