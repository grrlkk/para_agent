"""Human-facing sheets contain only blinded text and empty answer columns."""
import csv
import json
from pathlib import Path

FIELDS=['item_id','prompt','before','after','context_before','context_after','human_label','human_reason']


def write_human_sheet(directory,items):
    directory.mkdir(parents=True,exist_ok=False)
    rows=[{'item_id':x['blind_id'],**{k:x[k] for k in FIELDS[1:6]},'human_label':'','human_reason':''} for x in items]
    with (directory/'semantic_labels.csv').open('x',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=FIELDS);writer.writeheader();writer.writerows(rows)
    # No private source map or judgments are embedded in this self-contained UI.
    encoded=json.dumps(rows,ensure_ascii=False).replace('<','\\u003c')
    page='''<!doctype html><html lang="ko"><meta charset="utf-8"><title>의미 보존 라벨링</title>
<style>body{font:16px sans-serif;max-width:1100px;margin:32px auto;padding:0 20px;line-height:1.65}article{border:1px solid #ccc;padding:20px;margin:24px 0}.pair{display:grid;grid-template-columns:1fr 1fr;gap:24px}pre{white-space:pre-wrap;font:inherit}select,textarea,button{font:inherit;padding:8px}textarea{width:95%}header{position:sticky;top:0;background:white;padding:12px;border-bottom:1px solid #ccc}details{margin:12px 0}.muted{color:#555}</style>
<header><strong>실제 수정 문장 의미 보존 판정</strong> <button id="export">CSV 저장</button> <span id="count"></span></header>
<p>원문과 수정문의 뜻을 비교하세요. 문체의 세련됨 대신 사실·주장·감정 강도·가능성·시제·부정·수량을 확인하세요. 자세한 문맥은 펼쳐 볼 수 있습니다. 선택 내용은 브라우저를 닫으면 사라지므로 CSV를 저장하세요.</p><main id="items"></main>
<script>const rows=__ROWS__;const host=document.getElementById('items');
function el(tag,text){let x=document.createElement(tag);if(text!==undefined)x.textContent=text;return x;}
function update(){document.getElementById('count').textContent=rows.filter(r=>r.human_label).length+' / '+rows.length;}
rows.forEach(r=>{let a=el('article');a.append(el('h2',r.item_id),el('p','문항: '+r.prompt));let pair=el('div');pair.className='pair';[['원문','before'],['수정문','after']].forEach(([name,key])=>{let d=el('div');d.append(el('strong',name),el('pre',r[key]||'(문장 없음)'));pair.append(d)});a.append(pair);let det=el('details');det.append(el('summary','문맥 보기'),el('h3','원문 문맥'),el('pre',r.context_before),el('h3','수정문 문맥'),el('pre',r.context_after));a.append(det);let s=el('select');['','보존','경미한 뉘앙스 변화','의미 변화'].forEach(v=>{let o=el('option',v||'판정 선택');o.value=v;s.append(o)});s.onchange=()=>{r.human_label=s.value;update()};let reason=el('textarea');reason.placeholder='판정 이유 (짧게)';reason.oninput=()=>r.human_reason=reason.value;a.append(s,el('p'),reason);host.append(a)});update();
document.getElementById('export').onclick=()=>{let fields=Object.keys(rows[0]);let cell=v=>'"'+String(v).replaceAll('"','""')+'"';let csv='\\ufeff'+[fields.map(cell).join(','),...rows.map(r=>fields.map(k=>cell(r[k])).join(','))].join('\\r\\n');let url=URL.createObjectURL(new Blob([csv],{type:'text/csv;charset=utf-8'}));let a=el('a');a.href=url;a.download='semantic_labels.csv';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)};
</script></html>'''
    (directory/'semantic_labels.html').write_text(page.replace('__ROWS__',encoded),encoding='utf-8')
    return rows


def read_human_labels(path,expected_ids):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    ids=[r['item_id'] for r in rows]
    if len(ids)!=len(set(ids)) or set(ids)!=set(expected_ids):raise ValueError('Label IDs must match the blinded sheet exactly')
    valid={'보존','경미한 뉘앙스 변화','의미 변화'}
    if any(r['human_label'] not in valid for r in rows):raise ValueError('All human labels must be filled with one of the three categories')
    return {r['item_id']:r['human_label'] for r in rows}
