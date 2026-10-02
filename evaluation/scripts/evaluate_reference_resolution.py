from pathlib import Path
import json, re, unicodedata
from difflib import SequenceMatcher

HOME = Path.home()
BENCH = HOME / 'tabulusbench'
PAPER_BASE = BENCH / 'Energy_Materials_Chemical_Sciences' / 'atomic_layer_deposition'
RUN_BASE = BENCH / 'runs' / 'stage6' / 'controlled-p251-p252-final'
PAPERS = ('P251','P252')


def plain(s):
    s = str(s or '').replace('–','-').replace('—','-').replace('−','-')
    s = re.sub(r'''\\["'`^~=.uvHckbdtr]\s*\{?\s*([A-Za-z])\s*\}?''', r'\1', s)
    s = re.sub(r'\\(?:textit|textbf|emph|mathrm|mathbf|rm|it|em|url|mbox)\s*\{([^{}]*)\}', r'\1', s)
    s = re.sub(r'\\[A-Za-z@]+\*?(?:\[[^\]]*\])?', ' ', s)
    s = s.replace('{',' ').replace('}',' ').replace('~',' ')
    s = unicodedata.normalize('NFKD', s)
    return ''.join(c for c in s if not unicodedata.combining(c))


def norm_text(s):
    return ' '.join(re.sub(r'[^a-z0-9]+',' ',plain(s).casefold()).split())


def norm_doi(s):
    s = str(s or '').strip().casefold()
    s = re.sub(r'^https?://(?:dx\.)?doi\.org/', '', s)
    s = re.sub(r'^doi\s*:\s*', '', s)
    return s.strip().rstrip('.,;')


def title_match(a,b):
    a,b = norm_text(a), norm_text(b)
    return bool(a and b) and (a == b or SequenceMatcher(None,a,b,autojunk=False).ratio() >= 0.95)


def venue_tokens(s):
    stop={'of','the','and','for','in','on','a','an'}
    return [x for x in norm_text(s).split() if x not in stop]


def compat(a,b):
    if a == b: return True
    if len(a)==1: return b.startswith(a)
    if len(b)==1: return a.startswith(b)
    k=min(len(a),len(b),4)
    return k>=2 and a[:k]==b[:k]


def venue_match(a,b):
    A,B=venue_tokens(a),venue_tokens(b)
    if not A or not B: return False
    if A==B: return True
    short,long=(A,B) if len(A)<=len(B) else (B,A)
    return sum(any(compat(x,y) for y in long) for x in short)/len(short) >= 0.80


def gold_surnames(author_field):
    out=[]
    for person in re.split(r'\s+and\s+', plain(author_field), flags=re.I):
        person=person.strip()
        if not person or person.casefold()=='others': continue
        sur=person.split(',',1)[0] if ',' in person else (person.split()[-1] if person.split() else '')
        n=norm_text(sur).replace(' ','')
        if n: out.append(n)
    return out


def pred_surnames(authors):
    if not isinstance(authors,list): return []
    out=[]
    for person in authors:
        p=plain(person).strip()
        if not p: continue
        sur=p.split(',',1)[0] if ',' in p else (p.split()[-1] if p.split() else '')
        n=norm_text(sur).replace(' ','')
        if n: out.append(n)
    return out


def authors_match(g,p): return bool(gold_surnames(g)) and gold_surnames(g)==pred_surnames(p)


def load_prediction(path):
    d=json.loads(path.read_text()); reg={}
    for e in d.get('entries',[]):
        r=e.get('resolution',{}) if isinstance(e,dict) else {}
        try: idx=int(r.get('reference_index'))
        except (TypeError,ValueError): continue
        if idx in reg: raise RuntimeError(f'Duplicate prediction reference_index {idx}')
        reg[idx]=r
    return d,reg


def pct(a,b): return 100.0*a/b if b else None

def harmonic(p,r): return 2*p*r/(p+r) if p is not None and r is not None and p+r else 0.0


def evaluate_paper(paper):
    G=json.loads((PAPER_BASE/paper/'reference_resolution'/'gold.json').read_text())
    Ptop,P=load_prediction(RUN_BASE/paper/'run_01'/'references'/'reference_resolution.json')
    rows=G['records']

    # The special controlled Step 6 inputs preserve the benchmark gold index as
    # reference_index. step4_index is provenance from the Step 4 one-to-one
    # alignment and is not the identifier serialized by these controlled runs.
    expected={int(r['gold_index']) for r in rows}
    missing=sorted(expected-set(P)); extra=sorted(set(P)-expected)
    if missing or extra:
        raise RuntimeError(f'{paper}: prediction index mismatch missing={missing[:10]} extra={extra[:10]}')

    s={'paper':paper,'refs':len(rows),'resolved':0,'rejected':0,
       'gold_doi':0,'doi_pred_on_gold':0,'doi_exact':0,
       'title_n':0,'title_ok':0,'authors_n':0,'authors_ok':0,
       'year_n':0,'year_ok':0,'venue_n':0,'venue_ok':0,
       'llm_assisted':int(Ptop.get('llm_adjudicated_count') or 0),
       'retry':int(Ptop.get('retry_count') or 0)}

    for g in rows:
        p=P[int(g['gold_index'])]
        accepted=str(p.get('status') or '') in {'validated_with_doi','validated_without_doi'}
        s['resolved'] += int(accepted); s['rejected'] += int(not accepted)

        gd,pd=norm_doi(g.get('gold_doi')),norm_doi(p.get('canonical_doi'))
        # DOI evaluation is conditional on source-available DOI annotations.
        # Absence of a DOI in historical BibTeX is not evidence that the work has
        # no DOI, so predictions on source-DOI-missing records are not false positives.
        if gd:
            s['gold_doi'] += 1
            if pd: s['doi_pred_on_gold'] += 1
            if pd == gd: s['doi_exact'] += 1

        gt=g.get('gold_title') or ''
        if gt:
            s['title_n'] += 1; s['title_ok'] += int(title_match(gt,p.get('canonical_title') or ''))
        ga=g.get('gold_authors') or ''
        if ga:
            s['authors_n'] += 1; s['authors_ok'] += int(authors_match(ga,p.get('canonical_authors') or []))
        gy=str(g.get('gold_year') or '').strip()
        if gy:
            s['year_n'] += 1; s['year_ok'] += int(str(p.get('canonical_year') or '').strip()==gy)
        gv=g.get('gold_venue') or ''
        if gv:
            s['venue_n'] += 1; s['venue_ok'] += int(venue_match(gv,p.get('canonical_venue') or ''))
    return s


def finish(s):
    doi_p=pct(s['doi_exact'],s['doi_pred_on_gold'])
    doi_r=pct(s['doi_exact'],s['gold_doi'])
    return {**s,
        'resolution_yield':pct(s['resolved'],s['refs']),
        'source_doi_rate':pct(s['gold_doi'],s['refs']),
        'doi_exact_acc':pct(s['doi_exact'],s['gold_doi']),
        'doi_precision':doi_p,'doi_recall':doi_r,'doi_f1':harmonic(doi_p or 0,doi_r or 0),
        'title_acc':pct(s['title_ok'],s['title_n']),
        'authors_acc':pct(s['authors_ok'],s['authors_n']),
        'year_acc':pct(s['year_ok'],s['year_n']),
        'venue_acc':pct(s['venue_ok'],s['venue_n']),
        'llm_rate':pct(s['llm_assisted'],s['refs']),
        'retry_rate':pct(s['retry'],s['refs'])}

raw=[evaluate_paper(p) for p in PAPERS]
pooled={k:0 for k in raw[0] if k!='paper'}
for s in raw:
    for k,v in s.items():
        if k!='paper': pooled[k]+=v
pooled['paper']='Overall'
rows=[finish(s) for s in raw]+[finish(pooled)]

print('\nSTEP 6 CONTROLLED RUN 01 — SOURCE-DERIVED GOLD')
print('='*156)
print(f"{'Scope':<9} {'Refs':>5} {'Yield':>8} {'SrcDOI':>12} {'DOI Exact':>10} {'DOI-P':>8} {'DOI-R':>8} {'DOI-F1':>8} {'Title':>8} {'Authors':>8} {'Year':>8} {'Venue':>8} {'LLM':>8} {'Retry':>8}")
print('-'*156)
for x in rows:
    F=lambda v:'n/a' if v is None else f'{v:.2f}'
    print(f"{x['paper']:<9} {x['refs']:>5} {F(x['resolution_yield']):>8} {x['gold_doi']:>5}/{x['refs']:<5} {F(x['doi_exact_acc']):>10} {F(x['doi_precision']):>8} {F(x['doi_recall']):>8} {F(x['doi_f1']):>8} {F(x['title_acc']):>8} {F(x['authors_acc']):>8} {F(x['year_acc']):>8} {F(x['venue_acc']):>8} {F(x['llm_rate']):>8} {F(x['retry_rate']):>8}")

print('\nDENOMINATORS / COUNTS')
for x in rows:
    print(f"{x['paper']}: DOI exact {x['doi_exact']}/{x['gold_doi']} source-DOI-bearing; "
          f"DOI predicted on source-DOI-bearing {x['doi_pred_on_gold']}; "
          f"title {x['title_ok']}/{x['title_n']}; authors {x['authors_ok']}/{x['authors_n']}; "
          f"year {x['year_ok']}/{x['year_n']}; venue {x['venue_ok']}/{x['venue_n']}; "
          f"resolved {x['resolved']}/{x['refs']}; LLM {x['llm_assisted']}; retry {x['retry']}")

print('\nNOTE: DOI metrics are conditional on references whose authoritative source BibTeX explicitly contains a DOI.')
print('A missing DOI field in the historical source is treated as unknown, not as evidence that the work has no DOI.')
