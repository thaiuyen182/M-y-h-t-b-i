#!/usr/bin/env python3
import argparse, json, os, re, sys, time, unicodedata, html, hashlib, platform, importlib.metadata as md
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SEED=42
PHONE_RE=re.compile(r'(?<!\d)(?:\+?84|0)(?:[ .-]?\d){8,10}(?!\d)')
EMAIL_RE=re.compile(r'\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b',re.I)
HANDLE_RE=re.compile(r'(?<!\w)@[A-Za-z0-9_.-]{2,}\b')
URL_RE=re.compile(r'https?://\S+|www\.\S+',re.I)
ZW_RE=re.compile(r'[\u200b-\u200f\u202a-\u202e\ufeff]')
TAG_RE=re.compile(r'<[^>]+>')
SPACE_RE=re.compile(r'\s+')

COMM_PATTERNS=[
    (re.compile(r'\b(inbox|ib|zalo|lh|liên hệ|nhắn tin|dm)\b',re.I),'contact CTA'),
    (re.compile(r'\b(nhận sửa|sửa chữa|sửa robot|bảo hành|báo giá|chốt đơn|đặt hàng|ship|freeship|giảm giá|khuyến mãi|sale)\b',re.I),'sales/service CTA'),
    (PHONE_RE,'phone'),(EMAIL_RE,'email')
]

def normalize(s):
    s=html.unescape('' if pd.isna(s) else str(s)); s=unicodedata.normalize('NFC',s); s=ZW_RE.sub('',s)
    s=URL_RE.sub(' URL ',s); s=EMAIL_RE.sub(' EMAIL ',s); s=PHONE_RE.sub(' PHONE ',s); s=HANDLE_RE.sub(' HANDLE ',s)
    s=TAG_RE.sub(' ',s); s=SPACE_RE.sub(' ',s).strip(); return s

def mask(s):
    s=URL_RE.sub('[URL]',s); s=EMAIL_RE.sub('[EMAIL]',s); s=PHONE_RE.sub('[PHONE]',s); s=HANDLE_RE.sub('[HANDLE]',s); return s

def commercial_label(s):
    hits=[]
    for p,r in COMM_PATTERNS:
        if p.search(s): hits.append(r)
    issue=bool(re.search(r'\b(lỗi|hỏng|không|chưa|kêu|ồn|kẹt|mất|không chạy|không hút|không sạc|pin|cảm biến|sensor|app|wifi|lau|hút|đụng|né|bánh|chổi|dock|robot)\b',s,re.I))
    if len(hits)>=2: return ('mixed_commercial' if issue else 'commercial','; '.join(hits))
    if hits: return ('mixed_commercial' if issue else 'commercial','; '.join(hits))
    return ('organic','')

def pkgver(name):
    try:return md.version(name)
    except:return 'unavailable'

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input',required=True); ap.add_argument('--sheet',default='comments'); ap.add_argument('--output',default='outputs'); ap.add_argument('--seed',type=int,default=42); args=ap.parse_args()
    np.random.seed(args.seed); out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    t0=time.time(); warnings=[]
    xls=pd.ExcelFile(args.input)
    assert args.sheet in xls.sheet_names, f'Missing sheet {args.sheet}; found {xls.sheet_names}'
    df=pd.read_excel(args.input,sheet_name=args.sheet)
    assert 'text' in df.columns, f'Missing text column; columns={list(df.columns)}'
    df=df.copy(); df.insert(0,'source_row_id',np.arange(2,len(df)+2)); df['text_raw']=df['text']
    n0=len(df); null=int(df['text'].isna().sum()); empty=int(df['text'].fillna('').astype(str).str.strip().eq('').sum())
    df['text_norm']=df['text'].map(normalize); df['text_for_sentiment']=df['text_norm']; df['text_for_topic']=df['text_norm']
    df['commercial_label'],df['commercial_reason']=zip(*df['text_norm'].map(commercial_label))
    df['is_duplicate']=df['text_norm'].duplicated(keep='first')
    df['cleaning_status']=np.where(df['text_norm'].str.strip().eq(''),'removed_empty','retained')
    dup=int(df['is_duplicate'].sum()); empty_norm=int(df['text_norm'].str.strip().eq('').sum())
    # Primary: organic + issue-bearing mixed; exclude pure commercial and exact duplicates.
    primary=(~df['is_duplicate']) & df['text_norm'].str.strip().ne('') & df['commercial_label'].isin(['organic','mixed_commercial'])
    organic=(~df['is_duplicate']) & df['text_norm'].str.strip().ne('') & df['commercial_label'].eq('organic')
    df['is_retained_primary']=primary; df['is_retained_organic']=organic
    audit=df[['source_row_id','text_raw','text_norm','commercial_label','commercial_reason','is_duplicate','cleaning_status','is_retained_primary','is_retained_organic']].copy(); audit['text_raw']=audit['text_raw'].map(mask); audit['text_norm']=audit['text_norm'].map(mask)
    audit.to_csv(out/'cleaning_audit.csv',index=False,encoding='utf-8-sig')
    review=df[df['commercial_label'].isin(['commercial','mixed_commercial'])].copy()
    if len(review): review=review.sample(min(100,len(review)),random_state=args.seed,stratify=None)
    else: review=df.sample(min(100,len(df)),random_state=args.seed)
    review[['source_row_id','text_norm','commercial_label','commercial_reason']].assign(text=review['text_norm'].map(mask)).drop(columns='text_norm').to_csv(out/'commercial_review_sample.csv',index=False,encoding='utf-8-sig')

    work=df[primary].copy().reset_index(drop=True)
    if len(work)==0: raise RuntimeError('No retained comments after cleaning/commercial filtering.')

    # Sentiment: verify labels/config at runtime; no assumed class ordering.
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    model_id='wonrax/phobert-base-vietnamese-sentiment'
    tok=AutoTokenizer.from_pretrained(model_id); model=AutoModelForSequenceClassification.from_pretrained(model_id); model.eval()
    id2label={int(k):v for k,v in model.config.id2label.items()} if isinstance(model.config.id2label,dict) else {}
    labels={i:str(v).lower() for i,v in id2label.items()}
    def canon(v):
        v=v.lower();
        if 'pos' in v or v in ('1','positive'): return 'positive'
        if 'neg' in v or v in ('0','negative'): return 'negative'
        if 'neu' in v or 'neutral' in v: return 'neutral'
        return v
    canon_labels={i:canon(v) for i,v in labels.items()}
    if not {'positive','negative'}.issubset(set(canon_labels.values())): raise RuntimeError(f'Unexpected model labels: {id2label}')
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); model.to(device)
    probs=[]; batch=32
    texts=work.text_for_sentiment.tolist()
    for st in range(0,len(texts),batch):
        enc=tok(texts[st:st+batch],padding=True,truncation=True,max_length=256,return_tensors='pt').to(device)
        with torch.no_grad(): p=torch.softmax(model(**enc).logits,dim=-1).cpu().numpy()
        probs.append(p)
    probs=np.vstack(probs); pos_i=next(i for i,v in canon_labels.items() if v=='positive'); neg_i=next(i for i,v in canon_labels.items() if v=='negative')
    work['sentiment_score']=probs[:,pos_i]-probs[:,neg_i]; work['sentiment_confidence']=probs.max(axis=1); work['sentiment_label']=[canon_labels[int(i)] for i in probs.argmax(axis=1)]
    for i,v in canon_labels.items(): work['prob_'+v]=probs[:,i]
    dist=work['sentiment_label'].value_counts().reindex(['positive','neutral','negative'],fill_value=0); distdf=pd.DataFrame({'sentiment_label':dist.index,'count':dist.values}); distdf['percentage']=distdf['count']/len(work)*100; distdf.to_csv(out/'sentiment_distribution.csv',index=False,encoding='utf-8-sig')
    plt.figure(figsize=(8,5)); ax=dist.plot(kind='bar'); ax.set_title(f'Phân bố cảm xúc (n={len(work):,})'); ax.set_xlabel('Cảm xúc'); ax.set_ylabel('Số bình luận'); ax.set_xticklabels(['Tích cực','Trung tính','Tiêu cực'],rotation=0); [ax.text(i,v+max(1,len(work)*.01),f'{int(v)} ({v/len(work)*100:.1f}%)',ha='center') for i,v in enumerate(dist.values)]; plt.tight_layout(); plt.savefig(out/'sentiment_distribution.png',dpi=180); plt.close()
    plt.figure(figsize=(8,5)); plt.hist(work.sentiment_score,bins=25); plt.title(f'Phân bố sentiment score (n={len(work):,})'); plt.xlabel('P(tích cực) − P(tiêu cực)'); plt.ylabel('Số bình luận'); plt.tight_layout(); plt.savefig(out/'sentiment_score_distribution.png',dpi=180); plt.close()

    # BERTopic
    from sentence_transformers import SentenceTransformer
    from bertopic import BERTopic
    from umap import UMAP
    from hdbscan import HDBSCAN
    from sklearn.feature_extraction.text import CountVectorizer
    emb_id='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
    n=len(work); min_topic=max(5,min(25,max(5,n//80))); min_cluster=max(5,min_topic); nn=max(5,min(15,int(np.sqrt(max(n,25)))))
    vectorizer=CountVectorizer(ngram_range=(1,2),min_df=max(1,2),max_df=.95)
    umap=UMAP(n_neighbors=nn,n_components=5,min_dist=.0,metric='cosine',random_state=args.seed)
    hdb=HDBSCAN(min_cluster_size=min_cluster,min_samples=min_cluster,prediction_data=True)
    emb=SentenceTransformer(emb_id)
    vectors=emb.encode(work.text_for_topic.tolist(),show_progress_bar=False,normalize_embeddings=True)
    topic_model=BERTopic(embedding_model=None,umap_model=umap,hdbscan_model=hdb,vectorizer_model=vectorizer,calculate_probabilities=True,verbose=False,language='multilingual',min_topic_size=min_topic,low_memory=True)
    topics, probs2=topic_model.fit_transform(work.text_for_topic.tolist(),vectors)
    work['topic_id']=topics
    if probs2 is not None:
        work['topic_probability']=[float(np.max(x)) if np.ndim(x)>0 else float(x) for x in probs2]
    else: work['topic_probability']=np.nan
    info=topic_model.get_topic_info(); names={-1:'Other / Unassigned'}
    for tid in info.Topic:
        if int(tid)==-1: continue
        words=[w for w,_ in topic_model.get_topic(int(tid))[:5]]
        names[int(tid)]=' / '.join(words[:4]) if words else f'Thảo luận chủ đề {tid}'
    work['topic_name_vi']=work.topic_id.map(names).fillna('Other / Unassigned')
    # representative masked examples
    rows=[]
    total=len(work)
    for tid,g in work.groupby('topic_id',sort=True):
        kws=[w for w,_ in topic_model.get_topic(int(tid))[:10]] if int(tid)!=-1 else []
        ex=g.text_norm.head(2).map(mask).tolist(); rows.append({'topic_id':int(tid),'topic_name_vi':names.get(int(tid),'Other / Unassigned'),'count':len(g),'share_all_retained':len(g)/total*100,'top_keywords':', '.join(kws),'representative_masked_examples':' || '.join(ex)})
    topicdf=pd.DataFrame(rows).sort_values(['topic_id']); topicdf.to_csv(out/'topic_summary.csv',index=False,encoding='utf-8-sig')
    cros=pd.crosstab(work.topic_name_vi,work.sentiment_label).reindex(columns=['positive','neutral','negative'],fill_value=0); cros['total']=cros.sum(axis=1); cros['negative_pct']=cros['negative']/cros['total']*100; cros.to_csv(out/'topic_sentiment_crosstab.csv',encoding='utf-8-sig')
    top=topicdf.sort_values('count',ascending=False).head(15).iloc[::-1]; plt.figure(figsize=(10,max(6,.35*len(top)))); ax=plt.gca(); ax.barh(top.topic_name_vi,top['count']); ax.set_xlabel('Số bình luận'); ax.set_title(f'Top chủ đề (n={total:,}; -1 giữ riêng trong CSV)');
    for i,(c,p) in enumerate(zip(top['count'],top.share_all_retained)): ax.text(c,i,f' {int(c)} ({p:.1f}%)',va='center'); plt.tight_layout(); plt.savefig(out/'topic_bar_chart.png',dpi=180,bbox_inches='tight'); plt.close()
    heat=cros.sort_values('negative_pct',ascending=True); plt.figure(figsize=(9,max(5,.32*len(heat)))); ax=plt.gca(); im=ax.imshow(heat[['negative_pct']].values,aspect='auto'); ax.set_yticks(range(len(heat))); ax.set_yticklabels(heat.index); ax.set_xticks([0]); ax.set_xticklabels(['Tỷ lệ tiêu cực (%)']);
    for i,v in enumerate(heat.negative_pct): ax.text(0,i,f'{v:.1f}%',ha='center',va='center'); plt.title('Tỷ lệ sentiment tiêu cực theo chủ đề'); plt.tight_layout(); plt.savefig(out/'topic_sentiment_heatmap.png',dpi=180,bbox_inches='tight'); plt.close()

    export_cols=['source_row_id','text_for_sentiment','sentiment_label','sentiment_score','sentiment_confidence','topic_id','topic_probability','topic_name_vi','commercial_label','commercial_reason']+[c for c in work.columns if c.startswith('prob_')]
    export=work[export_cols].copy().rename(columns={'text_for_sentiment':'text_masked'}); export['text_masked']=export.text_masked.map(mask); export.to_csv(out/'comment_level_results.csv',index=False,encoding='utf-8-sig')

    # Report with measured findings only.
    neg=dist.get('negative',0); pos=dist.get('positive',0); neu=dist.get('neutral',0)
    lines=['# Báo cáo NLP — Bình luận Facebook robot hút bụi','',f'## 1. Phạm vi dữ liệu','',f'- Workbook: `{Path(args.input).name}`, worksheet: `{args.sheet}`.','- Chỉ phân tích cột `text`; không dùng `group_posts`.','- Tổng dòng đọc: **{n0:,}**; null: **{null:,}**; rỗng/whitespace: **{empty:,}**; duplicate exact bị loại: **{dup:,}**.','- Retained primary (organic + mixed_commercial có nội dung vấn đề): **{len(work):,}**.','- Phân tích organic-only sensitivity: **{int(organic.sum()):,}**.', '', '## 2. Sentiment', '', f'- Tích cực: **{pos:,} ({pos/len(work)*100:.1f}%)**; trung tính: **{neu:,} ({neu/len(work)*100:.1f}%)**; tiêu cực: **{neg:,} ({neg/len(work)*100:.1f}%)**.','- Score = P(positive) − P(negative); confidence = xác suất lớp cao nhất.','- Mô hình: `wonrax/phobert-base-vietnamese-sentiment`; label mapping được đọc từ model config tại runtime.','- Đây là classifier tự động, chưa có human-labeled validation; không diễn giải thành satisfaction survey.','', '## 3. Chủ đề', '', 'Các chủ đề dưới đây là mô tả từ topic keywords/representative comments của dữ liệu, không phải nguyên nhân nhân quả.','']
    for _,r in topicdf.sort_values('count',ascending=False).head(10).iterrows(): lines.append(f"- **Topic {int(r.topic_id)} — {r.topic_name_vi}**: {int(r['count']):,} bình luận ({r.share_all_retained:.1f}%). Keywords: {r.top_keywords}. Evidence: {r.representative_masked_examples}")
    lines += ['', '## 4. Pain points / cơ hội quan sát được', '', 'Các điểm dưới đây phải được đọc như tín hiệu thảo luận. Tần suất chủ đề không chứng minh tác động đến mua hàng.']
    for _,r in topicdf.sort_values('count',ascending=False).head(8).iterrows():
        lines.append(f"- **Topic {int(r.topic_id)} ({r.topic_name_vi})** — {int(r['count']):,} comments: xem keywords và examples trong `topic_summary.csv`; cần đọc định tính trước khi biến thành thông điệp sản phẩm/dịch vụ.")
    lines += ['', '## 5. Giới hạn & cảnh báo', '', '- Facebook group comments là mẫu không ngẫu nhiên; có sampling bias và promotional contamination.','- Mixed-commercial được giữ nếu có nội dung vấn đề; commercial thuần bị loại khỏi primary.','- BERTopic phụ thuộc embedding/UMAP/HDBSCAN và có thể không ổn định khi corpus nhỏ/thưa; topic -1 được giữ trong denominator.','- Sentiment model có domain shift; sarcasm, chẩn đoán kỹ thuật như “lỗi cảm biến” có thể không đồng nghĩa cảm xúc tiêu cực.','- Không có user ID thì “share” là share comment, không phải share unique customer.','', '## 6. Reproducibility', '',f'- Seed: {args.seed}; Python: {platform.python_version()}; runtime: {time.time()-t0:.1f}s; device: {device}.',f'- Libraries: pandas {pkgver("pandas")}, numpy {pkgver("numpy")}, transformers {pkgver("transformers")}, torch {pkgver("torch")}, sentence-transformers {pkgver("sentence-transformers")}, bertopic {pkgver("bertopic")}, umap-learn {pkgver("umap-learn")}, hdbscan {pkgver("hdbscan")}.']
    (out/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    meta={'input':str(args.input),'sheet':args.sheet,'seed':args.seed,'rows_original':n0,'null_text':null,'empty_text':empty,'exact_duplicates_removed':dup,'retained_primary':len(work),'retained_organic_only':int(organic.sum()),'commercial_counts':df.commercial_label.value_counts().to_dict(),'sentiment_counts':dist.to_dict(),'topic_counts':{str(k):int(v) for k,v in work.topic_id.value_counts().sort_index().items()},'models':{'sentiment':model_id,'embedding':emb_id},'parameters':{'batch_size':batch,'max_length':256,'min_topic_size':min_topic,'hdbscan_min_cluster_size':min_cluster,'umap_n_neighbors':nn,'seed':args.seed},'environment':{'python':platform.python_version(),'platform':platform.platform(),'device':str(device),'libraries':{x:pkgver(x) for x in ['pandas','numpy','transformers','torch','sentence-transformers','bertopic','umap-learn','hdbscan']}},'warnings':['No human-labeled validation performed; review CSV is exported for human checking.','Topic names are derived from topic keywords; human review recommended.'],'runtime_seconds':time.time()-t0}
    (out/'run_metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    # Acceptance checks
    assert len(distdf)==3 and int(distdf['count'].sum())==len(work)
    assert int(topicdf['count'].sum())==len(work)
    assert int(cros[['positive','neutral','negative']].sum().sum())==len(work)
    for p in out.glob('*.png'): assert p.stat().st_size>0
    for p in [out/'comment_level_results.csv',out/'topic_summary.csv',out/'report.md']:
        txt=p.read_text(encoding='utf-8') if p.suffix in ['.md'] else p.read_text(encoding='utf-8-sig')
        assert not PHONE_RE.search(txt) and not EMAIL_RE.search(txt), f'PII leak in {p}'
    print(json.dumps({'status':'SUCCESS','input':str(Path(args.input).resolve()),'output':str(out.resolve()),'retained':len(work),'sentiment':dist.to_dict(),'topics':len(topicdf),'runtime_seconds':round(time.time()-t0,1)},ensure_ascii=False))

if __name__=='__main__': main()
