import os, zipfile, json, random, shutil, warnings, time
from pathlib import Path
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
SEED=42; random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.set_num_threads(2)
BASE=Path('/mnt/data/roadvision_ai_executed'); DATA=BASE/'data'; IMG_DIR=DATA/'Images'; OUT=BASE/'outputs'; FIG=OUT/'figures'; MODELS=BASE/'models'
for p in [DATA, OUT, FIG, MODELS]: p.mkdir(parents=True, exist_ok=True)
if not IMG_DIR.exists() or len(list(IMG_DIR.glob('*.jpg'))) < 100:
    with zipfile.ZipFile('/mnt/data/Images.zip') as z: z.extractall(DATA)
print('Preparing data...', flush=True)
labels=pd.read_csv('/mnt/data/labels.csv',header=None,names=['image_id','class','xmin','ymin','xmax','ymax'])
labels['image_id_str']=labels.image_id.astype(int).map(lambda x:f'{x:08d}')
image_paths=sorted(IMG_DIR.glob('*.jpg')); image_ids={p.stem for p in image_paths}
labels=labels[labels.image_id_str.isin(image_ids)].copy()
labels['box_area']=(labels.xmax-labels.xmin).clip(lower=0)*(labels.ymax-labels.ymin).clip(lower=0)
rep=labels.sort_values(['image_id_str','box_area'],ascending=[True,False]).drop_duplicates('image_id_str').copy()
rep['path']=rep.image_id_str.map(lambda s:str(IMG_DIR/f'{s}.jpg'))
rep=rep[rep.path.map(lambda p:Path(p).exists())].reset_index(drop=True)
# dimensions
wh=[]
for p in rep.path:
    im=Image.open(p); wh.append((im.width,im.height))
rep['width']=[w for w,h in wh]; rep['height']=[h for w,h in wh]
for src,dst,den in [('xmin','x1n','width'),('ymin','y1n','height'),('xmax','x2n','width'),('ymax','y2n','height')]: rep[dst]=(rep[src]/rep[den]).clip(0,1)
rep=rep[(rep.x2n>rep.x1n)&(rep.y2n>rep.y1n)].reset_index(drop=True)
# subset with minimum class support
max_records=300
rep_sub=rep.sample(min(max_records,len(rep)),random_state=SEED).copy()
# keep classes with >=8 examples in subset; if too few, use top 8 classes from full set then sample
cnt=rep_sub['class'].value_counts()
if (cnt>=8).sum()<2:
    top=rep['class'].value_counts().head(8).index
    rep_sub=rep[rep['class'].isin(top)].sample(min(max_records, len(rep[rep['class'].isin(top)])), random_state=SEED).copy()
else:
    rep_sub=rep_sub[rep_sub['class'].isin(cnt[cnt>=8].index)].copy()
le=LabelEncoder(); rep_sub['class_id']=le.fit_transform(rep_sub['class']); classes=le.classes_.tolist()
train_df,temp_df=train_test_split(rep_sub,test_size=.30,random_state=SEED,stratify=rep_sub.class_id)
val_df,test_df=train_test_split(temp_df,test_size=.50,random_state=SEED,stratify=temp_df.class_id)
for name,df in [('object_detection_training_table',rep),('object_detection_modeling_subset',rep_sub),('train_split',train_df),('val_split',val_df),('test_split',test_df)]: df.to_csv(OUT/f'{name}.csv',index=False)
summary={'n_images_zip':len(image_paths),'n_label_rows_total':int(pd.read_csv('/mnt/data/labels.csv',header=None).shape[0]),'n_label_rows_matching_images':int(labels.shape[0]),'n_images_with_labels':int(rep.shape[0]),'n_modeling_records':int(rep_sub.shape[0]),'classes':classes}
# Plots
labels['class'].value_counts().plot(kind='bar',figsize=(10,5),title='All Annotation Class Distribution'); plt.tight_layout(); plt.savefig(FIG/'annotation_class_distribution.png',dpi=150); plt.close()
rep_sub['class'].value_counts().plot(kind='bar',figsize=(10,5),title='Modeling Subset Class Distribution'); plt.tight_layout(); plt.savefig(FIG/'modeling_subset_class_distribution.png',dpi=150); plt.close()
class DS(Dataset):
    def __init__(self,df): self.df=df.reset_index(drop=True)
    def __len__(self): return len(self.df)
    def __getitem__(self,i):
        r=self.df.iloc[i]; im=Image.open(r.path).convert('RGB').resize((64,64)); arr=np.asarray(im,dtype=np.float32)/255.0; arr=(arr-.5)/.5
        return torch.tensor(arr).permute(2,0,1).float(), torch.tensor(int(r.class_id)).long(), torch.tensor([r.x1n,r.y1n,r.x2n,r.y2n],dtype=torch.float32)
class M(nn.Module):
    def __init__(self,n):
        super().__init__(); self.f=nn.Sequential(nn.Conv2d(3,8,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(8,16,3,padding=1),nn.ReLU(),nn.AdaptiveAvgPool2d((1,1)),nn.Flatten()); self.c=nn.Linear(16,n); self.b=nn.Sequential(nn.Linear(16,4),nn.Sigmoid())
    def forward(self,x): h=self.f(x); return self.c(h), self.b(h)
def iou(a,b):
    xA=torch.maximum(a[:,0],b[:,0]); yA=torch.maximum(a[:,1],b[:,1]); xB=torch.minimum(a[:,2],b[:,2]); yB=torch.minimum(a[:,3],b[:,3]); inter=(xB-xA).clamp(min=0)*(yB-yA).clamp(min=0); area1=(a[:,2]-a[:,0]).clamp(min=0)*(a[:,3]-a[:,1]).clamp(min=0); area2=(b[:,2]-b[:,0]).clamp(min=0)*(b[:,3]-b[:,1]).clamp(min=0); return inter/(area1+area2-inter+1e-8)
train_loader=DataLoader(DS(train_df),batch_size=64,shuffle=True,num_workers=0); val_loader=DataLoader(DS(val_df),batch_size=64); test_loader=DataLoader(DS(test_df),batch_size=64)
model=M(len(classes)); opt=torch.optim.Adam(model.parameters(),lr=1e-3); ce=nn.CrossEntropyLoss(); sl=nn.SmoothL1Loss()
def eval_loader(loader):
    model.eval(); tot=0; corr=0; losses=[]; ious=[]
    with torch.no_grad():
        for x,y,b in loader:
            logits,pb=model(x); loss=ce(logits,y)+5*sl(pb,b); losses.append(loss.item()*len(x)); tot+=len(x); corr+=(logits.argmax(1)==y).sum().item(); ious+=iou(pb,b).numpy().tolist()
    return {'loss':sum(losses)/tot,'accuracy':corr/tot,'mean_iou':float(np.mean(ious))}
print('Training model...', flush=True)
hist=[]
for ep in range(1,3):
    model.train(); total=0; n=0
    for bi,(x,y,b) in enumerate(train_loader):
        opt.zero_grad(); logits,pb=model(x); loss=ce(logits,y)+5*sl(pb,b); loss.backward(); opt.step(); total+=loss.item()*len(x); n+=len(x)
    vm=eval_loader(val_loader); rec={'epoch':ep,'train_loss':total/n,**{f'val_{k}':v for k,v in vm.items()}}; hist.append(rec); print(rec, flush=True)
metrics={'train':eval_loader(train_loader),'validation':eval_loader(val_loader),'test':eval_loader(test_loader),'history':hist,'summary':summary}
json.dump(metrics,open(OUT/'model_metrics.json','w'),indent=2)
torch.save({'model_state_dict':model.state_dict(),'classes':classes,'input_size':64,'metrics':metrics},MODELS/'roadvision_tiny_detector.pt')
# curves
h=pd.DataFrame(hist); plt.figure(figsize=(7,5)); plt.plot(h.epoch,h.train_loss,marker='o',label='Train'); plt.plot(h.epoch,h.val_loss,marker='o',label='Validation'); plt.legend(); plt.title('Training Loss'); plt.tight_layout(); plt.savefig(FIG/'training_loss_curve.png',dpi=150); plt.close()
plt.figure(figsize=(7,5)); plt.plot(h.epoch,h.val_accuracy,marker='o',label='Validation Accuracy'); plt.plot(h.epoch,h.val_mean_iou,marker='o',label='Validation Mean IoU'); plt.legend(); plt.title('Validation Performance'); plt.tight_layout(); plt.savefig(FIG/'validation_performance.png',dpi=150); plt.close()
# sample predictions
pred_dir=OUT/'sample_predictions'; pred_dir.mkdir(exist_ok=True)
rows=[]
for _,r in test_df.sample(min(6,len(test_df)),random_state=SEED).iterrows():
    x,_,_=DS(pd.DataFrame([r]))[0]
    with torch.no_grad(): logits,pb=model(x.unsqueeze(0)); pr=torch.softmax(logits,1).numpy()[0]; pred_idx=int(pr.argmax()); box=pb.numpy()[0]
    im=Image.open(r.path).convert('RGB'); W,H=im.size; dr=ImageDraw.Draw(im)
    true=[int(r.x1n*W),int(r.y1n*H),int(r.x2n*W),int(r.y2n*H)]; pred=[int(box[0]*W),int(box[1]*H),int(box[2]*W),int(box[3]*H)]
    dr.rectangle(true,outline='green',width=3); dr.rectangle(pred,outline='red',width=3); dr.text((5,5),f"true={r['class']} pred={classes[pred_idx]} {pr[pred_idx]:.2f}",fill='yellow')
    out=pred_dir/f"prediction_{r.image_id_str}.jpg"; im.save(out); rows.append({'image_id':r.image_id_str,'true_class':r['class'],'predicted_class':classes[pred_idx],'confidence':float(pr[pred_idx]),'true_box':true,'pred_box':pred})
pd.DataFrame(rows).to_csv(OUT/'sample_predictions.csv',index=False)
# Tesla EDA
print('Running Tesla EDA...', flush=True)
raw=pd.read_csv('/mnt/data/Tesla - Deaths.csv'); raw.columns=[c.strip() for c in raw.columns]; df=raw.dropna(how='all').copy();
if 'Case #' in df: df=df[df['Case #'].notna()].copy()
if 'Date' in df: df['Date_parsed']=pd.to_datetime(df['Date'],errors='coerce')
for c in df.columns:
    if c not in ['Date','Date_parsed','Country','State','Description','Source','Note','Deceased 1','Deceased 2','Deceased 3','Deceased 4','Model']:
        df[c]=pd.to_numeric(df[c],errors='coerce')
df.to_csv(OUT/'tesla_deaths_cleaned.csv',index=False)
eda={'raw_rows':int(raw.shape[0]),'usable_case_rows':int(df.shape[0]),'total_reported_deaths':float(df.get('Deaths',pd.Series(dtype=float)).fillna(0).sum()),'tesla_driver_death_events':int((df.get('Tesla driver',pd.Series(dtype=float)).fillna(0)>0).sum()),'tesla_occupant_death_events':int((df.get('Tesla Occupant',pd.Series(dtype=float)).fillna(0)>0).sum()),'cyclist_or_pedestrian_events':int((df.get('Cyclists/ Peds',pd.Series(dtype=float)).fillna(0)>0).sum()),'other_vehicle_collision_events':int((df.get('Other vehicle',pd.Series(dtype=float)).fillna(0)>0).sum()),'verified_autopilot_death_sum':float(df.get('Verified Tesla Autopilot Deaths',pd.Series(dtype=float)).fillna(0).sum())}
json.dump(eda,open(OUT/'tesla_eda_summary.json','w'),indent=2)
plots=[('Year','tesla_events_by_year','Tesla Death Events by Year',None),('Country','tesla_events_by_country','Top Countries by Event Count',10),('State','tesla_events_by_state','Top States by Event Count',15),('Deaths','deaths_per_event_distribution','Deaths per Accident Event',None),('Model','tesla_events_by_model','Event Distribution by Tesla Model',10),('Verified Tesla Autopilot Deaths','verified_autopilot_deaths_distribution','Verified Tesla Autopilot Deaths per Event',None)]
for col,fname,title,top in plots:
    if col in df:
        vals=df[col].replace('-',np.nan).dropna();
        if col in ['Year','Deaths','Verified Tesla Autopilot Deaths']: vals=vals.astype(int).value_counts().sort_index()
        else: vals=vals.astype(str).value_counts().head(top)
        if len(vals)>0: vals.plot(kind='bar',figsize=(10,5),title=title); plt.tight_layout(); plt.savefig(FIG/f'{fname}.png',dpi=150); plt.close()
readme=f"""# RoadVision AI: Vehicle Detection and Autopilot Safety Analytics\n\n## Executed end-to-end result\nThis repository contains an executed baseline for the capstone: a PyTorch computer-vision model for vehicle-type prediction and bounding-box localization, plus exploratory safety analytics using the Tesla deaths dataset.\n\n## Dataset validation\n- Images extracted: {summary['n_images_zip']}\n- Total annotation rows: {summary['n_label_rows_total']:,}\n- Annotation rows matching extracted images: {summary['n_label_rows_matching_images']:,}\n- Images with usable labels: {summary['n_images_with_labels']:,}\n- Modeling subset used for this CPU run: {summary['n_modeling_records']:,}\n- Usable Tesla event records: {eda['usable_case_rows']}\n\n## Model test metrics\n- Classification accuracy: {metrics['test']['accuracy']:.3f}\n- Mean IoU: {metrics['test']['mean_iou']:.3f}\n- Combined loss: {metrics['test']['loss']:.3f}\n\n## Business value\nRoadVision AI demonstrates how perception AI and safety analytics can support AV/ADAS programs, intelligent transportation systems, traffic monitoring, road-incident response, and executive safety-risk reviews.\n\n## Limitation\nThe image labels contain multiple objects per image, while this fast baseline trains on the largest object per image. For a stronger portfolio version, convert the full annotations to YOLO or COCO and train a true multi-object detector.\n"""
(BASE/'README.md').write_text(readme)
(BASE/'requirements.txt').write_text('pandas\nnumpy\nmatplotlib\nscikit-learn\npillow\ntorch\ntorchvision\n')
(BASE/'run_end_to_end.py').write_text(Path(__file__).read_text())
zip_path='/mnt/data/roadvision_ai_executed_project.zip'
with zipfile.ZipFile(zip_path,'w',zipfile.ZIP_DEFLATED) as z:
    for path in BASE.rglob('*'):
        if path.is_file():
            rel=path.relative_to(BASE)
            if str(rel).startswith('data/Images/') or rel.name in ['labels.csv','Tesla - Deaths.csv','1739779188_capstone1problemstatement.pdf']: continue
            z.write(path,Path('roadvision_ai_executed')/rel)
print('DONE',zip_path, flush=True)
print(json.dumps(metrics,indent=2), flush=True)
print(json.dumps(eda,indent=2), flush=True)
