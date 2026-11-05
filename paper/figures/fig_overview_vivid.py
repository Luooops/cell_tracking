"""Vivid-guided revision of the manuscript overview; no layer-level architecture.

Existing Python/SVG workflow retained. Real masked appearance inputs use the
project's preprocessing. All graph geometry is schematic; results are measured.
"""
from pathlib import Path
import json
import sys
import copy
import xml.etree.ElementTree as ET
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Ellipse, Rectangle
from PIL import Image
import numpy as np
import pandas as pd
import fig_overview as source
import data

ROOT=Path(__file__).resolve().parents[2]
REF=ROOT/'development/paper_results/figure_refs'
RESULTS=ROOT/'development/paper_results'
OUT=ROOT/'development/overview_design/vivid'
OUT.mkdir(parents=True,exist_ok=True)
W,H=174.6,78
B,T,R,A='#3B6FB6','#298E87','#BD6856','#B1842F'
INK,MUTED,LINE='#273944','#61727E','#C9D5DE'
plt.rcParams.update({'font.family':'Arial','font.size':6,'svg.fonttype':'none','pdf.fonttype':42,
                     'image.composite_image':False})
fig=plt.figure(figsize=(W/25.4,H/25.4),facecolor='white')
ax=fig.add_axes([0,0,1,1]); ax.set(xlim=(0,W),ylim=(H,0)); ax.axis('off')
def txt(x,y,s,fs=6,c=INK,b=False,ha='left',**kw):
    return ax.text(x,y,s,fontsize=fs,color=c,fontweight='bold' if b else 'normal',
                   ha=ha,va='center',zorder=8,**kw)
def box(x,y,w,h,fc='#F7F8FA',ec=LINE,r=1.6,lw=.5):
    p=FancyBboxPatch((x,y),w,h,boxstyle=f'round,pad=0,rounding_size={r}',
                    fc=fc,ec=ec,lw=lw,zorder=0); ax.add_patch(p); return p
def seg(a,b,c=LINE,lw=.6,ls='-',z=2):
    ax.plot([a[0],b[0]],[a[1],b[1]],color=c,lw=lw,ls=ls,zorder=z)
def arr(a,b,c=B,rad=0,lw=.85,ls='-',z=4):
    p=FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=5,connectionstyle=f'arc3,rad={rad}',
                      color=c,lw=lw,linestyle=ls,shrinkA=0,shrinkB=0,zorder=z); ax.add_patch(p)
def cell(x,y,c=B,ghost=False,w=2.4,h=1.8):
    ax.add_patch(Ellipse((x,y),w,h,fc='white' if ghost else c,ec=c,lw=.7,
                        ls='--' if ghost else '-',zorder=5))
def raster(im,x,y,w,h):
    im=np.asarray(im); ih,iw=im.shape[:2]; q=min(w/iw,h/ih); iw*=q; ih*=q
    ax.imshow(im,cmap='gray' if im.ndim==2 else None,vmin=0 if im.ndim==2 else None,
              vmax=1 if im.ndim==2 else None,
              extent=(x+(w-iw)/2,x+(w+iw)/2,y+(h+ih)/2,y+(h-ih)/2),aspect='auto',zorder=3)
def pic(name,x,y,w,h,crop=None):
    im=Image.open(REF/name).convert('RGB')
    if crop: im=im.crop(crop)
    raster(im,x,y,w,h)

stats=json.loads((RESULTS/'data_statistics.json').read_text())['scripta']
numbers=pd.read_csv(RESULTS/'key_numbers.csv')
def number(section,quantity):
    row=numbers[(numbers.section==section)&(numbers.quantity==quantity)]
    assert len(row)==1,(section,quantity)
    return float(row.value.iloc[0])
lnk=number('scripta','LNK mean, ours'); lnk_ta=number('scripta','LNK mean, ta_ft')
metrics,rel,coverage=data.answers('scripta_cv_oof_tta')
ece=metrics['calibration']['round1_decision']['ece']
cf=metrics['coverage']['false_links']['mutual_info']['at_percent']['5']
cm=metrics['coverage']['missed_links']['mutual_info']['at_percent']['5']
g=data.gap_counts().groupby('method').sum(numeric_only=True)
gap=1-g.loc['ours','missed_g2']/g.loc['ours','gt_g2']
bc=data.dnn_wells('ours')['BC(0)'].mean()

# Three-column reading path, with the central method split into association and events.
txt(1,2.6,'a  Why tracking is difficult',7,b=True)
txt(47,2.6,'b  GNN: appearance and neighbourhood context',7,B,True)
txt(130,2.6,'d  Tracks you can review',7,b=True)
box(1,6,41,67,'#FAFBFC')
box(47,6,78,34.5,'#EFF5FC','#BACDE2')
box(47,45,78,28,'#EFF8F5','#B9D7CC')
box(130,6,43.6,32,'#F3F7FC','#C4D3E5')
box(130,40.5,43.6,32.5,'#F6F3FA','#D7CEE2')
arr((43,23),(46,23),'#8499AA',lw=.7)
arr((122,41),(122,44),'#8499AA',lw=.7)
arr((126,58),(129,58),'#8499AA',lw=.7)
arr((126,23),(129,23),'#8499AA',lw=.7)

# Four local challenge blocks; source images stay physically distinct from schematics.
for y,h,fc in [(7,16,'#F2F5F9'),(24.5,14,'#F0F7F4'),(40,15,'#FAF5F0'),(56.5,15,'#F5F2F8')]:
    box(2,y,39,h,fc,ec='none',r=1.1)
txt(3,9,'Missed detections',6.2,b=True)
for i,t in enumerate([8,9,10]):
    pic(f'gap_r07c02_t{t}_outlined.png',3+i*9.3,11,8.3,8.3)
    txt(7.15+i*9.3,21,['t','t+1','t+2'][i],5,MUTED,ha='center')
txt(31,14,f"{stats['share_tracks_with_gap']:.1%}",6,B,True)
txt(31,18,'of tracks',5,MUTED)
txt(3,26.6,'Crowding',6.2,b=True)
pic('crowded_r03c12_t15_outlined.png',3,28.5,15,8.5,crop=(0,100,500,390))
txt(20,31,f"{stats['touching']:.0%} touch",5.8,T,True)
txt(20,34.5,'a neighbour',5.5,MUTED)
txt(3,42.2,'Unreliable overlap',6.2,b=True)
pic('nooverlap_r09c01_t24_outlined.png',3,44,8.5,8.5)
pic('nooverlap_r09c01_t25_with_t24_outline_white.png',15,44,8.5,8.5)
arr((12,48.2),(14,48.2),MUTED,lw=.5)
txt(25,47,f"{stats['iou_below_half']:.0%} of pairs",5.6,R,True)
txt(25,50.8,'IoU < 0.5',5.5,MUTED)
txt(3,58.6,'Too many links to check',6.2,b=True)
pic('uncertain_links_qualitative_fig.png',3,60.3,13.2,9,
    crop=(1587,394,2053,705))
txt(18,63.4,'Which links',5.8)
txt(18,67,'need review?',5.8)

# Actual appearance crops, produced with the same mask/normalisation as the model.
im,mask=source.frame(10)
ids=list(source.centroids(mask))[:4]
crops=source.cell_crops(im,mask,ids)
assert len(crops)>=3
for i,crop in enumerate(crops[:3]):
    x,y=50+i*3.1,13+i*3.0
    raster(crop,x,y,8.2,8.2)
    ax.add_patch(Rectangle((x,y),8.2,8.2,fill=False,ec='#71869B',lw=.45,zorder=4))
txt(50,9,'Appearance',6.2,B,True)
txt(49.5,29,'Masked cell crops',5.4)
txt(49.5,32.4,'Gap g = 1 or 2',5.2,MUTED)
arr((65,23),(72,23),B,lw=.75)

# Multiple cells in both frames; one highlighted source with three alternatives.
box(73.5,12,16.5,21.8,'#E5EEF9','#AFC4DB',r=1)
box(105.5,12,16.5,21.8,'#E6F4F0','#AED1C7',r=1)
txt(81.75,9,'Frame t',5.8,B,True,ha='center')
txt(113.75,9,'Frame t+g',5.8,T,True,ha='center')
left=[(77,16),(86,17.5),(80,23),(87,28),(76.8,30)]
right=[(109,16),(118,18),(109,24),(117,28),(110,31)]
for nodes in [left,right]:
    for i,j in [(0,1),(0,2),(1,2),(2,3),(2,4),(3,4)]:
        seg(nodes[i],nodes[j],'#9AAFC0',.55,'--')
for i,j in [(0,0),(0,1),(1,0),(1,1),(1,2),(2,0),(2,2),(2,3),(3,2),(3,3),(3,4),(4,2),(4,4)]:
    seg(left[i],right[j],'#C3CDD6',.5,z=1)
# Highlight the focal candidate fan, not a pre-solved one-to-one matching.
for j in [0,2,3]: arr((81.4,23),(right[j][0]-1.4,right[j][1]),B,lw=.8,z=4)
for x,y in left: cell(x,y,B,w=2.5,h=2)
for x,y in right: cell(x,y,T,w=2.5,h=2)
ax.add_patch(Ellipse(left[2],4.4,3.8,fill=False,ec=B,lw=.8,zorder=6))
txt(97.3,35.9,'One cell → multiple candidates',5.7,B,True,ha='center')
seg((50,37.8),(54,37.8),'#9AAFC0',.6,'--'); txt(55,37.8,'Same-frame neighbours',5.1,MUTED)
seg((92,38),(96,38),B,.8); txt(97,38,'Cross-frame candidates',5.1,MUTED)

# One coherent timeline demonstrates the three event types, rather than a layer diagram.
txt(47,43,'c  Frame-gap linking and division decisions',7,T,True)
for x,s in [(58,'t'),(84,'t+1'),(110,'t+2')]:
    txt(x,47.2,s,5.4,MUTED,ha='center'); seg((x,49),(x,68),'#D3E2DD',.45)
for y in [51,57]:
    cell(58,y,B); cell(110,y,T)
cell(84,51,T); cell(84,57,'#A3B4B0',True)
arr((59.5,51),(82.5,51),B); arr((85.5,51),(108.5,51),B)
arr((59.5,57),(108.5,57),B,rad=-.10,ls='--')
txt(119,51,'Same',5.2,B,ha='center')
txt(119,57,'Gap',5.2,B,ha='center')
cell(58,64,R)
for y in [62,67]:
    arr((59.5,64),(82.5,y),R); cell(84,y,R)
    arr((85.5,y),(108.5,y),R); cell(110,y,R)
txt(119,64.5,'Divide',5.2,R,ha='center')
box(50,69.1,72,2.8,'#E2F0EA',ec='none',r=.8)
txt(86,70.5,'Three-class links: unrelated / same cell / division',5.7,ha='center')

# Evidence and tangible output. Comparison conditions are explained in caption.
txt(133,9,'Tracks + lineage + link confidence',6,b=True)
for x,y in [(134,14),(143,14),(151,12),(151,17),(160,12),(160,17)]:
    cell(x,y,R if x>=151 else B,w=1.8,h=1.3)
for a,b,c in [((135,14),(142,14),B),((144,14),(150,12),R),((144,14),(150,17),R),
              ((152,12),(159,12),R),((152,17),(159,17),R)]: seg(a,b,c,.85)
ax.add_patch(Ellipse((155.5,17),7.8,4.0,fill=False,ec=A,lw=.7,ls='--'))
txt(165.5,16,'review',5,A,ha='center')
txt(133,22,f'LNK {lnk:.3f} vs {lnk_ta:.3f}',6.5,B,True)
txt(133,25.5,'Scripta · vs fine-tuned Trackastra',5.1,MUTED)
txt(133,29.4,f'{gap:.0%} gap links recovered',6,B,True)
txt(133,33,f'BC {bc:.3f} · RAW264.7',6,T,True)
txt(133,36,'0 false divisions · Scripta',5.6)

txt(133,43.4,'Ensemble confidence + targeted review',5.8,b=True)
cal=fig.add_axes([134/W,(H-57)/H,14/W,10/H])
r=rel[rel.population=='round1_decision']; p=r.frac_pos.to_numpy(); n=r.n.to_numpy(); z=1.96
center=(p+z*z/(2*n))/(1+z*z/n); half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
cal.plot([0,1],[0,1],':',color='#AAA6B5',lw=.5)
cal.fill_between(r.mean_p,np.clip(center-half,0,1),np.clip(center+half,0,1),color=B,alpha=.15,lw=0)
cal.plot(r.mean_p,p,color=B,lw=.9)
cal.set(xlim=(0,1),ylim=(0,1),xticks=[],yticks=[])
cal.spines[['top','right']].set_visible(False)
cal.spines[['left','bottom']].set_color('#A8A3B1'); cal.set_facecolor('#F6F3FA')
txt(141,59,'Predicted',5,MUTED,ha='center')
txt(132.5,52,'Observed',5,MUTED,ha='center',rotation=90,rotation_mode='anchor')
txt(151,49.2,f'ECE {ece:.3f}',6,B,True)
txt(151,52.7,'Measured OOF',5.3,MUTED)
txt(151,56.2,'Scripta',5.3,MUTED)
txt(133,62.5,'Top 5% by ensemble disagreement',5.4,b=True)
txt(133,66,f'{cf:.0f}% false links',6,B,True)
txt(133,69.7,f'{cm:.0f}% missed candidates',6,B,True)
txt(133,72,'Selected / rejected edges ranked separately',5,MUTED)
txt(W/2,75.8,'ProGCT: link segmented cells across time, recognise divisions, and quantify uncertainty.',6,ha='center')

# Actual panel rectangles are measured. Unequal spans are deliberate, not a grid failure.
sys.path.insert(0,'D:/AI/skills/nature-figure/scripts')
from audit_panel_alignment import require_matplotlib_panel_alignment
panels=[]
for i,(x,y,w,h) in enumerate([(47,6,78,34.5),(47,45,78,28),(130,6,43.6,32),(130,40.5,43.6,32.5)]):
    a=fig.add_axes([x/W,(H-y-h)/H,w/W,h/H],label=f'panel{i}');a.axis('off');panels.append(a)
require_matplotlib_panel_alignment(fig,axes=panels,column_groups=[['panel0','panel1'],['panel2','panel3']],
    json_out=OUT/'alignment.json')
for a in panels:a.remove()

# Stable semantic keys on actual editable vector/raster objects, enabling part export.
fig.canvas.draw()
renderer=fig.canvas.get_renderer()
part_keys={k:[] for k in ['challenges','association','events','outcomes']}
for i,artist in enumerate(ax.get_children()):
    if not artist.get_visible() or artist is ax.patch: continue
    try:
        bb=artist.get_window_extent(renderer)
        x,y=ax.transData.inverted().transform(((bb.x0+bb.x1)/2,(bb.y0+bb.y1)/2))
    except Exception: continue
    if not np.isfinite([x,y]).all() or y>74: continue
    part='challenges' if x<44 else ('outcomes' if x>127 else ('association' if y<41 else 'events'))
    key=f'part-{part}-{i}'
    artist.set_gid(key);part_keys[part].append(key)
cal.set_gid('part-outcomes-calibration');part_keys['outcomes'].append('part-outcomes-calibration')
fig.savefig(OUT/'progct_overview_vivid.pdf')
fig.savefig(OUT/'progct_overview_vivid.svg')
fig.savefig(OUT/'progct_overview_vivid.png',dpi=600)

# SVG parts contain the actual named objects, not raster crops of the complete figure.
svg=ET.parse(OUT/'progct_overview_vivid.svg').getroot()
ns='{http://www.w3.org/2000/svg}'
part_dir=OUT/'parts';part_dir.mkdir(exist_ok=True)
regions={'challenges':(0,0,44,74),'association':(46,0,80,41),
         'events':(46,41,80,33),'outcomes':(129,0,45.6,74)}
index=[]
for name,region in regions.items():
    x,y,w,h=region;q=72/25.4
    root=ET.Element(ns+'svg',{'width':f'{w}mm','height':f'{h}mm',
        'viewBox':f'{x*q} {y*q} {w*q} {h*q}','version':'1.1'})
    for defs in svg.iter(ns+'defs'):root.append(copy.deepcopy(defs))
    keys=set(part_keys[name])
    for element in svg.iter():
        if element.get('id') in keys:root.append(copy.deepcopy(element))
    ET.ElementTree(root).write(part_dir/f'{name}.svg',encoding='utf-8',xml_declaration=True)
    index.append({'id':name,'svg':f'{name}.svg','element_keys':part_keys[name],
                  'bbox_mm':region})
(part_dir/'index.json').write_text(json.dumps(index,indent=2))
(OUT/'structure.json').write_text(json.dumps({'parts':index,'connections':[
    ['challenges','association'],['association','events'],['events','outcomes'],
    ['association','outcomes']]},indent=2))
(OUT/'facts.json').write_text(json.dumps(dict(lnk=lnk,lnk_ta=lnk_ta,gap=gap,bc=bc,ece=ece,
    false_coverage=cf,missed_coverage=cm,appearance_labels=[int(i) for i in ids],
    appearance_source='Scripta r03c12 t10; project appearance_crop preprocessing',
    size_mm=[W,H]),indent=2))
print(OUT)
