"""Compact manuscript overview: challenge → solution → outcome, 174.6 × 67 mm."""
from pathlib import Path
import json
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch, Rectangle
from PIL import Image
import numpy as np
import data

ROOT=Path(__file__).resolve().parents[2]
REF=ROOT/'development/figure_refs'
OUT=ROOT/'development/overview_design'
W,H=174.6,67
B,T,R='#3B6FB6','#2A918A','#BE715E'
INK,MUTED,LINE='#283B46','#627480','#D5DFE5'
plt.rcParams.update({'font.family':'Arial','font.size':6,'pdf.fonttype':42,'svg.fonttype':'none'})
fig=plt.figure(figsize=(W/25.4,H/25.4),facecolor='white')
ax=fig.add_axes([0,0,1,1]); ax.set(xlim=(0,W),ylim=(H,0)); ax.axis('off')
def txt(x,y,s,fs=6,c=INK,bold=False,ha='left',**kw):
    ax.text(x,y,s,fontsize=fs,color=c,weight='bold' if bold else 'normal',ha=ha,va='center',**kw)
def line(x,y,xx,yy,c=LINE,lw=.5,ls='-'):
    ax.plot([x,xx],[y,yy],color=c,lw=lw,ls=ls)
def arrow(x,y,xx,yy,c=B,rad=0,lw=.8):
    ax.add_patch(FancyArrowPatch((x,y),(xx,yy),arrowstyle='-|>',mutation_scale=5,
                 connectionstyle=f'arc3,rad={rad}',color=c,lw=lw,shrinkA=0,shrinkB=0))
def cell(x,y,c=B,ghost=False,w=2.7,h=1.9):
    ax.add_patch(Ellipse((x,y),w,h,fc='white' if ghost else c,ec=c,lw=.7,
                        ls='--' if ghost else '-'))
def pic(name,x,y,w,h,crop=None):
    im=Image.open(REF/name)
    if crop: im=im.crop(crop)
    q=min(w/im.width,h/im.height); iw,ih=im.width*q,im.height*q
    ax.imshow(im,extent=(x+(w-iw)/2,x+(w+iw)/2,y+(h+ih)/2,y+(h-ih)/2),aspect='auto')

metrics,rel,_=data.answers('scripta_cv_oof_tta')
ece=metrics['calibration']['round1_decision']['ece']
cov=metrics['coverage']
false=cov['false_links']['mutual_info']['at_percent']['5']
missed=cov['missed_links']['mutual_info']['at_percent']['5']
gaps=data.gap_counts().groupby('method').sum(numeric_only=True)
rejoined=1-gaps.loc['ours','missed_g2']/gaps.loc['ours','gt_g2']
bc=data.dnn_wells('ours')['BC(0)'].mean()
ndiv=data.scripta_divisions('ours',data.all_wells())['divisions']

# Small column headings; no poster-style title.
txt(2,3,'a  Challenges',7,bold=True)
txt(64,3,'b  ProGCT solutions',7,B,True)
txt(127,3,'c  Outcomes',7,bold=True)
line(1,6,173.6,6,lw=.7)
rows=[(7,17),(25,21),(47,16)]
for y,h in rows:
    ax.add_patch(Rectangle((62,y),60,h,fc='#EFF5FA',ec='none',zorder=-1))
for y in [24.5,46.5]: line(1,y,173.6,y)
for y in [15.5,35.5,55]:
    arrow(58,y,61,y,'#96AAB7',lw=.65)
    arrow(123,y,126,y,'#96AAB7',lw=.65)

# Row 1 — missed observation, frame-gap mechanism, recovered track.
txt(2,9,'Missed detection',6.5,bold=True)
for i,t in enumerate([8,9,10]):
    pic(f'gap_r07c02_t{t}_outlined.png',2+i*12.2,11,10.5,10.5)
    txt(7.25+i*12.2,23,['t','t+1','t+2'][i],5.2,MUTED,ha='center')
txt(40,14,'Cell present;',5.6,MUTED)
txt(40,17,'not detected',5.6,MUTED)
txt(65,9,'Frame-gap graph',6.5,B,True)
for x,label in [(72,'t'),(91,'t+1'),(110,'t+2')]:
    txt(x,12,label,5.2,MUTED,ha='center')
cell(72,17); cell(91,17,'#A6B4BD',True); cell(110,17,T)
arrow(74,16.5,108,16.5,rad=-.14)
txt(91,20,'Link across one missed frame',5.8,ha='center')
txt(91,23,'One model, gap as input',5.5,MUTED,ha='center')
txt(128,9,'Continuous tracks',6.5,bold=True)
for x,y in [(132,14),(143,15),(154,13),(165,14)]: cell(x,y,w=2,h=1.4)
line(133,14,142,15,B,.9); line(144,15,153,13,B,.9,'--'); line(155,13,164,14,B,.9)
txt(128,19,f'{rejoined:.0%} of gap links rejoined',6.5,B,True)
txt(128,22.5,'Across one missed frame · Scripta',5.4,MUTED)

# Row 2 — two related challenges share one association mechanism.
txt(2,27.5,'Crowding',6.3,bold=True)
txt(25,27.5,'Unreliable overlap',6.3,bold=True)
pic('crowded_r03c12_t15_outlined.png',2,30,19,11,crop=(0,100,500,390))
pic('nooverlap_r09c01_t24_outlined.png',25,30,11,11)
pic('nooverlap_r09c01_t25_with_t24_outline_white.png',41,30,11,11)
arrow(37,35.5,40,35.5,MUTED,lw=.55)
txt(2,44,'38% touching',5.5,MUTED)
txt(25,44,'43% same-cell IoU < 0.5',5.5,MUTED)
txt(65,27.5,'GNN: appearance + context',6.5,B,True)
for ya,yb in [(32,32.5),(36,35.5),(40,40)]:
    line(71,ya,84,yb,B,.85); cell(71,ya,w=2.1,h=1.5); cell(84,yb,T,w=2.1,h=1.5)
line(72,32,83,35.5,'#ACBDCA',.55,'--'); line(72,36,83,32.5,'#ACBDCA',.55,'--')
line(71,33,71,35,'#ACBDCA',.55,'--')
cell(98,36,R)
for y in [32,40]: arrow(100,36,109,y,R); cell(111,y,R,w=2.3,h=1.7)
txt(91,44,'Unrelated / same cell / division',5.7,ha='center')
txt(128,27.5,'Tracks + division lineage',6.5,bold=True)
line(130,33,140,33,B,.85)
for y in [30.5,35.5]: line(140,33,149,y,R,.85); cell(149,y,R,w=1.8,h=1.3)
cell(130,33,w=1.8,h=1.3); cell(140,33,w=1.8,h=1.3)
txt(154,33,'No division\nswitch',5.6,MUTED)
txt(128,39,f'BC {bc:.3f} · RAW264.7',6.2,B,True)
txt(128,43,f'{ndiv} false divisions · Scripta',6.1,bold=True)

# Row 3 — uncertainty is actionable; calibration remains an evaluated property.
txt(2,49.5,'Which links need review?',6.4,bold=True)
pic('uncertain_links_qualitative_fig.png',2,52,21,10,
    crop=(1587,394,2053,705))
txt(26,55,'Thousands of links;',5.8,MUTED)
txt(26,59,'limited review time',5.8,MUTED)
txt(65,49.5,'Link probability + uncertainty',6.5,B,True)
cal=fig.add_axes([67/W,(H-59.6)/H,11/W,7.5/H])
r=rel[rel.population=='round1_decision']; p=r.frac_pos.to_numpy(); n=r.n.to_numpy(); z=1.96
center=(p+z*z/(2*n))/(1+z*z/n); half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
cal.plot([0,1],[0,1],':',color='#9EAEB9',lw=.5)
cal.fill_between(r.mean_p,np.clip(center-half,0,1),np.clip(center+half,0,1),color=B,alpha=.16,lw=0)
cal.plot(r.mean_p,p,color=B,lw=.8)
cal.set(xlim=(0,1),ylim=(0,1),xticks=[],yticks=[])
cal.spines[['top','right']].set_visible(False)
cal.spines[['bottom','left']].set_color('#9EAEB9'); cal.set_facecolor('#EFF5FA')
txt(72.5,61.7,'Predicted',5,MUTED,ha='center')
txt(65.5,55.9,'Observed',5,MUTED,ha='center',rotation=90,rotation_mode='anchor')
txt(82,54.2,f'Measured ECE {ece:.3f}',6.1,B,True)
txt(82,57.5,'Out-of-fold · Scripta',5.4,MUTED)
txt(82,61,'Ensemble disagreement → review',5.4)
txt(128,49.5,'Review the most uncertain 5%',6.3,bold=True)
txt(128,54.2,f'{false:.0f}% of false links',6.3,B,True)
txt(128,58,f'{missed:.0f}% of missed candidates',6.3,B,True)
txt(128,61.5,'Rank selected / rejected edges separately',5,MUTED)
line(1,64,173.6,64,lw=.6)
txt(W/2,65.7,'Segmented nuclei → tracks, lineage and confidence for every link',6,ha='center')

sys.path.insert(0,'D:/AI/skills/nature-figure/scripts')
from audit_panel_alignment import require_matplotlib_panel_alignment
panels=[]
for i,(y,h) in enumerate(rows):
    for j,(x,w) in enumerate([(1,55),(62,60),(127,46.6)]):
        a=fig.add_axes([x/W,(H-y-h)/H,w/W,h/H],label=f'r{i}c{j}')
        a.axis('off'); panels.append(a)
require_matplotlib_panel_alignment(fig,axes=panels,
    column_groups=[[f'r{i}c{j}' for i in range(3)] for j in range(3)],
    json_out=OUT/'compact.alignment.json')
for a in panels: a.remove()
fig.savefig(OUT/'progct_overview_compact.pdf')
fig.savefig(OUT/'progct_overview_compact.svg')
fig.savefig(OUT/'progct_overview_compact.png',dpi=600)
(OUT/'compact_facts.json').write_text(json.dumps(dict(gap_rejoined=rejoined,bc=bc,
    scripta_false_divisions=int(ndiv),ece=ece,false_link_coverage=false,
    missed_candidate_coverage=missed,width_mm=W,height_mm=H),indent=2))
print('Compact overview exported')
