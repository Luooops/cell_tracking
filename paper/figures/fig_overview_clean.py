"""Overview design: real microscopy, schematic associations, measured calibration.

Run with the project's torch Python environment. Existing paper figures are untouched.
"""
from pathlib import Path
import json
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Ellipse, FancyArrowPatch
from PIL import Image
import numpy as np
import data

ROOT = Path(__file__).resolve().parents[2]
REF = ROOT / 'development/figure_refs'
OUT = ROOT / 'development/overview_design'
OUT.mkdir(exist_ok=True)
W, H = 174.6, 103
BLUE, TEAL, RED = '#3B6FB6', '#2A918A', '#C66E59'
INK, MUTED, LINE = '#253C4B', '#627786', '#CEDDE5'
plt.rcParams.update({'font.family': 'Arial', 'font.size': 6.5,
                     'svg.fonttype': 'none', 'pdf.fonttype': 42,
                     'axes.linewidth': .55})
fig = plt.figure(figsize=(W/25.4, H/25.4), facecolor='white')
ax = fig.add_axes([0,0,1,1]); ax.set(xlim=(0,W), ylim=(H,0)); ax.axis('off')

def text(x,y,s,size=6.5,c=INK,weight='normal',ha='left',**kwargs):
    return ax.text(x,y,s,fontsize=size,color=c,fontweight=weight,ha=ha,va='center',**kwargs)

def box(x,y,w,h,fill,edge='none',r=1.8):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle=f'round,pad=0,rounding_size={r}',
                              facecolor=fill,edgecolor=edge,lw=.55))

def arrow(x,y,xx,yy,c=BLUE,lw=1,style='-|>',rad=0,dashed=False):
    ax.add_patch(FancyArrowPatch((x,y),(xx,yy),arrowstyle=style,mutation_scale=6,
                 lw=lw,color=c,connectionstyle=f'arc3,rad={rad}',
                 linestyle='--' if dashed else '-',shrinkA=0,shrinkB=0))

def cell(x,y,c=BLUE,w=3,h=2.3,fill=True,dashed=False):
    ax.add_patch(Ellipse((x,y),w,h,facecolor=c if fill else 'white',edgecolor=c,
                        linewidth=.85,alpha=.85 if fill else 1,
                        linestyle='--' if dashed else '-'))

def pic(name,x,y,w,h,crop=None):
    im=Image.open(REF/name)
    if crop: im=im.crop(crop)
    # Preserve aspect ratio; never stretch or alter microscopy intensities.
    ratio=min(w/im.width,h/im.height); iw,ih=im.width*ratio,im.height*ratio
    ax.imshow(im,extent=(x+(w-iw)/2,x+(w+iw)/2,y+(h+ih)/2,y+(h-ih)/2),aspect='auto',zorder=2)

stats=json.loads((REF/'challenge_stats.json').read_text())['scripta']
metrics,reliability,_=data.answers('scripta_cv_oof_tta')
ece=metrics['calibration']['round1_decision']['ece']
coverage=metrics['coverage']
cov_false=coverage['false_links']['mutual_info']['at_percent']['5']
cov_missed=coverage['missed_links']['mutual_info']['at_percent']['5']

# Message first: task, inputs and outputs are explicit before method details.
text(4,5,'ProGCT',12,BLUE,'bold')
text(25,5,'Cell tracking with confidence',11,INK,'bold')
text(4,11,'Segmented nuclei → cell identities, divisions and link probabilities',7,MUTED)

text(4,17,'a',8,INK,'bold'); text(8,17,'WHY LINKING IS HARD',6.2,MUTED,'bold')
xs=[4,46.4,88.8,131.2]; cw=39.4
titles=['Missed detections','Crowded nuclei','Unreliable overlap','Too many links to check']
for x,title in zip(xs,titles):
    box(x,20,cw,25,'#F4F7F8')
    text(x+cw/2,23.2,title,6.8,INK,'bold',ha='center')
for i,t in enumerate([8,9,10]):
    pic(f'gap_r07c02_t{t}_outlined.png',5.6+i*12.5,27,11.2,11.2)
    text(11.2+i*12.5,39.6,['t','t+1','t+2'][i],5.5,MUTED,ha='center')
text(23.7,43,'A cell disappears from the detections',5.5,MUTED,ha='center')
pic('crowded_r03c12_t15_outlined.png',48.5,26.5,35.2,13.8,crop=(0,105,500,301))
text(66.1,43,f"{stats['touching']:.0%} touch a neighbour",6,MUTED,ha='center')
pic('nooverlap_r09c01_t24_outlined.png',93,26.7,13.7,13.7)
pic('nooverlap_r09c01_t25_with_t24_outline_white.png',111.1,26.7,13.7,13.7)
text(109,33,'→',7,MUTED,ha='center')
text(108.5,43,f"{stats['iou_below_half']:.0%} have IoU < 0.5",6,MUTED,ha='center')
pic('uncertain_links_qualitative_fig.png',133.2,26.5,35.4,14,
    crop=(1587,394,2053,705))
text(150.9,43,'Which links need review?',6,MUTED,ha='center')

text(4,50,'b',8,INK,'bold')
text(8,50,'ProGCT',7,BLUE,'bold')
text(28,50,'Associate cells across time; quantify every link.',7,INK)
mx=[4,60.5,117]; mw=53.6
for x,fill in zip(mx,['#EEF4FC','#EDF7F5','#F2F4FA']): box(x,54,mw,31,fill)
for x,title in zip(mx,['Bridge missed frames','Learn cell identity and division','Know which links to trust']):
    text(x+mw/2,58,title,7,INK,'bold',ha='center')

# Gap schematic: missing detection is not a fabricated microscopy image.
for x,label in zip([13,30,47],['t','t+1','t+2']): text(x,61.8,label,5.6,MUTED,ha='center')
cell(13,68); cell(30,68,'#9DAEBC',fill=False,dashed=True); cell(47,68,TEAL)
arrow(15,67,45,67,BLUE,rad=-.20)
text(30,72.7,'missed',5.5,MUTED,ha='center')
text(30.8,77.8,'Frame-gap graph',6.6,BLUE,'bold',ha='center')
text(30.8,82,'One model for adjacent and gap links',5.7,MUTED,ha='center')

# Context schematic: dotted candidate edges vs selected continuation and division.
left=[(68,65),(68,70),(68,75)]; right=[(84,64),(84,69),(84,75)]
for a,b in [(left[0],right[1]),(left[1],right[0]),(left[2],right[1])]:
    arrow(*a,*b,'#B9CBD4',.5,style='-',dashed=True)
for a,b in zip(left,right): arrow(a[0]+1.6,a[1],b[0]-1.6,b[1],BLUE,.8)
for a in left: cell(*a,w=2.5,h=2)
for b in right: cell(*b,TEAL,w=2.5,h=2,fill=False)
arrow(68,66.2,68,68.6,'#8EA9BB',.55,style='-',dashed=True)
cell(95,69,RED,w=3.2,h=2.6)
for x,y in [(107,65),(107,73)]:
    arrow(97,69,x-1.8,y,RED,.85); cell(x,y,RED,w=2.6,h=2)
text(76,78,'GNN: appearance + context',5.7,BLUE,ha='center')
text(103,78,'Division',6.2,RED,ha='center')
text(87.3,82,'Unrelated / same cell / division',5.8,MUTED,ha='center')

# Actual calibration data, retaining the measured 95% Wilson interval.
cal=fig.add_axes([122/W,(H-77)/H,17/W,14/H])
r=reliability[reliability.population=='round1_decision']
p=r.frac_pos.to_numpy(); n=r.n.to_numpy(); z=1.96
center=(p+z*z/(2*n))/(1+z*z/n)
half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
cal.plot([0,1],[0,1],':',color='#AAB8C2',lw=.55)
cal.fill_between(r.mean_p,np.clip(center-half,0,1),np.clip(center+half,0,1),color=BLUE,alpha=.15,lw=0)
cal.plot(r.mean_p,p,color=BLUE,lw=1)
cal.set(xlim=(0,1),ylim=(0,1),xticks=[0,1],yticks=[0,1])
cal.spines[['top','right']].set_visible(False)
cal.spines[['bottom','left']].set_color('#8DA0AE')
cal.tick_params(labelsize=5,length=1.5,pad=1)
cal.set_facecolor('#F2F4FA')
text(130.5,81,'Predicted probability',5.4,MUTED,ha='center')
text(119.6,70,'Observed',5.2,MUTED,ha='center',rotation=90,rotation_mode='anchor')
text(143,64,'Measured calibration',5.7,INK,'bold')
text(143,68,f'OOF ECE {ece:.3f}',6.2,BLUE)
text(143,73,'Uncertainty flags',5.7,INK,'bold')
text(143,77,'links for review',5.7,MUTED)
text(147,83.5,'Scripta development wells',5.4,MUTED,ha='center')

# Output strip: a compact lineage is the tangible result, not another method block.
arrow(87.3,85.8,87.3,88.5,'#8AA9BC',.8)
box(4,89,166.6,11,'#F6F8FA',LINE)
text(8,94.5,'OUTPUT',6,MUTED,'bold')
for (x,y,xx,yy,c,dash) in [(29,94.5,36,94.5,BLUE,False),(36,94.5,43,92,RED,False),
                           (36,94.5,43,97,RED,False),(48,94.5,59,94.5,BLUE,True)]:
    arrow(x,y,xx,yy,c,.85,style='-',dashed=dash)
for x,y,c in [(29,94.5,BLUE),(36,94.5,BLUE),(43,92,RED),(43,97,RED),(48,94.5,BLUE),(59,94.5,TEAL)]:
    cell(x,y,c,w=1.6,h=1.3)
text(64,93,'Tracks + lineage',7,INK,'bold')
text(64,97,'Link probabilities + review flags',6,MUTED)
ax.plot([111,111],[91.2,97.8],color=LINE,lw=.6)
text(116,93,'Most uncertain 5% of rejected candidates',5.6,INK,'bold')
text(116,97,f'Find {cov_missed:.0f}% of missed candidates',6,BLUE)

# Render-time alignment of the repeated panel rectangles (image insets are exempt).
sys.path.insert(0,'D:/AI/skills/nature-figure/scripts')
from audit_panel_alignment import require_matplotlib_panel_alignment
panel_axes=[]
for i,(x,y,w,h) in enumerate([(x,20,cw,25) for x in xs]+[(x,54,mw,31) for x in mx]):
    a=fig.add_axes([x/W,(H-y-h)/H,w/W,h/H],label=f'panel_{i}')
    a.set_axis_off(); a.patch.set_alpha(0); panel_axes.append(a)
require_matplotlib_panel_alignment(fig,axes=panel_axes,
    row_groups=[[a.get_label() for a in panel_axes[:4]],
                [a.get_label() for a in panel_axes[4:]]],json_out=OUT/'overview.alignment.json')
for a in panel_axes: a.remove()
fig.canvas.draw()
fig.savefig(OUT/'progct_overview.pdf',facecolor='white')
fig.savefig(OUT/'progct_overview.svg',facecolor='white')
fig.savefig(OUT/'progct_overview.png',dpi=600,facecolor='white')
facts={'ece_oof_decision_band':ece,'false_link_coverage_at_5_percent':cov_false,
       'missed_candidate_coverage_at_5_percent':cov_missed,'challenge_stats':stats,
       'size_mm':[W,H]}
(OUT/'source_facts.json').write_text(json.dumps(facts,indent=2),encoding='utf-8')
print(OUT/'progct_overview.png')
