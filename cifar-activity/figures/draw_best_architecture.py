"""Draw the validation-selected CIFAR architecture using paper-style vector blocks."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

OUT = Path(__file__).resolve().parent
plt.rcParams.update({'font.family': 'DejaVu Sans', 'svg.fonttype': 'none'})
fig, ax = plt.subplots(figsize=(13, 11))
fig.patch.set_facecolor('white')
ax.set(xlim=(0,13), ylim=(0,11)); ax.axis('off')
ink='#283342'; conv='#b9d6eb'; norm='#f9e5a6'; act='#d0e5c4'; head='#ddcee9'; pool='#f3d0b9'

def txt(x,y,s,size=11,**kw):
    ax.text(x,y,s,fontsize=size,color=ink,ha='center',va='center',**kw)
def box(x,y,w,h,s,c,size=11):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.025,rounding_size=0.09',facecolor=c,edgecolor=ink,linewidth=1.25))
    txt(x+w/2,y+h/2,s,size)
def arrow(a,b,style='-|>',lw=1.4):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle=style,mutation_scale=12,color=ink,linewidth=lw))
def line(points):
    ax.plot(*zip(*points),color=ink,lw=1.4)

txt(6.5,10.65,'Current best architecture · CIFAR-10',21,fontweight='bold')
txt(6.5,10.25,'32-convolution residual CNN  |  1,239,274 trainable parameters',12)
# Left: bottom-up full architecture.
x=1.3; w=4.0; cx=x+w/2
box(x,0.9,w,.55,'RGB image  ·  3 × 32 × 32','#eeeeee')
box(x,1.8,w,.6,'Channel normalization','#eeeeee')
arrow((cx,1.45),(cx,1.8))
# Each stage explicitly gives transitions and repetition count.
box(x,2.8,w,1.05,'Residual block × 8\nC = 32   ·   first block: 3 → 32',conv,12)
arrow((cx,2.4),(cx,2.8)); txt(0.67,3.32,'Stage 1',10,rotation=90)
box(x,4.2,w,.55,'Max pool 2 × 2  ·  stride 2',pool)
arrow((cx,3.85),(cx,4.2)); txt(5.95,4.48,'32 × 16 × 16',10)
box(x,5.15,w,1.05,'Residual block × 8\nC = 64   ·   first block: 32 → 64',conv,12)
arrow((cx,4.75),(cx,5.15)); txt(0.67,5.67,'Stage 2',10,rotation=90)
box(x,6.55,w,.55,'Max pool 2 × 2  ·  stride 2',pool)
arrow((cx,6.2),(cx,6.55)); txt(5.93,6.83,'64 × 8 × 8',10)
box(x,7.45,w,.5,'Flatten  ·  4,096 features','#eeeeee')
arrow((cx,7.1),(cx,7.45))
box(x,8.25,w,.6,'Linear 4,096 → 128  +  ReLU',head)
arrow((cx,7.95),(cx,8.25))
box(x,9.2,w,.6,'Linear 128 → 10  ·  logits',head)
arrow((cx,8.85),(cx,9.2))
# Right: expanded two-convolution block, with skip addition before ReLU.
bx=8.15; bw=3.05; bc=bx+bw/2
ax.add_patch(FancyBboxPatch((7.55,2.12),4.7,7.1,boxstyle='round,pad=0.03,rounding_size=0.12',facecolor='#fafafa',edgecolor='#a6aeb7',linewidth=1.1))
txt(9.9,9.57,'Inside each residual block',14,fontweight='bold')
txt(bc,2.43,'Input x  ·  Cᵢₙ × H × W',11)
arrow((bc,2.63),(bc,3.05))
box(bx,3.05,bw,.65,'3 × 3 Conv  ·  Cᵢₙ → C',conv)
box(bx,4.05,bw,.65,'Batch normalization',norm)
arrow((bc,3.7),(bc,4.05))
box(bx,5.05,bw,.55,'ReLU',act)
arrow((bc,4.7),(bc,5.05))
box(bx,5.95,bw,.65,'3 × 3 Conv  ·  C → C',conv)
arrow((bc,5.6),(bc,5.95))
box(bx,6.95,bw,.65,'Batch normalization',norm)
arrow((bc,6.6),(bc,6.95))
ax.add_patch(Circle((bc,8.08),.19,facecolor='white',edgecolor=ink,lw=1.3)); txt(bc,8.08,'+',17)
arrow((bc,7.6),(bc,7.89))
box(bx,8.55,bw,.45,'ReLU',act)
arrow((bc,8.27),(bc,8.55))
# Actual skip route, zero channel padding on transitions; no learned projection.
ax.add_patch(Circle((bc,2.8),.035,color=ink))
line([(bc,2.8),(11.82,2.8),(11.82,8.08)])
arrow((11.82,8.08),(bc+.19,8.08))
txt(12.03,5.55,'Identity / zero-pad channels',10,rotation=90)
# Mathematical specification.
txt(9.9,1.55,'y = ReLU(BN₂(Conv₂(ReLU(BN₁(Conv₁(x))))) + S(x))',10)
txt(9.9,1.03,'Both convs: stride 1, padding 1, no bias\nS(x): identity; append zeros when channels increase',10,linespacing=1.55)
# Figure footer.
txt(6.5,.28,'Validation: 87.20 ± 0.57%   ·   Test: 86.43 ± 0.49%   ·   3 seeds, 160 epochs, no augmentation',11)
fig.subplots_adjust(left=.015,right=.985,top=.99,bottom=.01)
for ext in ('svg','pdf','png'):
    fig.savefig(OUT/f'best_residual_cnn_architecture.{ext}',dpi=190,facecolor='white')
