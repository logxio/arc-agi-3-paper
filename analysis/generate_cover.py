"""Scientific overview from the archived frame geometry and measured score receipts."""
from pathlib import Path
import sys,json,hashlib,re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
from matplotlib.collections import PolyCollection
from matplotlib.colors import to_rgb, rgb_to_hsv
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'analysis'))
import figcheck
import generate_xray_figures as src

def render_cover(output_dir, report):
    OUT=ROOT/'.ai/dispatch'
    inputs=['analysis/out/x0_specimens.json','analysis/out/component_ablation_ladder.json','analysis/out/a3_hazard.json','out/verification/s43_nonarc_5domain_universality.json','.ai/blind/patch_autonomous_virgin236_ladder_and_theory/virgin236_ladder_and_theory_receipt.json']
    X,L,H,N,T=[json.loads((ROOT/p).read_text()) for p in inputs]
    from palette import TIERS
    from matplotlib import font_manager
    from palette import register_fonts
    register_fonts()
    INK='#0F172A'
    BLUE=TIERS['navy']; PURP=TIERS['indigo']; GOLD=TIERS['amber']; GREY=TIERS['gray']
    plt.rcParams.update({'font.family':'STIX Two Text','pdf.fonttype':42,'mathtext.fontset':'stix'})
    fig=plt.figure(figsize=(16,9),dpi=210,facecolor='white')
    ax=fig.add_axes([0,0,1,1]);ax.set(xlim=(0,16),ylim=(0,9));ax.axis('off')
    texts=[]
    def text(x,y,s,size=12,ha='center',color=INK):
     s=s.replace('≤',r'$\leq$').replace('→',r'$\to$')
     t=ax.text(x,y,s,fontsize=size,weight='bold',color=color,ha=ha,va='center',zorder=90);texts.append(t);return t
    def poly(v,c,edge=INK,lw=.9,z=3):
     p=Polygon(v,closed=True,facecolor=c,edgecolor=edge,linewidth=lw,joinstyle='round',zorder=z);ax.add_patch(p);return p
    def line(v,c=INK,lw=.9,z=2):
     v=np.asarray(v);return ax.plot(v[:,0],v[:,1],color=c,lw=lw,zorder=z,solid_joinstyle='round',solid_capstyle='round')[0]
    def cross2(a,b):return a[0]*b[1]-a[1]*b[0]
    def prism(v,pal=BLUE,depth=.03,z=4,face=1):
     v=np.asarray(v)
     poly(v,pal[face],edge=INK,lw=.85,z=z+.1)
    def badge(x,y,w,s,pal=BLUE,size=12,face=1):
     prism([(x-w/2,y-.18),(x+w/2,y-.18),(x+w/2,y+.18),(x-w/2,y+.18)],pal,z=65,face=face)
     text(x,y,s,size,color=INK if face>=3 else 'white')
    def gem(x,y,pal=BLUE,d=10,z=40):
     dx=d/144;dy=dx
     poly([(x-dx,y),(x,y+dy),(x+dx,y),(x,y-dy)],pal[2],edge=INK,lw=.8,z=z)
    def pipe(points,pal=BLUE,lw=12,z=25):
     line(points,pal[1],1.8,z)
    titles=['a  Frontier Lock & quadratic dilution','b  D4 equivariant world model & certificate gate','c  Generalization across 316 games & 30 Hidden-110 runs']
    for x,title in zip([.2,5.45,10.7],titles):
     ax.add_patch(Rectangle((x,.24),5.1,8.5,fc='white',ec='#D0D7DE',lw=.7,zorder=-5))
     text(x+.13,8.43,title,10.3,ha='left')
    # A real 64x64 frame is an affine parallelogram, preserving every pixel.
    arc=list(src.ARC); arc[14]=PURP[2]
    def board(frame,cx,cy,w,h,z=5):
     a=np.asarray(frame);n,m=a.shape
     origin=np.array([cx-w/2,cy]);u=np.array([w*.64,h*.50]);v=np.array([w*.36,-h*.50]);corn=np.array([origin,origin+u,origin+u+v,origin+v])
     verts=[];colors=[]
     for r in range(n):
      # merge contiguous identical pixels horizontally: exact pixels, smaller vector PDF
      c=0
      while c<m:
       end=c+1
       while end<m and a[r,end]==a[r,c]:end+=1
       q=origin+u*c/m+v*r/n;verts.append([q,q+u*(end-c)/m,q+u*(end-c)/m+v/n,q+v/n]);colors.append(arc[int(a[r,c])]);c=end
     ax.add_collection(PolyCollection(verts,facecolors=colors,edgecolors='none',antialiased=False,zorder=z+.1))
     poly(corn,'none',edge=INK,lw=.9,z=z+.2)
     return corn
    # Seven actual levels, unfolded vertically. Tracks are explanatory routes, not invented action coordinates.
    centers=[]
    for i in range(7):
     cx=2.10+i*.22;cy=4.64+i*.43;centers.append((cx,cy))
     board(X['specimen']['level_frames'][str(i+1)],cx,cy,2.9,.89,z=5+i*3)
     text(.45+i*.22,cy-.06,f'L{i+1}',10)
    pipe([(3.65,4.35),(4.40,4.67),(4.4,7.56)],BLUE,11,35)
    for cx,cy in centers:
     pipe([(cx+.98,cy),(4.4,cy+.10)],BLUE,5,35);gem(4.4,cy+.10,BLUE,9)
    poly([(4.22,7.55),(4.58,7.55),(4.4,7.88)],BLUE[1],z=40)
    badge(2.65,7.84,3.6,'GiantsPlanner:7/7·95act·0tok',BLUE,13)
    text(2.65,3.94,'People:317act · 33runs:L1–L2 · h(80)≤0.091',10)
    # True analytic score surface, several A slices, no fitted observations.
    text(2.70,3.58,r'$(b_l/(A+k+1))^2$',16)
    k=np.linspace(0,600,140)
    for j,(A,col) in enumerate([(30,BLUE[3]),(50,BLUE[2]),(95,BLUE[1])]):
     xx=.7+3.7*k/600+j*.13; yy=.95+j*.18+2.1*((A+1)/(A+k+1))**2
     base=.95+j*.18; z0=20-j*5
     poly(list(zip(xx,yy))+[(xx[-1],base),(xx[0],base)],col,edge=INK,lw=.9,z=z0)
     line([(xx[0],base),(xx[-1],base)],BLUE[0],2.0,z0+.5)
     line(list(zip(xx,yy)),INK,1.1,z0+1.)
     for kk in [8,64,224,600]:gem(.7+3.7*kk/600+j*.13,base+2.1*((A+1)/(A+kk+1))**2,BLUE,6,z=z0+2)
    # Hazard trajectory (mean across recorded strata); shared normalized 0..1 vertical scale.
    ks_sorted=sorted({int(n) for v in H['actions'].values() for n,r in v['cond_pass_next20'].items() if r[0] is not None})
    pts=[(.70+3.7*k0/600,.95+2.1*float(np.mean([v['cond_pass_next20'][str(k0)][0] for v in H['actions'].values() if str(k0) in v['cond_pass_next20'] and v['cond_pass_next20'][str(k0)][0] is not None]))) for k0 in ks_sorted]
    if pts:line(pts,GOLD[1],2.2,35)
    for kx,ky in pts:gem(kx,ky,GOLD,6,z=36)
    text(2.7,.58,'A=30/50/95 · k:0→600',11)
    # D4 crystal: group orbit of a real 9x9 L2 neighborhood around the first action in level 2.
    frame=np.array(X['specimen']['level_frames']['1']);tile=frame
    orbit=[tile,np.rot90(tile),np.rot90(tile,2),np.rot90(tile,3),np.fliplr(tile),np.flipud(tile),tile.T,np.flipud(np.fliplr(tile.T))]
    cx,cy=8,6.19
    ring=[(cx+1.36*np.cos(t),cy+1.17*np.sin(t)) for t in np.arange(9)*np.pi/4]
    line(ring,PURP[2],.9,5)
    for i,t in enumerate(np.arange(8)*np.pi/4):
     bx=cx+1.36*np.cos(t);by=cy+1.17*np.sin(t)
     pipe([(cx+.70*np.cos(t),cy+.60*np.sin(t)),(bx,by)],PURP,3.8,13)
     board(orbit[i],bx,by,.78,.54,18)
    after_frame=np.asarray(json.loads((ROOT/'analysis/out/xray_transition.json').read_text())['after'])
    diff_mask=(frame!=after_frame)
    rs,cs=np.where(diff_mask)
    r0,r1=max(0,int(rs.min())-6),min(64,int(rs.max())+7)
    c0,c1=max(0,int(cs.min())-6),min(64,int(cs.max())+7)
    delta_patch=after_frame.copy()
    delta_patch[r0:r1,c0:c0+2]=12;delta_patch[r0:r1,c1-2:c1]=12
    delta_patch[r0:r0+2,c0:c1]=12;delta_patch[r1-2:r1,c0:c1]=12
    board(delta_patch,8,6.19,1.05,.85,30)
    text(8.37,5.72,'ΔF',13)
    # Eight measured channel prisms arranged as an outside horseshoe with orthogonal bus routing.
    positions=[(6.10,7.65),(7.35,7.65),(8.60,7.65),(9.85,7.65),(6.10,6.68),(9.85,6.68),(6.10,5.55),(9.85,5.55)]
    for i,(x,y) in enumerate(positions,1):
     pts=[(x+.49*np.cos(t),y+.26*np.sin(t)) for t in np.arange(6)*np.pi/3]
     prism(pts,PURP,.07,40,face=1);text(x,y,f'C{i}:{X["census"]["routes"][f"C{i}"]["sources"]}',9.5,color='white')
     side=5.68 if x<8 else 10.32
     anchor=6.32 if x<8 else 9.68
     if y>7.0:
      line([(x,y+.26),(x,7.98),(side,7.98),(side,4.36),(anchor,4.36)],PURP[1],1.3,1)
     else:
      line([(x+(-0.49 if x<8 else 0.49),y),(side,y),(side,4.36),(anchor,4.36)],PURP[1],1.3,1)
    badge(8,4.36,3.35,'L2·D₄:79.6%recall·0/219false',PURP,13,face=1)
    pipe([(8,4.12),(8,3.65)],PURP,13)
    # Mechanical unilateral gate, thick open and closed branches.
    pipe([(6.1,2.70),(7.12,2.70)],GREY,15)
    prism([(7.12,2.7),(8,3.53),(8.88,2.7),(8,1.87)],PURP,.14,38,face=1)
    text(8,2.92,r'$\mathcal{C}(F_0)$',17,color='white')
    text(6.55,3.02,'F₀',14)
    text(8,2.46,r'$(\hat{T},\tau,\hat{G})$',13,color='white')
    pipe([(8,1.87),(8,.90)],BLUE,15,45)
    pipe([(8.88,2.70),(9.55,2.70)],GOLD,15,45)
    badge(8.0,.70,3.9,'Certified:183/183·0.413×·0tok',BLUE,14,face=1)
    text(9.72,2.95,'k=0',11)
    text(9.72,2.42,r'$(b_l/a_l)^2$',12)
    # Six suite bars. Unknown middle-stage values are not fabricated.
    badge(13.15,7.72,4.15,'1,793/1,794·99.94%',BLUE,17,face=1)
    suite_rows=[]
    for i,(label,total,base,full,_) in enumerate(src.SUITES+[('5Non-ARC',1250,1022,1183,'')]):
     y=7.13-i*.49;name=['Public-25','Param-25','Tufa-50','Clone-118','Held-Out98','5Non-ARC'][i]
     text(10.88,y+.03,name,10,ha='left')
     start=12.02;width=1.96
     # prior/inducer/full segments available only where measured; zero-width unknown middle segment.
     mid=148 if i==4 else (1183 if i==5 else base)
     b=80 if i==4 else base
     for lo,hi,pal,fc in [(0,b,GOLD,2),(b,mid,PURP,2),(mid,full,BLUE,1)]:
      if hi<=lo:continue
      x0=start+width*lo/total;x1=start+width*hi/total
      prism([(x0,y-.12),(x1,y-.12),(x1,y+.13),(x0,y+.13)],pal,.06,10,face=fc)
     text(14.68,y+.01,(f'80→148/{total:,}' if i==4 else f'{full:,}/{total:,}'),10.5)
     suite_rows.append({'suite':name,'total':total,'baseline':b,'inducer':mid if i>=4 else None,'full':full})
    text(13.24,4.10,'Baseline / +OnlineInducer / Full·C1..C8',10)
    # Paired predictions: all points from saved cell measurements, not jittered onto y=x.
    px,py,pw,ph=11.10,1.12,1.90,2.20
    line([(px,py+ph),(px,py),(px+pw,py)],INK,.9,2)
    poly([(px,py+.09),(px+pw-.08,py+ph),(px+pw,py+ph-.09),(px+.08,py)],BLUE[3],edge='none',z=1)
    line([(px,py),(px+pw,py+ph)],INK,.9,2)
    arc_cells=[r for r in T['dilution_law']['cells'] if r['suite'] not in ('Public-25','Virgin-236')]
    # ARC G range is 0..100; non-ARC linear score is 0..1; plotted in common unit interval.
    for r in arc_cells:gem(px+pw*r['law']/100,py+ph*r['observed']/100,BLUE,8)
    for r in N['prediction_cells']:
     ax.scatter([px+pw*r['predicted']['linear']],[py+ph*r['observed']['linear']],s=28,fc=PURP[2],ec=INK,lw=.8,zorder=40)
    text(12.05,3.66,'R²=0.978/0.997',12)
    text(12.05,.77,'Predicted→Observed',9.5)
    # Thirty-record inventory: all 28 numerical receipts plus two explicit error marks.
    rc=X['receipts'];base_refs=[56374698,56408484,56656982,56702094,56767431,56789456]
    base=[r for r in rc if r['ref'] in base_refs]
    probe=[r for r in rc if r['ref'] in [57854657,57894564]]
    pool=[r for r in rc if r['description'].startswith('Milestone 2') or r['ref'] in [57329763,57827241]]
    others=[r for r in rc if r not in base+probe+pool]
    line([(13.42,1.12),(15.39,1.12)],INK,.9)
    for j,(group,pal,fc) in enumerate([(base,GREY,2),(probe,GOLD,2),(pool,BLUE,2)]):
     x=13.65+j*.66;vals=[r['score'] for r in group];mean=float(np.mean(vals));top=1.12+mean/32*2.2
     prism([(x-.17,1.12),(x+.17,1.12),(x+.17,top),(x-.17,top)],pal,.055,5,face=fc)
     for i,r in enumerate(group):gem(x+((i%4)-1.5)*.065,1.12+r['score']/32*2.2,pal,7,35)
    text(14.31,3.66,'30·Hidden-110',12)
    text(14.31,.77,'2.81 / 20.76·21.08 / 24.83±2.12',8.6)
    text(14.31,.93,'vLLM / Probe / k=0',9)
    # Remaining six COMPLETE and two ERROR records displayed as an inventory rail.
    for i,r in enumerate(others):
     x=13.47+i*.25
     if r['score'] is None:
      line([(x-.035,.45),(x+.035,.52)],INK,.9,50);line([(x-.035,.52),(x+.035,.45)],INK,.9,50)
     else:gem(x,.48,GREY,5)
    text(14.27,3.32,'25.70·U=0·p=0.0083',10)
    # Mechanical gates are run on the actual Figure, without suppressing its text or outlines.
    hits=figcheck.check(fig,'cover_cel_16x9')
    labels=[t.get_text() for t in texts]
    whitespace_tokens=sum(len(s.split()) for s in labels)
    prose_labels=[re.sub(r'\$.*?\$', '', s) for s in labels]
    words=sum(len(re.findall(r'(?<![\w])(?:[A-Za-z]{2,})(?:-[A-Za-z]+)*(?![\w])',s)) for s in prose_labels)
    fig.savefig(output_dir/'cover.png',dpi=210,facecolor='white')
    fig.savefig(output_dir/'cover.pdf',facecolor='white')
    im=Image.open(output_dir/'cover.png').convert('RGB');rgb=np.asarray(im,dtype=np.float32)/255;hsv=rgb_to_hsv(rgb);green=int(((hsv[:,:,0]*360>=70)&(hsv[:,:,0]*360<=165)&(hsv[:,:,1]>=.18)&(hsv[:,:,2]>=.18)).sum())
    receipt={'state':'done' if not hits and green==0 and words<=50 else 'needs_revision','size':list(im.size),'dpi':210,'figcheck_hits':len(hits),'figcheck_details':hits,'green_px':green,'pixels':im.width*im.height,'word_count_whitespace':whitespace_tokens,'prose_word_count':words,'light_gray_text_count':sum(1 for t in texts if (lambda c:max(c)-min(c)<.12 and .55<sum(c)/3<.97)(to_rgb(t.get_color()))),'labels':labels,'titles_single_line':all('\n' not in s for s in titles),'sources':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in inputs},'suite_rows':suite_rows,'arc_prediction_cells':len(arc_cells),'nonarc_prediction_cells':len(N['prediction_cells']),'receipt_statistics':{'baseline_mean':float(np.mean([r['score'] for r in base])),'baseline_sd':float(np.std([r['score'] for r in base],ddof=1)),'probe_scores':[r['score'] for r in probe],'pool_mean':float(np.mean([r['score'] for r in pool])),'pool_sd':float(np.std([r['score'] for r in pool],ddof=1))},'receipt_groups':{'baseline_refs':[r['ref'] for r in base],'probe_refs':[r['ref'] for r in probe],'pool_refs':[r['ref'] for r in pool],'remaining_refs':[r['ref'] for r in others]},'data_notes':['Board geometry preserves all 64x64 cells; ARC color 14 is remapped to indigo for color-blind safety. Routes show level order, not action coordinates.','D4 tiles are eight exact 64x64 transforms of the real tn36 L1 frame.','No unmeasured inducer segment is drawn for the first four suites.','Total 1793/1794 excludes non-ARC; Held-Out98 full suite and zero-prior induction are distinct conditions.','Parity band is a visual tolerance guide, not a statistical confidence interval.','Hazard traces use all stored dataset strata; dilution surfaces use illustrative A=30,50,95.','30 records include 2 errors; main groups have 6,2,14 numerical records; remaining rail shows the other 6 numerical and 2 error records.'],'outputs':{ext:hashlib.sha256((output_dir/f'cover.{ext}').read_bytes()).hexdigest() for ext in ['png','pdf']}}
    report['cover']={**receipt,'hits':hits,'red_px':src.hue_audit(output_dir/'cover.png')['red_px'],'pdf_bytes':(output_dir/'cover.pdf').stat().st_size}
    assert not hits and green==0 and words<=50, report['cover']
    print(json.dumps({k:receipt[k] for k in ['state','size','figcheck_hits','green_px','word_count_whitespace','arc_prediction_cells','nonarc_prediction_cells']}));print('\n'.join(hits))

if __name__ == "__main__":
    report={}
    render_cover(ROOT/"paper/figures",report)
    (ROOT/".ai/dispatch/cover_cel_16x9_receipt.json").write_text(json.dumps(report["cover"],indent=2))
