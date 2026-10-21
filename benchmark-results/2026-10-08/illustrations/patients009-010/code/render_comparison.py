"""Animate authenticated matched recordings; no new simulation or glucose interpolation.

The original user template is retained in the earlier animation bundle. This
adaptation shares the clock and axes across no shield, static and predictive.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from PIL import Image

MODES = ('none', 'static', 'predictive')
LABELS = ('No shield', 'Static guard', 'Predictive shield')
COLORS = ('#BC433C', '#717D89', '#087F7D')

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read(path):
    return json.loads(Path(path).read_text())

def require(value, message):
    if not value:
        raise ValueError(message)

def clock(minute, days):
    m = int(minute)
    return f'{m // 1440}d {m % 1440 // 60:02d}:{m % 60:02d}' if days > 1 else f'{m // 60:02d}:{m % 60:02d}'

def render(args):
    root, out = args.episode.resolve(), args.output.resolve()
    meta = [read(root / mode / 'metadata.json') for mode in MODES]
    rows = [read(root / mode / 'rollout.json') for mode in MODES]
    identity = ('algorithm', 'patient_type', 'patient_name', 'policy_seed', 'action_mode', 'env_seed', 'horizon_days')
    require(all(all(m[k] == meta[0][k] for k in identity) for m in meta), 'Episode identity differs')
    require(all(m['condition'] == mode for m, mode in zip(meta, MODES)), 'Condition mismatch')
    require(meta[0]['algorithm'] == 'PPOLag' and meta[0]['patient_type'] == 't2d_no_pump', 'Unexpected policy/cohort')
    require(all(m['source']['checkpoint_sha256'] == meta[0]['source']['checkpoint_sha256'] for m in meta), 'Checkpoint mismatch')
    require(all(m['source']['config_sha256'] == meta[0]['source']['config_sha256'] for m in meta), 'Policy configuration mismatch')
    require(all(r for r in rows), 'Empty recording')
    require(meta[2]['absolute_penalty'] == 5, 'Selected deterministic penalty must be5')
    require(all(m['action_mode'] == 'deterministic' for m in meta), 'Expected deterministic episode')
    initial = [r[0]['pre_observation'] for r in rows]
    require(initial[0] == initial[1] == initial[2], 'Initial observations differ')
    days = meta[0]['horizon_days']
    max_steps = days * 288
    require(all(len(r) == max_steps for r in rows), 'This illustration requires complete matched horizons')
    require(all(r[-1]['truncated'] and not any(s['terminated'] for s in r) for r in rows), 'Incomplete or terminated episode')
    for recording in rows:
        for i, row in enumerate(recording):
            require(row['sample_index'] == i and row['decision_minute'] == i * 5 and row['observation_minute'] == (i + 1) * 5, 'Unexpected sample clock')
            require(math.isfinite(row['cgm']), 'Nonfinite CGM')
            require(row['executed_action'] == [a if accept else 0 for a, accept in zip(row['action'], (row['bolus_accepted'], row['meal_accepted']))], 'Acceptance mismatch')
    require(any(a['cgm'] != b['cgm'] for a,b in zip(rows[0], rows[2])), 'Identical trajectories do not demonstrate a difference')
    require(any(a['executed_action'] != b['executed_action'] for a,b in zip(rows[0],rows[2])), 'No accepted action difference')
    require(any(r.get('prediction_recommendation_changed', False) for r in rows[2]), 'No forecast-driven recommendation change')
    out.mkdir(parents=True, exist_ok=False)
    times = np.arange(max_steps + 1) * 5 / 60
    glucose = [np.array([r[0]['pre_cgm']] + [s['cgm'] for s in r]) for r in rows]
    low = min(g.min() for g in glucose)
    high = max(g.max() for g in glucose)
    ymin = min(50, math.floor((low - 8) / 10) * 10)
    ymax = max(205, math.ceil((high + 8) / 10) * 10)
    requested_hours = days * 24
    # Every request event in any arm is retained, with at most10min between frames.
    indices = sorted({0, max_steps} | set(range(0,max_steps+1,2)) | {i+1 for r in rows for i,s in enumerate(r) if any(s['action'])})
    differences = [i for i,(a,b) in enumerate(zip(rows[0], rows[2])) if a['executed_action'] != b['executed_action']]
    first_difference = differences[0]
    indices = sorted(set(indices) | {first_difference + 1})
    durations = [100] * len(indices)
    durations[indices.index(first_difference + 1)] = 1400
    durations[-1] = 6000
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.spines.top':False,'axes.spines.right':False})
    fig = plt.figure(figsize=(12.8,7.2),dpi=100,facecolor='white')
    ink, muted = '#17212B', '#59636E'
    fig.text(.035,.951,'Same patient, different glucose response',fontsize=25,weight='bold',color=ink)
    fig.text(.035,.908,f'PPO-Lag / T2D no pump / {meta[0]["patient_name"]} / {days}-day recorded evaluation',fontsize=13,color=muted)
    time_text = fig.text(.97,.95,'00:00',ha='right',fontsize=18,color=ink)
    latest = []
    for x,label,color in zip((.045,.36,.675), LABELS,COLORS):
        fig.text(x,.853,label,fontsize=14,weight='bold',color=color)
        latest.append(fig.text(x,.808,'No accepted request yet',fontsize=10,color=ink))
    ax = fig.add_axes([.12,.365,.84,.365])
    ax.set_xlim(0,requested_hours)
    ax.set_ylim(ymin,ymax)
    ax.axhspan(70,180,color='#EDF5EF',zorder=0)
    for y in (70,180):
        ax.axhline(y,color='#AEC8B4',lw=1,ls='--')
    ax.set_ylabel('Recorded CGM (mg/dL)',color=ink)
    ax.tick_params(axis='x',labelbottom=False)
    ax.set_yticks(sorted(set([70,120,180] + ([250] if ymax>250 else []) + ([40] if ymin<40 else []))))
    ax.grid(axis='x',color='#EEF1F4')
    traces = [ax.plot([],[],color=c,lw=2.1,ls='--' if mode=='static' else '-')[0] for mode,c in zip(MODES,COLORS)]
    dots = [ax.plot([],[],'o',ms=4,color=c)[0] for c in COLORS]
    cursor = ax.axvline(0,color='#17212B',lw=.8,alpha=.55)
    ev = fig.add_axes([.12,.16,.84,.17],sharex=ax)
    ev.set_ylim(-.6,5.6)
    ev.set_yticks([5,4,3,2,1,0],['None: bolus','None: meal','Static: bolus','Static: meal','Predictive: bolus','Predictive: meal'])
    ev.tick_params(axis='y',length=0,labelsize=9)
    ev.spines['left'].set_visible(False)
    for y in range(6):
        ev.axhline(y,color='#E8EDF1',lw=.8,zorder=0)
    if days == 1:
        ev.set_xticks(np.arange(0,25,4))
        ev.set_xlabel('Elapsed hours',fontsize=10)
    else:
        ev.set_xticks(np.arange(0,days*24+1,24), [str(i) for i in range(days+1)])
        ev.set_xlabel('Elapsed days',fontsize=10)
    events = [[ev.scatter([],[],s=24,marker=marker,color=color,zorder=3) for marker in ('o','^')] for color in COLORS]
    event_cursor = ev.axvline(0,color=ink,lw=.8,alpha=.55)
    fig.text(.035,.082,'Accepted recommendations:  ● bolus  ▲ meal. Scheduled meals and physical delivered dose were not archived.',fontsize=9,color=muted)
    epoch = meta[0].get('policy_epoch',meta[0].get('epoch'))
    fig.text(.035,.054,f'Policy seed {meta[0]["policy_seed"]} / epoch {epoch} / reset {meta[0]["env_seed"]} / deterministic / model {meta[2]["model_seed"]} / absolute penalty5',fontsize=9,color=muted)
    fig.text(.035,.025,'Benefit-selected illustration, not aggregate evidence. Actual recorded samples; no smoothing or glucose interpolation.',fontsize=9,color=muted)
    statistics = []
    for r,g in zip(rows,glucose):
        values = g[1:]
        statistics.append({'tir_percent':float(np.mean((values>=70)&(values<=180))*100),'below54_percent':float(np.mean(values<54)*100),
                           'sample_count':len(r),'coverage_percent':100.,'early_termination':False})
    def update(k):
        time_text.set_text(clock(k*5,days))
        for j,(recording,g,line,dot) in enumerate(zip(rows,glucose,traces,dots)):
            line.set_data(times[:k+1],g[:k+1])
            dot.set_data([times[k]],[g[k]])
            accepted = [(i,r) for i,r in enumerate(recording[:k]) if any(r['executed_action'])]
            if accepted:
                i,r = accepted[-1]
                parts = [f'{name} level {level}' for name,level in zip(('Bolus','Meal'),r['executed_action']) if level]
                latest[j].set_text(' / '.join(parts)+'\n'+clock(r['decision_minute'],days))
            else:
                latest[j].set_text('No accepted request yet')
            if k == max_steps:
                s = statistics[j]
                latest[j].set_text(f'TIR {s["tir_percent"]:.2f}% / below54 {s["below54_percent"]:.2f}%\nComplete {days}-day episode')
            for dim,artist in enumerate(events[j]):
                points = [(r['decision_minute']/60, 5-j*2-dim) for r in recording[:k] if r['executed_action'][dim]>0]
                artist.set_offsets(np.array(points).reshape(-1,2))
        cursor.set_xdata([times[k]]*2)
        event_cursor.set_xdata([times[k]]*2)
        fig.canvas.draw()
    ffmpeg = shutil.which('ffmpeg')
    if ffmpeg is None:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    matplotlib.rcParams['animation.ffmpeg_path'] = ffmpeg
    gif_frames = []
    writer = FFMpegWriter(fps=10,codec='libx264',extra_args=['-pix_fmt','yuv420p','-crf','20','-threads','2'])
    video_frames = 0
    with writer.saving(fig,str(out/'shield_comparison.mp4'),100):
        for k,ms in zip(indices,durations):
            update(k)
            rgb = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy())
            gif_frames.append(rgb.quantize(colors=256,dither=Image.Dither.NONE))
            repeats = ms // 100
            for _ in range(repeats):
                writer.grab_frame()
            video_frames += repeats
            if k == max_steps:
                rgb.save(out/'shield_comparison.png')
            if k == first_difference+1:
                rgb.save(out/'first_accepted_difference.png')
    gif_frames[0].save(out/'shield_comparison.gif',save_all=True,append_images=gif_frames[1:],duration=durations,loop=0,optimize=False)
    receipt = {'renderer_sha256':sha(__file__),'inputs_sha256':{str(root/m/name):sha(root/m/name) for m in MODES for name in ('metadata.json','rollout.json')},
               'identity':{k:meta[0][k] for k in identity},'indices':indices,'durations_ms':durations,'gif_frames':len(indices),
               'gif_duration_seconds':sum(durations)/1000,'mp4_frames':video_frames,'mp4_duration_seconds':video_frames/10,
               'full_horizon':True,'new_simulation':False,'glucose_interpolation':False,'all_request_event_frames_included':True,
               'glucose_y_range':[ymin,ymax],'observed_glucose_range':[float(low),float(high)],'statistics':dict(zip(MODES,statistics)),
               'first_accepted_difference_minute':first_difference*5,'predictive_vs_none_different_accepted_steps':len(differences),
               'forecast_changed_recommendation_steps':sum(r.get('prediction_recommendation_changed',False) for r in rows[2]),
               'outputs_sha256':{name:sha(out/name) for name in ('shield_comparison.gif','shield_comparison.mp4','shield_comparison.png','first_accepted_difference.png')}}
    (out/'render_receipt.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
    plt.close(fig)
    print(json.dumps({k:receipt[k] for k in ('identity','statistics','gif_frames','first_accepted_difference_minute')},indent=2))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episode',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    render(parser.parse_args())
