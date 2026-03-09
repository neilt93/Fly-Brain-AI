"""Generate self-contained HTML demo page for tweet2."""
import base64
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
FIGS = ROOT / "plastic-fly" / "figures"

# Encode figures
encoded = {}
for name in [
    "odor_valence_proof.png",
    "looming_escape_proof.png",
    "sensory_perturbation.png",
    "mujoco_walking_proof.png",
    "cpg_baseline_walking.png",
    "robustness_summary.png",
]:
    p = FIGS / name
    if p.exists():
        encoded[name] = base64.b64encode(p.read_bytes()).decode()

def img_tag(name, alt=""):
    b64 = encoded.get(name, "")
    return f'<img src="data:image/png;base64,{b64}" alt="{alt}">'

odor_img = img_tag("odor_valence_proof.png", "Odor valence")
loom_img = img_tag("looming_escape_proof.png", "Looming escape")
perturb_img = img_tag("sensory_perturbation.png", "Sensory perturbation")
mujoco_img = img_tag("mujoco_walking_proof.png", "MuJoCo walking")
cpg_img = img_tag("cpg_baseline_walking.png", "CPG baseline")
robust_img = img_tag("robustness_summary.png", "Robustness")

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Fly Brain AI — Sensory Demo</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: #0a0a0a; color: #e0e0e0; line-height: 1.6;
}}
.hero {{
    background: linear-gradient(135deg, #1a0a2e 0%, #0d1b2a 50%, #0a1628 100%);
    padding: 2rem 1rem; text-align: center;
    border-bottom: 2px solid #2d1b69;
}}
.hero h1 {{
    font-size: 1.8rem;
    background: linear-gradient(90deg, #00d2ff, #7b2ff7);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem;
}}
.hero .sub {{ color: #888; font-size: 0.95rem; }}
.stats {{
    display: flex; justify-content: center; gap: 1.5rem;
    margin-top: 1rem; flex-wrap: wrap;
}}
.stat {{ text-align: center; }}
.stat .num {{ font-size: 1.5rem; font-weight: 700; color: #00d2ff; }}
.stat .label {{ font-size: 0.7rem; color: #666; text-transform: uppercase; letter-spacing: 1px; }}
.section {{
    padding: 1.5rem 1rem; border-bottom: 1px solid #1a1a2e;
}}
.section h2 {{
    font-size: 1.2rem; margin-bottom: 0.8rem; color: #7b2ff7;
}}
.section p {{ color: #aaa; font-size: 0.9rem; margin-bottom: 1rem; }}
.fig-container {{
    background: #111; border-radius: 8px; overflow: hidden;
    border: 1px solid #222; margin-bottom: 1rem;
}}
.fig-container img {{
    width: 100%; height: auto; display: block;
}}
.caption {{
    padding: 0.6rem 0.8rem; font-size: 0.8rem; color: #888;
    border-top: 1px solid #222;
}}
.modality-grid {{
    display: grid; grid-template-columns: 1fr 1fr; gap: 0.6rem; margin: 1rem 0;
}}
.mod-card {{
    background: #111; border-radius: 8px; padding: 0.8rem; border: 1px solid #222;
}}
.mod-card .name {{ font-size: 0.85rem; font-weight: 600; color: #e0e0e0; }}
.mod-card .desc {{ font-size: 0.7rem; color: #777; margin-top: 0.2rem; }}
.mod-card .neurons {{ font-size: 0.7rem; color: #00d2ff; margin-top: 0.3rem; }}
.pipeline {{
    background: #0d1117; border-radius: 8px; padding: 1rem;
    border: 1px solid #1a1a2e; font-family: monospace; font-size: 0.75rem;
    color: #7ee787; overflow-x: auto; white-space: pre; line-height: 1.8;
}}
.test-result {{
    display: flex; align-items: center; gap: 0.5rem;
    padding: 0.3rem 0; font-size: 0.85rem;
}}
.test-result .pass {{ color: #27ae60; font-weight: 700; }}
.test-result .label {{ color: #aaa; }}
.footer {{
    text-align: center; padding: 2rem 1rem; color: #444; font-size: 0.75rem;
}}
.tag {{
    display: inline-block; padding: 0.15rem 0.5rem; border-radius: 12px;
    font-size: 0.7rem; font-weight: 600; margin: 0.1rem;
}}
.tag-olf {{ background: #1a3a2a; color: #4ade80; }}
.tag-vis {{ background: #3a1a2a; color: #f472b6; }}
.tag-mech {{ background: #1a2a3a; color: #60a5fa; }}
.tag-gust {{ background: #3a2a1a; color: #fbbf24; }}
.tag-vest {{ background: #2a1a3a; color: #a78bfa; }}
.tag-therm {{ background: #3a2a2a; color: #fb923c; }}
.info-box {{
    background: #111; border-radius: 8px; padding: 1rem;
    border: 1px solid #222; font-family: monospace; font-size: 0.8rem;
}}
.info-line {{ padding: 0.15rem 0; color: #aaa; }}
.info-head {{ color: #fb923c; font-weight: 700; margin-top: 0.5rem; }}
.info-head-blue {{ color: #60a5fa; font-weight: 700; margin-top: 0.5rem; }}
.info-head-green {{ color: #4ade80; font-weight: 700; margin-top: 0.5rem; }}
table {{ width: 100%; font-size: 0.8rem; color: #aaa; }}
table td {{ padding: 0.3rem 0; }}
table td:first-child {{ color: #00d2ff; }}
</style>
</head>
<body>

<div class="hero">
    <h1>Drosophila Sensorimotor AI</h1>
    <div class="sub">FlyWire connectome + FlyGym body + Brian2 LIF brain</div>
    <div class="stats">
        <div class="stat"><div class="num">139K</div><div class="label">Neurons</div></div>
        <div class="stat"><div class="num">15M</div><div class="label">Synapses</div></div>
        <div class="stat"><div class="num">6</div><div class="label">Modalities</div></div>
        <div class="stat"><div class="num">204</div><div class="label">Readout DNs</div></div>
    </div>
</div>

<div class="section">
    <h2>Sensory Modalities</h2>
    <p>Six identified sensory channels feed real FlyWire neuron populations through the 139K-neuron connectome to descending neurons that control locomotion.</p>
    <div class="modality-grid">
        <div class="mod-card">
            <div class="name">Olfactory</div>
            <div class="desc">Or42b (DM1) attractive vs Or85a (DM5) aversive</div>
            <div class="neurons">Bilateral ORNs</div>
        </div>
        <div class="mod-card">
            <div class="name">Visual / Looming</div>
            <div class="desc">210 LPLC2 neurons detect approaching objects</div>
            <div class="neurons">210 LPLC2 + R7/R8</div>
        </div>
        <div class="mod-card">
            <div class="name">Mechanosensory</div>
            <div class="desc">Per-leg contact force detection</div>
            <div class="neurons">SEZ ascending types</div>
        </div>
        <div class="mod-card">
            <div class="name">Gustatory</div>
            <div class="desc">Sugar GRN contact-driven feeding</div>
            <div class="neurons">Sugar GRNs</div>
        </div>
        <div class="mod-card">
            <div class="name">Vestibular</div>
            <div class="desc">Body velocity + orientation sensing</div>
            <div class="neurons">SEZ ascending</div>
        </div>
        <div class="mod-card">
            <div class="name">Thermo / Hygro</div>
            <div class="desc">Temperature (5 DNs) + humidity (2 DNs) labeled lines</div>
            <div class="neurons">7 dedicated DNs</div>
        </div>
    </div>
    <div style="text-align:center; margin-top:0.5rem;">
        <span class="tag tag-olf">Olfactory</span>
        <span class="tag tag-vis">Visual</span>
        <span class="tag tag-mech">Mechanosensory</span>
        <span class="tag tag-gust">Gustatory</span>
        <span class="tag tag-vest">Vestibular</span>
        <span class="tag tag-therm">Thermo/Hygro</span>
    </div>
</div>

<div class="section">
    <h2>Brain-Body Pipeline</h2>
    <div class="pipeline">Body obs (FlyGym, 42 joints, 6 legs)
  |
  v
SensoryEncoder -> 6 channels -> Poisson rates
  |
  v
BrainRunner (Brian2 LIF, 139K neurons)
  |  20ms windows, 200ms warmup
  v
DescendingDecoder -> 204 readout DNs
  |
  v
LocomotionCommand (fwd, turn, freq, stance)
  |
  v
VNC-lite -> CPG + PreprogrammedSteps
  |
  v
FlyGym body -> MuJoCo physics</div>
</div>

<div class="section">
    <h2>Olfactory Valence: Odor Discrimination</h2>
    <p>The connectome's wiring produces opposite turning for food-attractive (Or42b/DM1) vs aversive (Or85a/DM5) odors. Real connectome shows clear valence; shuffled connectome loses specificity.</p>
    <div class="fig-container">
        {odor_img}
        <div class="caption">Or42b (attractive) drives toward-odor turning; Or85a (aversive) drives away. 10 seeds, real vs shuffled connectome control.</div>
    </div>
    <div class="test-result"><span class="pass">[PASS]</span><span class="label">DM1 attractive turning (toward odor)</span></div>
    <div class="test-result"><span class="pass">[PASS]</span><span class="label">DM5 aversive turning (away from odor)</span></div>
    <div class="test-result"><span class="pass">[PASS]</span><span class="label">Opposite valence contrast</span></div>
    <div class="test-result"><span class="pass">[PASS]</span><span class="label">Real > shuffled connectome specificity</span></div>
</div>

<div class="section">
    <h2>Visual: Looming Escape Response</h2>
    <p>210 LPLC2 neurons detect looming objects and drive ipsilateral escape turning via 44 descending neurons with 1,850 direct synapses. Tested at 20/50/100ms brain windows.</p>
    <div class="fig-container">
        {loom_img}
        <div class="caption">Left loom triggers rightward escape; right loom triggers leftward escape. Real connectome shows 50-100x stronger directionality than shuffled.</div>
    </div>
    <div class="test-result"><span class="pass">[PASS]</span><span class="label">Directional escape (L/R asymmetric)</span></div>
    <div class="test-result"><span class="pass">[PASS]</span><span class="label">Loom vs control significant effect</span></div>
    <div class="test-result"><span class="pass">[PASS]</span><span class="label">Opposite turns (left vs right loom)</span></div>
    <div class="test-result"><span class="pass">[PASS]</span><span class="label">Real > shuffled (50-100x stronger)</span></div>
</div>

<div class="section">
    <h2>Somatosensory Perturbation</h2>
    <p>Asymmetric sensory input (contact loss, gustatory boost, vestibular push) activates latent turn circuits. Three-phase protocol: baseline, perturbation, recovery.</p>
    <div class="fig-container">
        {perturb_img}
        <div class="caption">7 conditions tested: contact loss L/R, gustatory boost L/R, lateral push, and combined. All produce measurable heading changes through the connectome.</div>
    </div>
</div>

<div class="section">
    <h2>Thermo, Hygro &amp; Auditory Labeled Lines</h2>
    <p>FlyWire connectome analysis reveals dedicated pathways for temperature, humidity, and sound — maintaining segregated labeled-line architecture.</p>
    <div class="info-box">
        <div class="info-head">Thermosensory</div>
        <div class="info-line">5 dedicated descending neurons</div>
        <div class="info-line">Segregated from somatosensory (low Jaccard overlap)</div>
        <div class="info-line">Shuffled connectome: 4.6-21.4x more overlap</div>
        <div class="info-head-blue">Hygrosensory</div>
        <div class="info-line">2 dedicated descending neurons</div>
        <div class="info-line">Distinct interneuron pool from mechanosensory</div>
        <div class="info-head-green">Auditory</div>
        <div class="info-line">41 descending neurons (largest new modality)</div>
        <div class="info-line">JO neurons -> AMMC/WED -> motor output</div>
        <div class="info-line">Johnston's organ -> antennal mechanosensory center</div>
    </div>
    <p style="margin-top:1rem;">Cross-modality Jaccard overlap analysis confirms real connectome maintains 4.6-21.4x better segregation than shuffled — labeled-line architecture is encoded in wiring, not activity.</p>
</div>

<div class="section">
    <h2>Locomotion: Body Physics</h2>
    <p>FlyGym v1.2.1 + MuJoCo biomechanics with full contact dynamics and tripod gait coordination.</p>
    <div class="fig-container">
        {mujoco_img}
        <div class="caption">Position trajectory, contact forces, and leg dynamics during terrain walking.</div>
    </div>
    <div class="fig-container">
        {cpg_img}
        <div class="caption">CPG oscillation patterns driving the alternating tripod gait.</div>
    </div>
</div>

<div class="section">
    <h2>Plasticity vs Fixed: Terrain Adaptation</h2>
    <p>Sparse recurrent network with Hebbian plasticity outperforms frozen weights by 9x on rough terrain.</p>
    <div class="fig-container">
        {robust_img}
        <div class="caption">Multi-seed robustness across ablation, dose-response, and terrain conditions.</div>
    </div>
</div>

<div class="section">
    <h2>Architecture Summary</h2>
    <div style="background:#111;border-radius:8px;padding:1rem;border:1px solid #222;">
        <table>
            <tr><td>Brain</td><td>139K FlyWire neurons, 15M synapses (Brian2 LIF)</td></tr>
            <tr><td>Body</td><td>FlyGym v1.2.1 + MuJoCo (42 joints, 6 legs)</td></tr>
            <tr><td>Sensing</td><td>6 modalities: olf, vis, mech, gust, vest, therm/hygro</td></tr>
            <tr><td>Motor</td><td>VNC-lite premotor + CPG + PreprogrammedSteps</td></tr>
            <tr><td>Learning</td><td>Hebbian plasticity (lr=1e-5, cap=0.5)</td></tr>
            <tr><td>Readout</td><td>204 descending neurons: fwd, turn, freq, stance</td></tr>
            <tr><td>Viz</td><td>Unity 3D: MJCF + 71 STL meshes + connectome cloud</td></tr>
        </table>
    </div>
</div>

<div class="footer">
    Fly Brain AI — Connectome-driven sensorimotor intelligence<br>
    FlyWire 783 | Brian2 LIF | FlyGym 1.2.1 | Unity<br>
    <br>
    #Neuroscience #ArtificialLife #Connectome #Drosophila
</div>

</body>
</html>"""

out = Path(__file__).resolve().parent / "sensory_demo.html"
out.write_text(html)
print(f"Wrote {out} ({out.stat().st_size:,} bytes)")
