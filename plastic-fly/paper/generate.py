"""
Generate a submission-quality academic paper PDF using fpdf2.
Two-column layout, embedded figures, proper references.

Usage:
    python paper/generate.py
"""
from pathlib import Path
from fpdf import FPDF
from fpdf.enums import XPos, YPos

ROOT = Path(__file__).resolve().parent.parent
PAPER_DIR = Path(__file__).resolve().parent
FIG_DIR = ROOT / "figures"


class Paper(FPDF):
    """Academic paper with two-column layout support."""

    def __init__(self):
        super().__init__(format="letter")
        self.set_auto_page_break(auto=True, margin=20)
        # Load TTF fonts with Unicode support (use "tnr" to avoid conflict with core "tnr")
        self.add_font("tnr", style="", fname="c:/windows/fonts/times.ttf")
        self.add_font("tnr", style="B", fname="c:/windows/fonts/timesbd.ttf")
        self.add_font("tnr", style="I", fname="c:/windows/fonts/timesi.ttf")
        self.add_font("tnr", style="BI", fname="c:/windows/fonts/timesbi.ttf")

    def footer(self):
        self.set_y(-15)
        self.set_font("tnr", "", 9)
        self.set_text_color(128)
        self.cell(0, 10, str(self.page_no()), align="C", new_x=XPos.RIGHT, new_y=YPos.TOP)

    # ── Layout helpers ───────────────────────────────────────────────
    def section_heading(self, num, title):
        """Full-width section heading."""
        self.set_font("tnr", "B", 12)
        self.set_text_color(0)
        self.ln(4)
        self.cell(0, 7, f"{num}. {title}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(2)

    def subsection_heading(self, num, title):
        """Subsection heading within column."""
        self.set_font("tnr", "B", 10)
        self.set_text_color(0)
        self.ln(2)
        self.multi_cell(0, 5, f"{num} {title}")
        self.ln(1)

    def body_text(self, text, indent=True):
        """Body paragraph in serif font."""
        self.set_font("tnr", "", 10)
        self.set_text_color(0)
        if indent:
            # Manual first-line indent
            self.cell(8, 5, "", new_x=XPos.END, new_y=YPos.TOP)
            w = self.w - self.l_margin - self.r_margin - 8
            self.multi_cell(w, 5, text.strip())
        else:
            self.multi_cell(0, 5, text.strip())
        self.ln(1)

    def bold_start_para(self, bold_part, rest):
        """Paragraph starting with bold text, then normal."""
        self.set_font("tnr", "B", 10)
        self.set_text_color(0)
        self.write(5, bold_part)
        self.set_font("tnr", "", 10)
        self.write(5, " " + rest.strip())
        self.ln(6)

    def italic_start_para(self, italic_part, rest):
        """Paragraph starting with italic text, then normal."""
        self.set_font("tnr", "I", 10)
        self.write(5, italic_part)
        self.set_font("tnr", "", 10)
        self.write(5, " " + rest.strip())
        self.ln(6)

    def add_table(self, headers, rows, col_widths=None):
        """Simple table with borders."""
        self.set_font("tnr", "B", 8.5)
        self.set_text_color(0)
        total_w = self.w - self.l_margin - self.r_margin
        if col_widths is None:
            col_widths = [total_w / len(headers)] * len(headers)

        # Top border
        x0 = self.get_x()
        self.set_draw_color(0)
        self.set_line_width(0.4)
        self.line(x0, self.get_y(), x0 + sum(col_widths), self.get_y())
        self.ln(1)

        # Headers
        for i, h in enumerate(headers):
            self.cell(col_widths[i], 5, h, new_x=XPos.END, new_y=YPos.TOP)
        self.ln(5)

        # Header bottom border
        self.set_line_width(0.2)
        self.line(x0, self.get_y(), x0 + sum(col_widths), self.get_y())
        self.ln(1)

        # Rows
        self.set_font("tnr", "", 8.5)
        for row in rows:
            y_start = self.get_y()
            max_h = 5
            for i, cell_text in enumerate(row):
                self.set_xy(x0 + sum(col_widths[:i]), y_start)
                self.multi_cell(col_widths[i], 5, str(cell_text), new_x=XPos.END, new_y=YPos.TOP)
                max_h = max(max_h, self.get_y() - y_start)
            self.set_y(y_start + max_h)

        # Bottom border
        self.set_line_width(0.4)
        self.line(x0, self.get_y(), x0 + sum(col_widths), self.get_y())
        self.ln(3)

    def add_figure(self, img_path, caption, fig_num, width=None):
        """Full-width figure with caption."""
        if not img_path.exists():
            self.body_text(f"[Figure {fig_num}: {img_path.name} not found]", indent=False)
            return
        if width is None:
            width = self.w - self.l_margin - self.r_margin
        x = self.l_margin + (self.w - self.l_margin - self.r_margin - width) / 2
        self.image(str(img_path), x=x, w=width)
        self.ln(2)
        self.set_font("tnr", "B", 9)
        self.write(4.5, f"Figure {fig_num}. ")
        self.set_font("tnr", "", 9)
        self.multi_cell(0, 4.5, caption)
        self.ln(3)


def build_paper():
    pdf = Paper()

    # ══════════════════════════════════════════════════════════════════
    #  TITLE PAGE
    # ══════════════════════════════════════════════════════════════════
    pdf.add_page()
    pdf.ln(30)
    pdf.set_font("tnr", "B", 18)
    pdf.multi_cell(0, 10,
        "Connectome-Constrained Sensorimotor Behaviors and\n"
        "Modality-Specific Motor Channels in Drosophila",
        align="C")
    pdf.ln(10)
    pdf.set_font("tnr", "", 12)
    pdf.cell(0, 7, "Neil Tripathi", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(4)
    pdf.set_font("tnr", "I", 10)
    pdf.set_text_color(100)
    pdf.cell(0, 6, "Draft \u2014 March 2026", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_text_color(0)

    # ══════════════════════════════════════════════════════════════════
    #  ABSTRACT
    # ══════════════════════════════════════════════════════════════════
    pdf.ln(15)
    pdf.set_font("tnr", "B", 11)
    pdf.cell(0, 6, "ABSTRACT", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(3)

    abstract = (
        "The Drosophila melanogaster FlyWire connectome provides a neuron-level wiring "
        "diagram of 139,255 neurons and approximately 50 million synapses, but whether "
        "connectome-structured dynamics can support meaningful sensorimotor transformations "
        "without learning or task-specific optimization remains unknown. We built a "
        "closed-loop system coupling a Brian2 leaky integrate-and-fire simulation of "
        "138,639 neurons (FlyWire 783 completeness snapshot; 15 million connection pairs, "
        "54.5 million synapses) to a MuJoCo biomechanical fly body (FlyGym), with "
        "biologically identified sensory populations encoding stimuli and descending neuron "
        "populations decoding motor commands through an interpretable sensorimotor interface. "
        "The connectome-structured brain model produces adaptive responses without learning: "
        "causal locomotion control (10/10 ablation tests, forward drive reduced 46%), "
        "olfactory valence discrimination (opposite turning for attractive vs aversive odors, "
        "6/6 tests), and looming escape (escape index 1.11, abolished 21-fold in shuffled "
        "controls). Direct (1-hop) sensory-to-motor connectivity is highly modality-specific "
        "(Jaccard 0.005\u20130.060), with convergence occurring one synapse deeper through "
        "modality-specific interneuron pools. Olfactory signals reach only 1 DN directly but, "
        "when weighted by synaptic strength, drive ~2% of the DN population at 2-hops \u2014 "
        "matching physiological recordings (Aymanns et al. 2022). Thermo-hygro signals "
        "converge on DNb05, a multimodal steering neuron (Namiki et al. 2018; Yang et al. "
        "2024) that despite receiving only ~1.4% of its input from thermo/hygro, is the sole "
        "direct thermo/hygro\u2192DN pathway \u2014 its ablation collapses thermo/hygro throughput "
        "93\u2013100% (specificity 16.4\u00d7), demonstrating that labeled-line function can emerge "
        "from a minority input to a multimodal neuron."
    )
    pdf.set_font("tnr", "", 10)
    pdf.set_left_margin(25)
    pdf.set_right_margin(25)
    pdf.multi_cell(0, 5, abstract)
    pdf.set_left_margin(15)
    pdf.set_right_margin(15)

    pdf.ln(3)
    pdf.set_font("tnr", "B", 9)
    pdf.set_x(25)
    pdf.write(4.5, "Keywords: ")
    pdf.set_font("tnr", "", 9)
    pdf.write(4.5, "connectome, Drosophila, sensorimotor, descending neurons, labeled lines, "
              "looming escape, olfactory valence, whole-brain simulation")
    pdf.ln(6)

    # ══════════════════════════════════════════════════════════════════
    #  1. INTRODUCTION
    # ══════════════════════════════════════════════════════════════════
    pdf.add_page()
    pdf.section_heading("1", "Introduction")

    pdf.body_text(
        "The completion of the Drosophila melanogaster whole-brain connectome "
        "(Dorkenwald et al. 2024, FlyWire) provides, for the first time, a neuron-level "
        "wiring diagram of an animal capable of complex behavior. The full resource "
        "comprises 139,255 neurons and approximately 50 million chemical synapses, including "
        "identified sensory input populations and descending motor output pathways. Yet a "
        "fundamental question remains: can connectome-structured dynamics support meaningful "
        "embodied sensorimotor transformations without learning or task-specific optimization?",
        indent=False)

    pdf.body_text(
        "Previous work has used the connectome for circuit analysis [2], connectome-constrained "
        "network simulations [3], and anatomical tracing of specific pathways. To our knowledge, "
        "no open system has previously coupled a FlyWire-scale whole-brain model to an embodied "
        "fly simulator and validated behaviorally specific circuit function in closed loop. "
        "Locomotion in our system is mediated through a CPG/VNC-like interface \u2014 the brain "
        "modulates gait parameters rather than controlling individual joints directly \u2014 which "
        "we state explicitly as both a design choice and a current limitation.")

    pdf.body_text("Here we build such a system and make three contributions:", indent=False)

    pdf.bold_start_para("Behavioral specificity.",
        "The connectome-structured brain model, coupled to an embodied fly through an "
        "interpretable sensorimotor interface, produces three distinct adaptive behaviors "
        "\u2014 locomotion control, olfactory valence discrimination, and looming escape "
        "\u2014 without parameter fitting or learning.")

    pdf.bold_start_para("Graded causal control.",
        "Targeted ablation of identified neuron populations produces graded, quantitatively "
        "predicted behavioral deficits, confirming that specific connectome pathways causally "
        "drive specific behaviors.")

    pdf.bold_start_para("Modality-specific descending channels.",
        "Analysis of the descending neurons functionally recruited by each behavior reveals "
        "a previously unreported structural principle: sensory modalities maintain near-complete "
        "segregation at the direct sensory-to-descending interface, converging only through a "
        "single interneuron layer. The connectome implements parallel labeled lines for "
        "multisensory motor control.")

    # ══════════════════════════════════════════════════════════════════
    #  2. RESULTS
    # ══════════════════════════════════════════════════════════════════
    pdf.section_heading("2", "Results")

    # ── 2.1 ──────────────────────────────────────────────────────────
    pdf.subsection_heading("2.1", "A closed-loop connectome-body interface")

    pdf.body_text(
        "We built a brain-body bridge coupling the FlyWire connectome to a biomechanical "
        "fly body (Fig. 1). The system has three components, two from public resources and "
        "one that is our contribution:", indent=False)

    pdf.bold_start_para("Brain (public).",
        "A Brian2 leaky integrate-and-fire (LIF) simulation of 138,639 neurons from the "
        "FlyWire 783 completeness snapshot (15 million connection pairs comprising 54.5 "
        "million synapses), following the model architecture of Shiu et al. [3]. All neurons "
        "share identical biophysical parameters; synaptic weights are proportional to synapse "
        "count from the connectome. No parameters were tuned or optimized.")

    pdf.bold_start_para("Body (public).",
        "A MuJoCo-based biomechanical hexapod via FlyGym v1.2.1 [5] with 42 actuated degrees "
        "of freedom, realistic contact physics, and preprogrammed tripod gait kinematics.")

    pdf.bold_start_para("Sensorimotor interface (our contribution).",
        "The bridge between brain and body consists of a sensory encoder and a descending "
        "decoder. These are the designed components of our system \u2014 they define which "
        "neurons receive sensory input, which neurons are read out as motor commands, and how "
        "those signals are encoded/decoded. The brain itself is unmodified from the connectome.")

    pdf.italic_start_para("Sensory encoder.",
        "Body state and environmental stimuli are encoded as Poisson firing rates injected "
        "into biologically identified sensory neuron populations. We define 10 sensory channels "
        "totaling 275\u2013485 neurons (depending on experiment): gustatory (20 sugar GRNs), "
        "proprioceptive (33 SEZ ascending neurons), mechanosensory (15 SEZ ascending), "
        "vestibular (7 SEZ ascending), bilateral olfactory (100 ORNs per side), bilateral "
        "visual (50 photoreceptors per side), and bilateral LPLC2 looming detectors (108 left, "
        "102 right).")

    pdf.italic_start_para("Descending decoder.",
        "Firing rates from 204\u2013389 descending neurons (DNs) are decoded into four locomotion "
        "commands \u2014 forward drive, turn drive, step frequency, and stance gain \u2014 which "
        "modulate the body\u2019s CPG. DN populations were selected using a hybrid approach: "
        "annotated motor/descending types from the connectome plus connectivity-augmented "
        "supplements. Each simulation step runs a 20\u2013100 ms brain window followed by "
        "200\u20131000 body timesteps. A 200 ms warmup period allows brain activity to stabilize.")

    # ── 2.2 ──────────────────────────────────────────────────────────
    pdf.subsection_heading("2.2", "Causal locomotion control from connectome wiring")

    pdf.body_text(
        "We tested whether the connectome produces functionally distinct locomotion commands "
        "by selectively ablating (silencing) each of five neuron groups: forward, turn-left, "
        "turn-right, rhythm, and stance (Fig. 2A). All 10 causal tests passed (5 command-level, "
        "5 behavioral):", indent=False)

    w = pdf.w - pdf.l_margin - pdf.r_margin
    pdf.add_table(
        ["Ablation", "Command effect", "Behavioral effect"],
        [
            ["Forward silenced", "Fwd drive: 0.97 \u2192 0.10 (\u221290%)", "Distance: 20.6 \u2192 11.2 mm (\u221246%)"],
            ["Turn-left silenced", "Turn shifts rightward", "Heading: +129.6\u00b0 rightward"],
            ["Turn-right silenced", "Turn shifts leftward", "Heading: \u2212137.0\u00b0 leftward"],
            ["Rhythm silenced", "Step freq: 2.05 \u2192 1.00 (\u221251%)", "Reduced stepping"],
            ["Stance silenced", "Stance gain: 1.49 \u2192 1.00 (\u221233%)", "Altered contact"],
        ],
        col_widths=[w*0.25, w*0.40, w*0.35]
    )

    pdf.body_text(
        "The shuffled connectome control (postsynaptic indices permuted, preserving out-degree) "
        "produced 3/10 passing tests \u2014 only trivial contrast comparisons between two "
        "near-zero signals. All neuron groups showed 0% activity (0.0 Hz mean firing rate), "
        "confirming that the shuffled network has no functional circuit structure.", indent=False)

    pdf.bold_start_para("Dose-response.",
        "Progressive ablation of the forward neuron population produced a graded, near-linear "
        "reduction in forward distance (Fig. 2B): 0% ablation = 19.7 mm, 25% = 18.8 mm, "
        "50% = 16.4 mm, 75% = 11.6 mm, 100% = 8.6 mm. Each 10 neurons inactivated reduced "
        "forward distance by approximately 1 mm, demonstrating that the connectome implements "
        "a distributed, graded population code for locomotion drive.")

    # ── 2.3 ──────────────────────────────────────────────────────────
    pdf.subsection_heading("2.3", "Olfactory valence discrimination")

    pdf.body_text(
        "Drosophila innately approach some odors and avoid others. We tested whether the "
        "connectome encodes this valence discrimination by presenting bilateral olfactory "
        "stimuli corresponding to two identified odor channels: Or42b (projecting to DM1 "
        "glomerulus, attractive) and Or85a (projecting to DM5 glomerulus, aversive). For each "
        "odor type, we presented asymmetric stimulation (high intensity on one side, zero on "
        "the other) and measured the resulting turn drive.", indent=False)

    # Figure 3
    pdf.add_figure(FIG_DIR / "odor_valence_proof.png",
        "Connectome-encoded odor valence discrimination. Asymmetric olfactory stimulation "
        "reveals opposite turning for attractive (DM1/Or42b) and aversive (DM5/Or85a) odors. "
        "Shuffled connectome eliminates all valence signals. 6/6 validation tests pass.",
        fig_num=3, width=w*0.95)

    pdf.bold_start_para("Results (n=10 seeds):",
        "DM1 (attractive): turn contrast = \u22120.002 (toward odor source). "
        "DM5 (aversive): turn contrast = +0.033 (away from odor source). "
        "Valence contrast (DM5 \u2212 DM1) = +0.035. All 6 validation tests passed. "
        "The shuffled connectome eliminated all valence signals (contrast = +0.002), "
        "confirming that odor valence is encoded in specific synaptic wiring.")

    # ── 2.4 ──────────────────────────────────────────────────────────
    pdf.subsection_heading("2.4", "Looming escape via LPLC2 descending pathway")

    pdf.body_text(
        "Flies exhibit rapid escape responses to looming visual stimuli. The LPLC2 neurons "
        "in the lobula plate are known looming detectors. We identified 210 LPLC2 neurons in "
        "the FlyWire connectome (108 left, 102 right) and traced their projections to 44 "
        "descending neurons via 1,850 direct synapses, confirming a strongly ipsilateral "
        "projection pattern.", indent=False)

    # Figure 4
    pdf.add_figure(FIG_DIR / "looming_escape_proof.png",
        "Connectome-encoded looming escape response. LPLC2 looming detectors (210 neurons) "
        "drive contralateral escape turning via 44 descending neurons. Escape index 1.112, "
        "robust across 20/50/100 ms brain windows. Shuffled connectome: 21-fold weaker. "
        "5/5 tests pass.",
        fig_num=4, width=w*0.95)

    pdf.bold_start_para("Results (n=5 seeds, 50 ms brain window):",
        "Loom left (left LPLC2 active): turn drive = +0.364 (rightward escape). "
        "Loom right (right LPLC2 active): turn drive = \u22120.748 (leftward escape). "
        "Control: turn drive = \u22120.181. Escape index (loom_left \u2212 loom_right): 1.112. "
        "Robust across brain windows: 1.084 at 20 ms, 1.112 at 50 ms, 1.109 at 100 ms. "
        "Shuffled connectome: escape index 0.053 \u2014 21-fold weaker. 5/5 tests pass.")

    # ── 2.5 ──────────────────────────────────────────────────────────
    pdf.add_page()
    pdf.subsection_heading("2.5", "Modality-specific descending motor channels")

    pdf.body_text(
        "Recent large-scale connectomic analyses have established that descending neuron types "
        "cluster by sensory input modality (St\u00fcrner et al. 2025), with 16 clusters ranging from "
        "modality-enriched to broadly multimodal. However, these analyses used information-flow "
        "distance (shortest path length) to assign sensory modality, which does not distinguish "
        "strong direct wiring from weak multi-synaptic paths. Here we complement that approach "
        "with direct (1-hop) and synapse-weighted connectivity measurements, asking whether the "
        "modality organization observed at the type level by St\u00fcrner et al. is also present "
        "\u2014 and potentially sharper \u2014 at the level of individual synaptic connections.", indent=False)

    pdf.body_text(
        "We traced all direct (1-hop) connections from each sensory modality to the 350-neuron "
        "descending pool (389 selected, 350 mapped to valid connectome root IDs), then repeated "
        "the analysis at 2-hops (sensory \u2192 interneuron \u2192 DN) with synapse-count weighting "
        "to assess functional drive strength.")

    pdf.bold_start_para("1-hop: near-complete segregation (Fig. 5A).", "")

    pdf.add_table(
        ["Modality", "Neurons", "DNs reached", "Edges", "Synapses"],
        [
            ["Somatosensory", "75", "186", "783", "7,374"],
            ["Visual (LPLC2)", "310", "44", "1,850", "8,823"],
            ["Olfactory", "100", "1", "2", "7"],
        ],
        col_widths=[w*0.25, w*0.15, w*0.20, w*0.20, w*0.20]
    )

    pdf.body_text(
        "Pairwise overlap: Olfactory\u2013Visual Jaccard = 0.023, Olfactory\u2013Somatosensory "
        "= 0.005, Visual\u2013Somatosensory = 0.060. 173 DNs are exclusively "
        "somatosensory-reachable; 31 exclusively visual-reachable; zero exclusively "
        "olfactory-reachable. 133 readout DNs receive no direct sensory input from any "
        "modality. The direct sensory-to-motor interface maintains near-complete modality "
        "segregation.", indent=False)

    pdf.bold_start_para("Shuffled connectome control.",
        "To test whether this segregation could arise from population size differences, we "
        "shuffled the connectome (n=5 permutations, preserving out-degree). Shuffled wiring "
        "produced 4.6\u201321.4\u00d7 higher pairwise Jaccard: somatosensory\u2013visual 0.273 "
        "vs 0.060 (4.6\u00d7), somatosensory\u2013olfactory 0.115 vs 0.005 (21.4\u00d7), "
        "visual\u2013olfactory 0.155 vs 0.023 (6.8\u00d7). The near-zero overlap is not a "
        "trivial consequence of small populations \u2014 it is actively maintained by the "
        "specific synaptic wiring.")

    pdf.bold_start_para("The 13 shared visual-somatosensory DNs.",
        "All 13 receive input specifically from LPLC2 looming detectors (not generic R7/R8 "
        "photoreceptors, which have zero direct DN connections). These are annotated "
        "looming-escape turning types: DNp11, DNp27, DNp05, DNp35, DNp69, DNp70, DNa07, "
        "DNae007, DNc01, DNpe042 \u2014 8 in turn-right, 4 in turn-left, 1 in rhythm. The "
        "sole multimodal convergence point at the DN level is the escape circuit, where "
        "millisecond integration of visual threat and somatosensory body state is "
        "survival-critical.")

    pdf.bold_start_para("Excitatory/inhibitory asymmetry.",
        "Visual\u2192DN connections are 100% excitatory (1,850/1,850). Somatosensory\u2192DN "
        "connections include 21.2% inhibitory (166/783 edges), suggesting active suppression "
        "of specific motor programs by body state.")

    pdf.bold_start_para("2-hop: rapid convergence through separate interneuron pools.", "")

    pdf.add_table(
        ["Modality", "Active intermediates", "DNs reached (2-hop)"],
        [
            ["Somatosensory", "4,475", "350 (100%)"],
            ["Visual", "2,279", "304 (87%)"],
            ["Olfactory", "488", "175 (50%)"],
        ],
        col_widths=[w*0.30, w*0.35, w*0.35]
    )

    pdf.body_text(
        "Somatosensory reaches every readout DN at 2-hop. Critically, this convergence occurs "
        "through modality-specific interneuron pools. Cross-modality sharing is minimal: "
        "Visual\u2013Somatosensory Jaccard = 0.048 (310 shared intermediates), "
        "Somatosensory\u2013Olfactory = 0.002 (10 shared), Visual\u2013Olfactory = 0.000 "
        "(zero shared). Visual and olfactory modalities share zero intermediates en route to "
        "DNs. The 2-hop convergence is not mediated by shared relay neurons \u2014 each modality "
        "maintains its own interneuron pool that independently projects to overlapping DN "
        "targets.", indent=False)

    pdf.bold_start_para("Somatosensory subchannels are themselves segregated.",
        "Within the somatosensory modality, proprioceptive (65 DNs), mechanosensory (86 DNs), "
        "vestibular (77 DNs), and gustatory (23 DNs) subchannels maintain low pairwise overlap "
        "(Jaccard 0.048\u20130.180), with 34\u201343 exclusive DNs per subchannel. The labeled-line "
        "principle extends to individual sensory submodalities.")

    pdf.body_text(
        "The visual channel shows extreme lateralization: left LPLC2 reaches 24 DNs, right "
        "LPLC2 reaches 21 DNs, with only 1 shared DN (Jaccard \u2248 0). Generic R7/R8 "
        "photoreceptors have zero direct DN connections \u2014 all 8,823 visual\u2192DN synapses "
        "come from LPLC2 looming detectors. This bilateral segregation is the structural basis "
        "for the contralateral escape response.")

    # ── 2.6 ──────────────────────────────────────────────────────────
    pdf.subsection_heading("2.6", "Functional mapping of decoder groups by modality")

    pdf.body_text(
        "The segregation extends into specific motor functions. We quantified per-neuron "
        "synaptic drive (total synapses / group size) from each modality into each decoder "
        "group:", indent=False)

    pdf.add_table(
        ["Group", "Somato (syn/n)", "Visual (syn/n)", "Olf.", "Dominant subchannel"],
        [
            ["Forward (59)", "86.1", "0.0", "0.1", "gustatory (44.7)"],
            ["Turn left (136)", "15.7", "31.2", "0.0", "LPLC2_L (31.2)"],
            ["Turn right (167)", "14.5", "27.6", "0.0", "LPLC2_R (27.4)"],
            ["Rhythm (25)", "41.6", "0.1", "0.0", "proprioceptive (15.6)"],
            ["Stance (25)", "46.0", "0.0", "0.0", "gustatory (32.6)"],
        ],
        col_widths=[w*0.18, w*0.18, w*0.18, w*0.10, w*0.36]
    )

    pdf.body_text(
        "Visual input (LPLC2) preferentially targets turning groups (31.2 and 27.6 syn/neuron) "
        "\u2014 5\u00d7 stronger than somatosensory drive to those groups \u2014 consistent with "
        "looming-evoked escape turns. Forward locomotion is dominated by gustatory input "
        "(44.7 syn/neuron), consistent with food-seeking drive. Stance control is exclusively "
        "somatosensory at 1-hop \u2014 zero visual and zero olfactory synapses. This pattern "
        "fell out of the connectome analysis and was not designed into the decoder.", indent=False)

    # ── 2.7 ──────────────────────────────────────────────────────────
    pdf.subsection_heading("2.7", "Extended labeled lines: auditory, thermosensory, and hygrosensory")

    pdf.body_text(
        "To test whether the labeled-line principle generalizes beyond the three modalities "
        "used in behavioral experiments, we extended the segregation analysis to three "
        "additional sensory populations identified from FlyWire annotations: auditory "
        "(390 Johnston\u2019s organ neurons), thermosensory (29 neurons: 7 heating/TRN_VP2, "
        "9 cold/TRN_VP3a-b, 13 humidity-sensitive/TRN_VP1m), and hygrosensory (74 neurons: "
        "29 dry/HRN_VP4, 16 moist/HRN_VP5, 16 evaporative cooling/HRN_VP1d, 13 cooling/"
        "HRN_VP1l).", indent=False)

    pdf.bold_start_para("Auditory: a semi-independent turning/rhythm channel.",
        "Auditory neurons reach 41 DNs at 1-hop (405 edges, 1,563 synapses). The auditory "
        "channel shows moderate overlap with visual (Jaccard = 0.164, 12 shared DNs) and "
        "somatosensory (0.066, 14 shared), but zero overlap with olfactory (0.000). Its "
        "strongest per-neuron drive targets the rhythm group (13.7 syn/neuron) and bilateral "
        "turning (5.1 left, 4.9 right), consistent with auditory-driven courtship orientation "
        "and song-evoked locomotor modulation. Shuffled controls confirm the visual-auditory "
        "overlap is still 3.8\u00d7 below chance.")

    pdf.bold_start_para("Thermosensory: an extremely narrow labeled line.",
        "Thermosensory neurons reach only 5 DNs (14 edges, 162 synapses) \u2014 the narrowest "
        "direct pathway of any modality tested. All 5 target DNs are in the turning groups "
        "(4 turn-right, 1 turn-left), consistent with thermotaxis. The dominant target is "
        "DNb05 (151/162 synapses, 93%). DNb05 is a known multimodal steering neuron "
        "(Namiki et al. 2018; Yang et al. 2024) whose total input is dominated by visual "
        "(31%) and central (62%) sources \u2014 thermo/hygro constitutes only ~1.4% of its "
        "24,866 input synapses. Yet it is the sole direct thermo/hygro\u2192DN pathway.")

    pdf.bold_start_para("Hygrosensory: convergence with thermosensory on DNb05.",
        "Hygrosensory neurons reach just 2 DNs (13 edges, 200 synapses) \u2014 both are DNb05. "
        "The thermo-hygro Jaccard (0.400) is 10\u00d7 higher than shuffled controls, confirming "
        "genuine convergence. Separately, DNp44 \u2014 identified by Marin et al. (2020) as the "
        "primary hygrosensory descending neuron \u2014 is in our readout pool but receives "
        "hygrosensory input via a 2-hop pathway through VP projection neurons (15\u201322 relay "
        "neurons, ~2,700 synapses), not directly. The coexistence of a direct but weak pathway "
        "(DNb05, 200 synapses) and an indirect but strong pathway (DNp44, ~2,700 at 2-hop) "
        "suggests parallel thermo/hygro channels operating at different latencies.")

    pdf.bold_start_para("Six-modality segregation summary.", "")

    pdf.add_table(
        ["Pair", "Shared DNs", "Jaccard"],
        [
            ["Auditory\u2013Visual", "12", "0.164"],
            ["Auditory\u2013Somatosensory", "14", "0.066"],
            ["Auditory\u2013Thermosensory", "2", "0.045"],
            ["Thermo\u2013Hygrosensory", "2", "0.400"],
            ["Olfactory\u2013(all others)", "0\u20131", "0.000"],
        ],
        col_widths=[w*0.45, w*0.25, w*0.30]
    )

    pdf.body_text(
        "Direct (1-hop) connectivity is highly modality-specific across all six modalities. "
        "Olfactory remains the most isolated (Jaccard = 0.000 with all five others), "
        "consistent with its multi-synaptic pathway architecture and with physiological "
        "recordings showing only 2\u20135% of DNs encode odor information (Aymanns et al. 2022). "
        "The two cross-modal convergence points \u2014 thermo-hygro (DNb05) and auditory-visual "
        "(turning DNs, consistent with Sturner et al. 2025 Clusters 14 and 16) \u2014 occur where rapid "
        "multimodal integration is biologically relevant. At 2-hops, olfactory and "
        "thermo/hygrosensory share substantial intermediate overlap (Jaccard 0.287\u20130.306), "
        "consistent with Sturner et al. grouping these modalities into a shared "
        "information-flow cluster.", indent=False)

    # ── 2.8 ──────────────────────────────────────────────────────────
    pdf.subsection_heading("2.8", "Causal bottleneck validation")

    pdf.body_text(
        "To test whether the structural bottlenecks identified in Section 2.7 have functional "
        "consequences, we performed two targeted silencing experiments.", indent=False)

    pdf.bold_start_para("Experiment 1: DNb05 silencing.",
        "DNb05 is a bilateral descending neuron pair known as a multimodal steering neuron "
        "receiving visual (31%), central (62%), auditory (0.8%), hygrosensory (0.8%), and "
        "thermosensory (0.6%) input (Namiki et al. 2018; Yang et al. 2024). Despite "
        "thermo/hygro constituting only ~1.4% of its 24,866 input synapses, DNb05 is the "
        "sole direct target for these modalities. Silencing produced complete hygrosensory "
        "collapse (100% of DNs and synapses lost) and near-complete thermosensory collapse "
        "(40% of DNs, 93.2% of synapses). Somatosensory lost 0.5%, visual 0%, olfactory 0% "
        "\u2014 specificity ratio 16.4\u00d7. This demonstrates that labeled-line function does not "
        "require dedicated anatomy: a multimodal neuron can serve as a functional bottleneck "
        "for a minority input modality when it is the sole direct pathway.")

    pdf.bold_start_para("Experiment 2: Auditory-visual orientation channel silencing.",
        "The 12 DNs shared between auditory and visual modalities (5 turn-left, 7 turn-right; "
        "consistent with Sturner et al. 2025 Clusters 14 and 16) were silenced. Auditory lost 29.3% of "
        "DN targets and 25.5% of synapses. Visual lost 27.3% of DN targets and 44.2% of "
        "synapses \u2014 nearly half of visual motor output passes through these shared turning "
        "DNs. The impact was turning-specific: auditory turn_right throughput dropped 35.1%, "
        "turn_left 16.5%, while forward, rhythm, and stance remained at 0% loss. Olfactory, "
        "thermosensory, and hygrosensory were completely unaffected (0% each). All 4/4 tests "
        "passed. Combined: 8/8 causal tests across both experiments.")

    # -- 2.9 ──────────────────────────────────────────────────────────
    pdf.subsection_heading("2.9", "Sparsity, not specific wiring, determines learning speed")

    pdf.body_text(
        "We tested whether the connectome's specific wiring topology provides an inductive bias "
        "for motor learning by training recurrent policies on a forward-locomotion task using "
        "evolutionary strategies (ES). Three architectures were compared: (1) connectome-constrained "
        "(recurrent mask from VNC connectome), (2) random sparse (same sparsity, random edges), "
        "and (3) shuffled (connectome edges randomly reassigned). All architectures had identical "
        "neuron counts (2,314: 1,314 DN + 500 MN + 500 intrinsic), identical sparsity (3.6%), "
        "and identical training hyperparameters (population 128, sigma 0.03, 50 generations).",
        indent=False)

    pdf.body_text(
        "Random sparse networks reached connectome-level performance by generation 40 "
        "(fitness +11.6 vs connectome band +10 to +15). This demonstrates that "
        "sparsity alone explains the learning advantage -- the specific wiring pattern of the "
        "connectome does not accelerate optimization. However, the connectome architecture "
        "provides a qualitative advantage in interpretability: 4.3x higher modularity (Newman Q: "
        "0.20 vs 0.05), 100% intrinsic-neuron criticality vs 33% for random sparse, and 2x "
        "higher pathway concentration (weight Gini: 0.76 vs 0.66). The connectome routes "
        "information through identifiable bottlenecks that can be named, traced, and ablated; "
        "the random network achieves equal performance through redundant parallel paths that "
        "resist interpretation.")

    # ══════════════════════════════════════════════════════════════════
    #  3. DISCUSSION
    # ══════════════════════════════════════════════════════════════════
    pdf.add_page()
    pdf.section_heading("3", "Discussion")

    pdf.body_text(
        "The connectome-structured brain model, when coupled to an embodied fly through a "
        "transparent sensorimotor interface, produces adaptive and behaviorally specific "
        "responses without learning or parameter fitting. This establishes that connectome "
        "wiring carries substantial functional information \u2014 sufficient, within this "
        "embodied interface and task family, to support stimulus-specific sensorimotor "
        "transformations.", indent=False)

    pdf.bold_start_para("Scope of the sufficiency claim.",
        "Our system includes designed components: a sensory encoder, a descending decoder, "
        "and a CPG locomotion layer. We have shown sufficiency of the connectome within this "
        "interface, not sufficiency of wiring in the philosophical sense. Real flies rely on "
        "neuromodulation, synaptic plasticity, and experience to refine these circuits. Our "
        "simulation provides a computational baseline against which the contributions of these "
        "additional mechanisms can be measured.")

    pdf.bold_start_para("Direct connectivity is modality-specific; convergence is layer-dependent.",
        "At the direct (1-hop) sensory-to-DN interface, connectivity is highly modality-specific "
        "(Jaccard 0.005\u20130.060), with rapid convergence one synapse deeper. This complements "
        "Sturner et al. (2025), who found most DN types are multimodal when assessed by "
        "information-flow ranking. Our analysis adds a finer-grained measurement: direct "
        "synaptic connectivity is more specific than information-flow metrics suggest, and "
        "this specificity dilutes at each additional synaptic layer.")

    pdf.body_text(
        "Three aspects of the segregation strengthen this interpretation. First, the 13 shared "
        "visual-somatosensory DNs are specifically looming-escape turning neurons \u2014 the one "
        "behavior where millisecond integration of visual threat and body state is critical. "
        "Second, 2-hop convergence occurs through modality-specific interneuron pools (visual and "
        "olfactory share zero intermediates), meaning the wiring maintains labeled lines at both "
        "the DN level and the relay level. Third, visual\u2192DN connections are entirely "
        "excitatory (100%), while somatosensory\u2192DN includes 21% inhibitory connections, "
        "suggesting that body state can actively suppress motor programs \u2014 an asymmetry that "
        "pure convergence architectures would not predict.")

    pdf.body_text(
        "The stance exclusivity result is particularly clean: stance-controlling DNs receive "
        "zero visual or olfactory input at 1-hop, exclusively somatosensory. The dominant "
        "subchannel is gustatory (32.6 syn/neuron), consistent with stance modulation during "
        "feeding. This emerged from the connectome analysis and was not designed into the decoder.")

    pdf.body_text(
        "This direct-connectivity specificity has not been previously quantified at the "
        "population level because it requires both (a) a complete connectome to trace all "
        "paths and (b) a behavioral readout to identify which DNs are functionally relevant. "
        "Our 1-hop analysis is complementary to, not contradictory with, the multimodal "
        "convergence at longer path lengths documented by Sturner et al. (2025) and the "
        "multi-synaptic sensory processing described by Marin et al. (2020).")

    pdf.bold_start_para("Limitations.",
        "Our model uses uniform synaptic parameters (weights proportional to synapse count, "
        "identical time constants). Real synapses vary in strength, sign, and dynamics. We do "
        "not model gap junctions, neuromodulation, or synaptic plasticity. The CPG is a "
        "preprogrammed tripod gait, not a VNC connectome model. Our sensory encoding uses "
        "simplified Poisson rate coding. Our analysis uses the FlyWire 783 completeness snapshot "
        "(138,639 of ~139,255 neurons). Neurons excluded due to low proofreading completeness "
        "may include bridging interneurons that would reduce the observed modality segregation "
        "\u2014 a conservative analysis using the full connectome is warranted. Despite these "
        "simplifications, the system produces robust, stimulus-specific behavior.")

    pdf.bold_start_para("Bottleneck-by-exclusion: labeled-line function without dedicated anatomy.",
        "DNb05 is a visual-dominant multimodal steering neuron (31% visual; Namiki et al. 2018; "
        "Yang et al. 2024), yet it functions as a labeled-line bottleneck for thermo/hygro "
        "because it provides exclusive synaptic access for these modalities to motor output. "
        "The mechanism is not \u2018dedicated anatomy\u2019 but \u2018exclusive access\u2019 \u2014 DNs "
        "are organized by output channel, not input identity. Prior work has shown labeled-line "
        "function at the population level (Huoviala et al. 2020). Here we demonstrate it at the "
        "single-neuron level: despite receiving only 1.4% of its input from thermo/hygro, DNb05 "
        "is the sole synaptic gateway \u2014 silencing 2 neurons collapses throughput 93\u2013100% "
        "(16.4\u00d7 specificity). The auditory-visual 12 shared DNs span Sturner et al. (2025) "
        "Clusters 14 (auditory-enriched) and 16 (visual-enriched), bridging two modality-specific "
        "processing streams for orientation. Olfactory signals require 3\u20134 synapses to reach "
        "DNs via the obligate PN\u2192LH pathway; synapse-weighted 2-hop analysis confirms "
        "only 2.0% of DNs receive strong olfactory drive, matching Aymanns et al. (2022).")

    pdf.bold_start_para("VNC-lite premotor dynamics.",
        "We replaced the instantaneous decoder-to-actuator mapping with a bilateral premotor "
        "state model (VNC-lite) that interposes leaky integrator dynamics between descending "
        "neuron rates and locomotion commands. DN input drives state derivatives rather than raw "
        "outputs: d(state)/dt = \u2212state/\u03c4 + f(DN_input) + g(body_feedback). This gives the "
        "motor system temporal smoothing, persistence, bilateral competition (left/right mutual "
        "inhibition for turning), and feedback stabilization (body state corrects motor errors). "
        "All 20 validation tests pass with simulated input (backward compatibility 4/4, robustness "
        "across 7 parameter configurations 8/8, temporal smoothing 4/4, causal dissociation 4/4), "
        "and 19/20 with the full Brian2 brain. VNC-lite reduces command jitter by 47\u201397% "
        "compared to the original decoder while preserving all headline behavioral effects.")

    pdf.bold_start_para("Topology learning: sparsity, not wiring, determines learning speed.",
        "Section 2.9 shows that random sparse networks match connectome performance on forward "
        "locomotion. This negative result strengthens rather than weakens our claims: the "
        "connectome's value is not as an inductive bias for optimization (which generic sparsity "
        "provides equally well), but as a documented circuit whose components can be identified, "
        "traced, and causally tested. The interpretability advantage (4.3x modularity, 100% vs 33% "
        "intrinsic criticality) is the primary engineering value of biological wiring over generic "
        "sparse alternatives.")

    pdf.bold_start_para("Negative result: VNC rhythm generation.",
        "We attempted to generate locomotor rhythm directly from the MANC ventral nerve cord "
        "connectome (13,101 neurons, 1.9 million synapses) without an imposed CPG. Specifically, "
        "we searched for half-center oscillation by testing 6,318 reciprocally connected premotor "
        "pairs across 41 parameter configurations. No configuration produced anti-phase flexor/"
        "extensor alternation; all reciprocal pairs showed positive correlation (r = +0.02 to "
        "+0.87). The MANC connectome, at LIF-level biophysical resolution, does not produce "
        "emergent rhythm. This is consistent with the hypothesis that rhythm generation requires "
        "membrane-level properties (persistent sodium, calcium channels) not captured by LIF "
        "neurons, and that CPG timing in insects may depend on neuromodulatory state rather than "
        "purely synaptic architecture.")

    pdf.bold_start_para("Future directions.",
        "Replacing the preprogrammed CPG with a biophysically detailed VNC model incorporating "
        "conductance-based neurons would test whether rhythm generation emerges from the full "
        "complement of ion channels. The dose-response relationship between population size and "
        "behavioral effect suggests that the system can predict the behavioral consequences of "
        "genetic manipulations that silence specific neuron types. Hardware deployment of the "
        "minimal VNC circuit (1,000 neurons extracted from MANC) on a physical hexapod robot "
        "would test whether connectome-derived controllers transfer to real systems.")

    # ══════════════════════════════════════════════════════════════════
    #  4. METHODS
    # ══════════════════════════════════════════════════════════════════
    pdf.add_page()
    pdf.section_heading("4", "Methods")

    pdf.subsection_heading("4.1", "Brain simulation")
    pdf.body_text(
        "We simulated 138,639 neurons from the FlyWire 783 completeness snapshot as leaky "
        "integrate-and-fire units using Brian2 (v2.5). This represents 99.6% of the full "
        "FlyWire resource (139,255 neurons). Synaptic connections (15,091,983 connection pairs "
        "comprising 54,492,922 individual synapses; mean 3.6 synapses per connection) used "
        "conductance-based synapses with weights proportional to synapse count. All neurons "
        "shared identical biophysical parameters (membrane time constant, threshold, reset "
        "potential). No parameter tuning or optimization was performed.", indent=False)

    pdf.subsection_heading("4.2", "Sensory populations")
    pdf.body_text(
        "Sensory neuron populations were identified from FlyWire annotations: sugar GRNs for "
        "gustatory input [3], SEZ ascending neuron types classified by out/in connectivity "
        "ratio (>1.5 = ascending) for proprioceptive, mechanosensory, and vestibular channels, "
        "bilateral ORNs for olfaction (Or42b/DM1, Or85a/DM5), bilateral photoreceptors (R7/R8) "
        "for vision, and LPLC2 lobula plate neurons for looming detection.", indent=False)

    pdf.italic_start_para("Sensory encoding.",
        "Each channel maps body observations to Poisson firing rates using channel-appropriate "
        "nonlinearities. Proprioceptive neurons receive rates proportional to tanh-normalized "
        "joint angles and velocities, tiled across the population. Mechanosensory neurons "
        "receive rates proportional to per-leg contact force magnitudes. Vestibular neurons "
        "encode body velocity and orientation via tanh normalization. Gustatory neurons receive "
        "a uniform rate modulated by mean contact force. All channels map to [10, 100] Hz. "
        "LPLC2 looming channels use [10, 200] Hz. Visual channels encode mean eye luminance.")

    pdf.subsection_heading("4.3", "Readout populations")
    pdf.body_text(
        "Descending neuron (DN) populations were selected using a hybrid approach: annotated "
        "motor/descending types from the SEZ neuron dataset (out/in ratio < 0.5) plus "
        "connectivity-augmented supplements identified by round-robin assignment from "
        "highest-connectivity downstream targets. Populations were assigned to five locomotion "
        "decoder groups (forward, turn-left, turn-right, rhythm, stance) based on annotated "
        "types and bilateral pairing.", indent=False)

    pdf.subsection_heading("4.4", "Body simulation")
    pdf.body_text(
        "The fly body was simulated using FlyGym v1.2.1 [5] with MuJoCo physics, featuring a "
        "hexapod with 42 actuated degrees of freedom. Locomotion used PreprogrammedSteps tripod "
        "gait kinematics with CPG phase offsets [0, \u03c0, 0, \u03c0, 0, \u03c0]. Brain output "
        "modulated four CPG parameters: forward drive (amplitude scaling), turn drive (left-right "
        "asymmetry), step frequency (CPG intrinsic frequency), and stance gain (joint magnitude "
        "multiplier).", indent=False)

    pdf.subsection_heading("4.5", "Ablation protocol")
    pdf.body_text(
        "For each of 8 conditions (baseline + 5 group ablations + 2 boost conditions), we "
        "built a fresh brain network, silenced the target population by setting firing rates "
        "to 0 Hz, and ran 5,000 body steps with a 20 ms brain integration window. Shuffled "
        "connectome controls used a fixed random seed (999) to permute all postsynaptic target "
        "indices while preserving out-degree.", indent=False)

    pdf.subsection_heading("4.6", "VNC-lite premotor dynamics")
    pdf.body_text(
        "Between the descending decoder and locomotion bridge, we interpose a bilateral premotor "
        "state model (VNC-lite) that replaces the instantaneous rate-to-command mapping with a "
        "dynamical system. Six state variables (drive_L, drive_R, turn_L, turn_R, rhythm, stance) "
        "evolve according to leaky integrator dynamics: d(state)/dt = \u2212state/\u03c4 + "
        "\u03b1 \u00b7 f(DN_input)/dt + coupling + feedback, where \u03c4 is a state-specific time "
        "constant (150\u2013300 ms), \u03b1 is an input gain, and f() is a tanh nonlinearity.", indent=False)

    pdf.body_text(
        "The model has three stages: (1) DN rate normalization and input mapping, (2) bilateral "
        "state dynamics with Euler integration \u2014 drive states are bilaterally coupled "
        "(synchronization), turn states have mutual inhibition \u2014 and (3) body feedback: velocity "
        "mismatch drives stance correction, body instability dampens rhythm, contact asymmetry "
        "produces corrective turning, and slip detection boosts stance. State variables are "
        "saturated to prevent runaway. Output mapping converts state to locomotion commands through "
        "tanh nonlinearities: forward_drive = 0.1 + 0.9 \u00b7 tanh(mean_drive), turn_drive = "
        "tanh(turn_L \u2212 turn_R), step_frequency = 1.0 + 1.5 \u00b7 tanh(rhythm), stance_gain = "
        "1.0 + 0.5 \u00b7 tanh(stance).")

    pdf.subsection_heading("4.7", "Segregation analysis")
    pdf.body_text(
        "For each sensory modality group (olfactory: 100 neurons; visual/LPLC2: 310; "
        "somatosensory: 75 across four subchannels), we identified all descending neurons "
        "receiving direct synaptic input (1-hop) from the full connectome. For 2-hop analysis, "
        "we identified all intermediate neurons that are both direct postsynaptic targets of "
        "sensory neurons and direct presynaptic to readout DNs. Overlap was quantified using "
        "the Jaccard index for all pairwise comparisons at both the DN level and the interneuron "
        "level. Excitatory/inhibitory classification used the Excitatory column from the "
        "connectome. Subchannel analysis repeated the 1-hop computation for each of the 10 "
        "individual sensory channels.", indent=False)

    pdf.subsection_heading("4.8", "Topology learning experiment")
    pdf.body_text(
        "We extracted a compressed VNC topology (2,314 neurons: 1,314 DN + 500 MN + 500 highest-"
        "connectivity intrinsic, 193,315 edges, sparsity 3.6%) from the MANC connectome. Three "
        "recurrent policies shared this neuron count and sparsity: connectome-constrained (edges "
        "from MANC), random sparse (same sparsity, random topology, seed 99), and shuffled (same "
        "edges, randomly reassigned targets). Policies were trained with evolutionary strategies "
        "(ES): population 128, sigma 0.03, 50 generations, on a forward-locomotion reward "
        "(distance - 0.1*instability - 0.01*energy). Interpretability metrics: Newman modularity Q "
        "(spectral bisection on learned W_rec), weight Gini coefficient, and group ablation "
        "deficit (DN, MN, intrinsic neuron classes silenced independently).", indent=False)

    # ══════════════════════════════════════════════════════════════════
    #  REFERENCES
    # ══════════════════════════════════════════════════════════════════
    pdf.section_heading("", "References")

    refs = [
        "Dorkenwald, S. et al. (2024). Neuronal wiring diagram of an adult brain. Nature, 634, 124\u2013138.",
        "Schlegel, P. et al. (2023). Whole-brain annotation and multi-connectome cell type quantification. Nature, 634, 139\u2013152.",
        "Shiu, P.K. et al. (2023). A leaky integrate-and-fire computational model based on the connectome of the entire adult Drosophila brain. bioRxiv.",
        "Lobato-Rios, V. et al. (2022). NeuroMechFly, a neuromechanical model of adult Drosophila melanogaster. Nature Methods, 19, 620\u2013627.",
        "Wang, F. et al. (2024). FlyGym: A comprehensive toolkit for biomechanical simulations of Drosophila. Nature Methods.",
        "Namiki, S. et al. (2018). The functional organization of descending sensory-motor pathways in Drosophila. eLife, 7, e34272.",
        "Cande, J. et al. (2018). Optogenetic dissection of descending behavioral control in Drosophila. eLife, 7, e34275.",
        "Marin, E.C. et al. (2020). Connectomics analysis reveals first-, second-, and third-order thermosensory and hygrosensory neurons. Curr. Biol., 30, 3167\u20133182.",
        "Aymanns, F., Chen, C.-L. & Ramdya, P. (2022). Descending neuron population dynamics during odor-evoked and spontaneous behaviors. eLife, 11, e81527.",
        "Yang, H.H. et al. (2024). Fine-grained descending control of steering in walking Drosophila. Cell, 187, 6290\u20136308.",
        "St\u00fcrner, T. et al. (2025). Comparative connectomics of Drosophila descending and ascending neurons. Nature, 643, 158\u2013172.",
        "Huoviala, P. et al. (2020). Neural circuit basis of aversive odour processing in Drosophila from sensory input to descending output. bioRxiv.",
        "Rayshubskiy, A. et al. (2024). Neural circuit mechanisms for steering control in walking Drosophila. eLife.",
        "Thoma, V. et al. (2016). Functional dissociation in sweet taste receptor neurons. Nature Communications.",
    ]
    pdf.set_font("tnr", "", 9)
    for i, ref in enumerate(refs, 1):
        pdf.cell(8, 4.5, f"[{i}]", new_x=XPos.END, new_y=YPos.TOP)
        pdf.multi_cell(0, 4.5, ref)
        pdf.ln(1)

    return pdf


if __name__ == "__main__":
    pdf = build_paper()
    out = PAPER_DIR / "connectome_sensorimotor_paper.pdf"
    pdf.output(str(out))
    print(f"PDF saved to {out}")
    print(f"Pages: {pdf.pages_count}")
