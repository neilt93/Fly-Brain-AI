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
    pdf.cell(0, 7, "Neil Bhatt", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
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
        "The connectome-structured brain model, when coupled to an embodied fly through this "
        "transparent interface, produces adaptive and behaviorally specific responses without "
        "learning or parameter fitting: causal locomotion control (10/10 ablation tests, "
        "forward drive reduced 53% by targeted silencing), olfactory valence discrimination "
        "(opposite turning for attractive vs aversive odors, 6/6 tests), and visually guided "
        "looming escape (contralateral turning with escape index 1.11, abolished 21-fold in "
        "shuffled connectome controls). Analysis of the descending neuron populations "
        "underlying these behaviors reveals a structural principle: at the direct "
        "sensory-to-motor interface, modalities maintain near-complete segregation (Jaccard "
        "index 0.005\u20130.060), with convergence occurring one synapse deeper through "
        "modality-specific interneuron pools that share zero intermediates between visual "
        "and olfactory pathways \u2014 the connectome implements labeled lines at both the "
        "descending and relay levels."
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
            ["Forward silenced", "Fwd drive: 0.90 \u2192 0.10 (\u221289%)", "Distance: 18.9 \u2192 9.0 mm (\u221253%)"],
            ["Turn-left silenced", "Turn shifts rightward", "Heading shifts rightward"],
            ["Turn-right silenced", "Turn shifts leftward", "Heading shifts leftward"],
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

    pdf.bold_start_para("Results (n=3 seeds):",
        "DM1 (attractive): turn contrast = \u22120.023 (toward odor source). "
        "DM5 (aversive): turn contrast = +0.044 (away from odor source). "
        "Valence contrast (DM5 \u2212 DM1) = +0.066. All 6 validation tests passed. "
        "The shuffled connectome eliminated all valence signals (contrast = \u22120.0005), "
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
        "The three behaviors above recruit different subsets of descending neurons. We asked "
        "whether these subsets overlap or are segregated \u2014 that is, whether the connectome "
        "implements shared or modality-specific motor output channels. We traced all direct "
        "(1-hop) connections from each sensory modality to the 350-neuron descending pool, then "
        "repeated the analysis at 2-hops (sensory \u2192 interneuron \u2192 DN).", indent=False)

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

    pdf.bold_start_para("The segregation principle.",
        "The most unexpected finding is the near-complete segregation of modality-specific "
        "descending channels at the direct sensory-to-motor interface (Jaccard 0.005\u20130.060), "
        "with rapid convergence one synapse deeper. This two-layer architecture has not been "
        "previously described.")

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
        "No previous study has identified this segregation because it requires both (a) a "
        "complete connectome to trace all paths and (b) a behavioral readout to identify which "
        "descending neurons are functionally relevant. Anatomical tracing alone cannot "
        "distinguish the 186 DNs reached by sensory populations from the 133 that receive no "
        "direct sensory input.")

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

    pdf.bold_start_para("Future directions.",
        "The segregation analysis can be extended to additional modalities (auditory, "
        "thermosensory) and deeper hop counts. Replacing the preprogrammed CPG with a VNC "
        "connectome model would close the final loop in the sensorimotor arc. The dose-response "
        "relationship between population size and behavioral effect suggests that the system can "
        "predict the behavioral consequences of genetic manipulations that silence specific "
        "neuron types.")

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

    pdf.subsection_heading("4.6", "Segregation analysis")
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
