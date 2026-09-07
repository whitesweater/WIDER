# Paper figures

These assets reproduce existing figures from **Think Wider: Mitigating Latent
Rank Collapse in Implicit Chain-of-Thought Reasoning**, by Yuwen Hao and
Menglin Yang (2026).

The PDFs are copied without modification from the paper's LaTeX source.
The PNGs are direct, full-page rasterizations of those PDFs at 300 DPI using
Poppler. The root README displays Figure 1 at 600 pixels while retaining the
full image resolution, with a link to the original vector PDF. Figures 4 and
6 are also available in this directory. Figure content and labels are
preserved.

| Paper figure | LaTeX reference | Original PDF | PNG export |
|:--|:--|:--|:--|
| Figure 1: latent trajectories | `acl_latex.tex`, `fig:rank_collapse` | [PDF](figure1_latent_heatmaps_stacked2.pdf) | [PNG](figure1_latent_heatmaps_stacked2.png) |
| Figure 4: latent-step NMI | `sections/sec6_discussion.tex`, `fig:asdiv_nmi_heatmap` | [PDF](asdiv_aug_simcot_latent_step_nmi_heatmap.pdf) | [PNG](asdiv_aug_simcot_latent_step_nmi_heatmap.png) |
| Figure 6: prefix effective rank | `sections/sec6_discussion.tex`, `fig:prefix_rank` | [PDF](asdiv_aug_decoder_prefix_effective_rank_line.pdf) | [PNG](asdiv_aug_decoder_prefix_effective_rank_line.png) |

All source PDFs retain their filenames from the paper's `figures/` directory.

To refresh a PNG after replacing its corresponding source PDF:

```bash
pdftoppm -png -singlefile -r 300 \
  figure1_latent_heatmaps_stacked2.pdf \
  figure1_latent_heatmaps_stacked2
```

Use the same command with the other PDF basenames to export their PNGs.
