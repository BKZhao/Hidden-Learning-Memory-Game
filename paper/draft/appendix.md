# Weighted projection and supplementary outcomes {#sec:projection}

Let $k_i$ be node $i$'s hyperdegree, $m_{ij}$ the number of shared groups, and $w_{ij}=m_{ij}/k_i$ for $i\ne j$. Write $x_i=1$ for a contributor, $p_i=1$ for a punisher, and $d_i=1$ for a defector, using the focal newly chosen action. Separating the focal term from each group's contribution sum and collecting neighbors gives

\begin{equation}
u_i=(r-1)cx_i+\sum_{j\ne i}w_{ij}\left(rcx_j-\alpha p_i d_j-\beta d_i p_j\right).
\end{equation}

This equality follows because each neighbor term occurs exactly $m_{ij}$ times and $\sum_{j\ne i}w_{ij}=g-1$. The self-penalty products vanish because an agent cannot be both $P$ and $D$. Thus the payoff has a weighted pairwise representation despite the group-based construction. It does not follow that an arbitrary unweighted projected graph would be equivalent.

```{=latex}
\begin{table}[t]
\centering
\caption{Exploratory temporal decomposition of the fixed primary endpoint. Each contribution is the mean within-window paired contrast multiplied by window length divided by 5000. All windows sum to 1.0679797 pp; none replaces the primary endpoint.}
\label{tab:windows}
\begin{tabular}{rr}
\toprule
MCS window & Contribution (pp)\\
\midrule
1--10 & $-0.02099705$\\
11--50 & $-0.03000745$\\
51--100 & $0.17086720$\\
101--500 & $0.63002130$\\
501--1000 & $0.14806595$\\
1001--5000 & $0.17002975$\\
\bottomrule
\end{tabular}
\end{table}
```

```{=latex}
\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{figA1-secondary.pdf}
\caption{Descriptive secondary paths in the main six-branch experiment. (a,b) Punisher fraction and (c,d) snapshot welfare without and with shock, respectively. All curves use the same averaging hierarchy as Fig. \ref{fig:controlled}; no temporal smoothing is applied. Welfare follows Eq. \eqref{eq:welfare} and is normalized by group memberships. These outcomes were not additional primary tests.}
\label{fig:secondary}
\end{figure*}
```

# Run hierarchy and provenance

The main study used root seed 2026100106 and bootstrap seed 2026100107. Its 20 network means each average two histories and ten futures per history. The matched-action comparison used new response seed 2026100110 and bootstrap seed 2026100111 on history 0 of those same networks. The replay sample used archived main-study event streams, not newly drawn trajectories. Natural-support calibration used root seed 2026100140 on five preliminary-screening graphs; validation and response used seed 2026100141 on main networks 10--19, history 1, with bootstrap seed 2026100142. Shared indices across washouts preserve the pairing when estimating changes from zero washout.

All main and natural response endpoints use MCS 1--5000 and $\delta=0.30$. The main experiment comprises 2400 adaptive response branches and 240 full-horizon frozen branches; the matched-action response comprises 200 adaptive branches and ten frozen branches on its first network. The revised natural validation comprises 300 adaptive branches, of which 180 are transplants and 120 raw-source continuations, plus six frozen branches. The branch counts are not the independent sample sizes.

The implementation stores float64 Q-tables, exact greedy tie sets, separate shock and event streams, checkpoint hashes, selected shock nodes, and archived execution configurations. A readable event-level implementation and an accelerated implementation were checked against shared inputs. Figure scripts read archived numeric outputs; their manifests record input and plotting-script hashes. Main-experiment source archives are retained separately from modules added for later exploratory analyses. Repository and artifact locations are listed in the accompanying manuscript README; public archival deposition and author declarations remain submission-preparation tasks rather than completed study outcomes.
