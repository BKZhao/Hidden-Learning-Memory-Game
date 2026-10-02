# Simulation results and analysis

## Reference-model reproduction and operating-point selection

We first compared the implementation with the three time-series conditions in Fig. 3(a--c) of @zou2026. With $N=1000$, $g=5$, $L=1382$, $\alpha=0.8$, and $\beta=0.6$, each condition used 20 independent 500000-MCS runs. Figure \ref{fig:reproduction} shows the full three-strategy evolution. The final 5000-MCS means of $(f_C,f_D,f_P)$ are $(0.5489,0.4395,0.0116)$ at $r=1.0$, $(0.8411,0.0704,0.0885)$ at $r=1.1$, and $(0.7908,0.0155,0.1937)$ at $r=1.2$. The decrease in defection and greater persistence of punishment across these conditions agree qualitatively with the published curves. This is a trend and terminal-scale reproduction; the original numerical trajectories are unavailable for pointwise validation. We follow the figure's $r=1.2$ label, while recording that the accompanying source text also mentions $r=1.5$ for the third case.

```{=latex}
\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{fig02-reproduction.pdf}
\caption{Reference-model reproduction at $r=1.0$, $1.1$, and $1.2$ in panels (a)--(c). Curves show cooperator (C), defector (D), and punisher (P) fractions averaged over 20 independent runs per condition; shading is between-run standard deviation, not a confidence interval. Time is logarithmic, with sampled points taken directly from the trajectories and no temporal smoothing. All conditions use $N=1000$, $g=5$, $L=1382$, $\alpha=0.8$, $\beta=0.6$, and 500000 MCS. The scope is the three specified conditions in Zou and Huang (2026), not their full parameter study.}
\label{fig:reproduction}
\end{figure*}
```

At the target scale, a dense contextual grid covered nine values of $r$ from 0.55 to 0.95, seven values of $\alpha$ from 0.02 to 0.30, and $\beta\in\{0.4,0.8,1.2\}$. Two independent 100000-MCS training runs were completed at each of the 189 points. This scan was run after the operating point and primary experiment had been fixed; it resolves the surrounding phase structure rather than contributing to selection. The maps in Fig. \ref{fig:phase-map} show a sharp staircase boundary at $\beta=0.4$: 32 cells are in the high-contribution phase, 26 are in the low-contribution phase, and five have mean $q$ between 0.2 and 0.8. Three of these five means combine one high and one low replicate, marked by crosses, while both replicates are intermediate in the other two cells. The split outcomes are consistent with stochastic basin selection near the boundary but do not establish bistability from two replicates. At $\beta=0.8$, 62 of 63 cells have $q>0.8$, while every cell at $\beta=1.2$ does.

High terminal contribution does not everywhere imply local response stability. At $(r,\alpha,\beta)=(0.60,0.30,0.8)$, both training replicates have $q\simeq0.990$, but the paired $\delta=0.30$ contributor reset produces mean signed loss 0.902813; the two replicate losses are 0.907285 and 0.898342. Their unshocked response means remain 0.990543 and 0.990583, with baseline $q$ drift below $1.23\times10^{-4}$, while the shocked means fall to 0.083258 and 0.092241. The adjacent $r=0.55$ cell is already in the low-contribution phase, whereas the cells with $r\geq0.65$ remain high-contribution with losses below 0.001. This isolated replicated collapse identifies a narrow fragile edge of the cooperative phase. It is a descriptive phase-map result based on two networks, not an estimate of collapse probability. At $\beta=1.2$, all 63 cells remain high-contribution and their mean losses are below $5\times10^{-5}$. The three partially supported boundary cells at $\beta=0.4$ have only one eligible replicate and are marked rather than treated as comparable phase estimates.

```{=latex}
\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{fig03-phase-map.pdf}
\caption{Dense parameter-space maps at the target scale. Columns fix $\beta=0.4$, $0.8$, and $1.2$. Rows show (a--c) final-window contribution rate $q$, (d--f) final-window punisher fraction $f_P$, and (g--i) signed sham loss $L_1$ under a $\delta=0.30$ contributor reset. Each cell in the first two rows averages the final 5000 MCS of two independent 100000-MCS training runs with $N=1000$, $g=5$, and $L=1600$; crosses mark cells combining one high- and one low-contribution replicate. Response simulation was run only for replicates that passed the public stability gate. Hatching marks cells with no eligible response; stippling marks $n=1$, while unmarked response cells have $n=2$. The loss color scale is nonlinear to retain variation below the replicated 0.903 collapse at $(0.60,0.30,0.8)$; the sole negative mean, $-1.1\times10^{-6}$, is displayed at the zero floor. The star marks the operating point, which was locked before this dense scan. Values are descriptive means, without inferential intervals or interpolation between sampled parameters.}
\label{fig:phase-map}
\end{figure*}
```

```{=latex}
\FloatBarrier
```

Separate screening sought a stable, high-contribution checkpoint with a measurable sham shock response at the target population size. The selected point, with $r=0.55$, $\alpha=0.02$, and $\beta=0.4$, passed the fixed public-state and moderate-response criteria on all ten validation graphs for $\delta=0.30$. A variance pilot used new future streams on these ten screening checkpoints to inform sample planning; it did not provide an additional independent set of training networks. The protocol minimum of 20 networks determined the final sample size. The main experiment used new graphs and did not replace a network or history based on its near/far effect. Operating-point selection establishes a usable experimental condition; it is not evidence that the controlled effect generalizes across parameters.

## Controlled gap interventions alter paired losses

All 20 networks, 40 histories, and 400 paired future groups completed the six-branch response experiment. The primary estimate is $\widehat\theta=0.010679797$, with a 95% network-bootstrap interval $[0.01017784,0.01117922]$. In percentage points (pp), the estimate is 1.068 with interval [1.018, 1.118]. The network-level standard deviation is 0.001173. The contrast is distinguishable from zero under the specified design, while the entire interval lies below the prespecified two-percentage-point relevance threshold. No formal equivalence claim follows from that comparison.

Figure \ref{fig:controlled} reports the absolute paths alongside the paired contrasts. Table \ref{tab:means} gives the fixed-window means. Near has an unshocked mean contribution rate of 0.991739 and a shocked mean of 0.980887, whereas far has corresponding means 0.979685 and 0.979513. Consequently, far's incremental loss is nearly zero even though its shocked mean is slightly lower than near's. The sham loss is 0.011442. These observations rule out interpreting the primary contrast as evidence that far improves absolute post-shock cooperation.

```{=latex}
\begin{table}[t]
\centering
\caption{Main-experiment contribution means over MCS 1--5000. Networks have equal weight after future and history averaging. Losses remain signed; values are proportions.}
\label{tab:means}
\begin{tabular}{lrrr}
\toprule
Gap level & Unshocked & Shocked & $L_\lambda$\\
\midrule
Near & 0.991739 & 0.980887 & 0.010852\\
Sham & 0.991557 & 0.980115 & 0.011442\\
Far & 0.979685 & 0.979513 & 0.000172\\
\bottomrule
\end{tabular}
\end{table}
```

```{=latex}
\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{fig03-controlled.pdf}
\caption{Controlled gap responses at the locked main operating point. (a,b) Mean contribution paths without and with the $\delta=0.30$ shock, respectively, using the same vertical scale. (c) Near-minus-far contribution differences in both conditions, in percentage points (pp). Curves average ten futures within each of two histories and then average 20 networks; no temporal smoothing or inferential bands are applied. Logarithmic time starts at the first post-branch MCS; the $t=0$ snapshot is excluded from the endpoint. (d) Twenty network estimates of $\theta=L_{0.5}-L_2$ and the pooled mean with its 95\% whole-network percentile-bootstrap interval (10000 resamples). The dashed vertical line is the prespecified relevance threshold of 2 pp. A positive $\theta$ is a larger near incremental loss, not a higher far shocked contribution rate.}
\label{fig:controlled}
\end{figure*}
```

## Baseline transients and the role of continued learning

The primary endpoint decomposes exactly into the mean near-minus-far difference without shock minus the corresponding difference with shock. These components are 1.205368 pp and 0.137388 pp, respectively. The far unshocked mean curve reaches a minimum of 0.756808 at 101 MCS, while the near unshocked curve stays above 0.990992. Thus the paired loss comparison depends substantially on a baseline transient induced by the gap intervention itself.

A descriptive partition of the fixed 5000-MCS horizon attributes 0.630021 pp, or approximately 59.0% of the total contrast, to MCS 101--500. The first 50 MCS contribute negatively; Appendix Table \ref{tab:windows} retains all six windows so that this decomposition is not mistaken for selection of a favorable endpoint. During MCS 1001--5000, the unshocked and shocked means are 0.991641 and 0.989523 for near, and 0.989267 and 0.989275 for far. These later means describe the transient's attenuation, without establishing long-run equality or stationarity.

All 40 full-horizon frozen-learning comparisons pass the predicted pathwise null: for each fixed shock condition, every intervention level has exactly the same saved behavioral and payoff trajectories as its sham. The shocked and unshocked paths need not coincide with each other. This control verifies that initial policy preservation survives changes in the state rows visited after a shock and that continued learning is necessary for intervention-dependent behavioral divergence in the paired construction. It does not isolate the punishment column as a unique mechanism. Punisher and welfare trajectories are reported in Fig. \ref{fig:secondary}; their changes are descriptive secondary outcomes.

## Local divergence and matched-action controls

All eight prespecified replay prefixes exhibit the same type of first policy separation: a $C$-value update decreases a competing value enough that the unchanged near punishment value becomes greedy. Figure \ref{fig:mechanism}(a) gives the first prespecified unshocked case, network 0. At event index 42, node 301 in its $P$ state updates $Q_C$ from 8.603078 to 8.416452 in both arms. The near $Q_P$ remains 8.457229 and becomes preferred. The far $Q_P$ remains 8.019681 and does not. Neither punishment value is updated at this event. This provides a direct local example of previously suboptimal values affecting later policy through a competing-value decrease.

Across the eight prefixes, first policy separation occurs at event indices 0--74 and first action separation at 101--609, all within the first MCS (Fig. \ref{fig:mechanism}b). Event index zero means the first update after branching, not a policy difference at the intervention itself. The first differing action need not occur at the same node as the first differing policy. These replays identify onset events in a fixed small sample; they neither estimate an onset distribution over the full experiment nor quantify how much of the macroscopic contrast each event mediates.

The matched controls further restrict interpretation (Fig. \ref{fig:mechanism}c). Across the 40 main checkpoints, only 0--5 eligible $C$ matches were available per checkpoint, below the 30-match gate; no $C$ response effect was estimated. The $P/D$ comparison on 20 networks used 742--840 matched entries per side. The matched $P$ contrast is 0.071206 pp (95% interval $[-0.176836,0.348598]$), the matched $D$ contrast is $-0.269290$ pp ($[-0.598235,0.057602]$), and their direct difference is 0.340496 pp ($[-0.084972,0.752603]$). All intervals include zero. The direct comparison does not establish punishment specificity, and the smaller $P$ subset estimate cannot be directly compared with the full intervention as if action identity were the only difference.

```{=latex}
\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{fig04-mechanism.pdf}
\caption{Exploratory mechanism checks. (a) Actual Q-values before and after the first policy-divergence event in network 0, history 0, future 0, without shock (node 301, state $P$, event 42). Both $Q_C$ trajectories coincide; unchanged near $Q_P$ becomes maximal after $Q_C$ falls, whereas far $Q_P$ remains suboptimal. (b) First policy and first actual-action divergence for all eight prespecified replay prefixes, indexed by network and shock indicator. Indices are zero based and each MCS contains 1000 events. (c) Matched $P$ and $D$ subset contrasts and their direct difference, with 95\% network-bootstrap intervals across 20 networks, one history and one fresh future per network. Units are percentage points. The $C$ comparison is not estimated because it lacks matching support. Panels (a,b) are local observations, not a full-sample mediation analysis.}
\label{fig:mechanism}
\end{figure*}
```

## Natural-history support and transplantation

The long-history comparisons do not supply interchangeable high- and low-cost gap donors. In the original zero-initialized high-cost training, mean contribution rates remain near 0.013--0.015 at the assessed washouts, whereas low-cost sources remain near 0.99. No common transplant rows pass the fixed support rules, and the high-cost sources contain too few contributors for a 300-node contributor shock. These are undefined comparisons, not zero effects.

Starting the two cost histories from the same high-cooperation checkpoint does not solve this problem when the cost phase lasts 10000 MCS. Across ten networks, the high-cost mean curve falls below $q=0.5$ at 107 MCS and below $q=0.1$ at 330 MCS (Fig. \ref{fig:natural}a). These are instantaneous descriptive crossings, not the protocol's sustained-collapse endpoint. The final 1000-MCS mean is 0.013411, compared with 0.991513 in the low-cost branch. Common matching support again fails. This contrast demonstrates a substantial public-state response to prolonged cost changes, preventing the intended hidden-value comparison under the fixed matching and shock rules.

The revised short-phase calibration separates the support problem from response estimation. All three prespecified candidate histories pass the input gate on all five calibration graphs. The fixed priority therefore selects the first candidate, a ten-MCS phase at $\alpha=0.30$, without inspecting future shock effects. On the ten validation graphs, all pass before response simulation. The common eligible set contains 1459--1514 of the 3000 Q-rows per network across all washouts (Fig. \ref{fig:natural}b); the minimum source contribution rate is 0.982. The short-phase experiment completes 180 transplant branches and 120 raw-history branches, with no excluded network. A full six-branch frozen-learning check on the first validation network at zero washout also passes.

The natural-gap contrast estimates at washouts 0, 1000, and 10000 MCS are 0.041304, 0.017650, and 0.055418 pp, respectively. Their 95% intervals are $[-0.177719,0.309974]$, $[-0.086734,0.116583]$, and $[-0.053859,0.157760]$ pp (Fig. \ref{fig:natural}c). All include zero. The corresponding raw-history estimates are $-0.006974$, 0.026802, and 0.058352 pp, also with intervals spanning zero (Fig. \ref{fig:natural}d). The paired changes in transplant contrast relative to zero washout likewise have intervals spanning zero, so a washout-decay claim is unsupported.

The intervention was not a numerical sham: the mean absolute difference between low- and high-source transplanted gaps is 0.013140, 0.016311, and 0.016664 per common row at the three washouts. However, the signed mean gap difference changes sign across washouts, emphasizing that low/high source labels do not encode a uniform near/far treatment. This experiment establishes feasible transplantation of observed gap values under the revised short-history design. It neither confirms a stable natural-history effect nor establishes its absence or equivalence to zero. It also does not complete the original zero-initialized natural-history comparison, whose support failure remains part of the evidence.

```{=latex}
\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{fig05-natural-history.pdf}
\caption{Natural-history support and response are separate questions. (a) Mean contribution during the 10000-MCS common-start cost phase on networks 0--9: sustained high cost drives public-state separation and leaves no legal common transplant support at the assessed washouts. (b--d) A different, explicitly revised design uses a ten-MCS cost phase selected by input-only calibration on five other graphs, followed by validation on networks 10--19. (b) Common recipient rows across all three washouts, with the 600-row input threshold. (c) Natural-gap transplant contrasts and (d) raw-source-history contrasts at washouts 0, 1000, and 10000 MCS. Each contrast is $L_{\mathrm{low}}-L_{\mathrm{high}}$; low/high denotes source cost, not near/far gap ordering. Points and 95\% whole-network bootstrap intervals use ten networks and one future per network. All intervals are exploratory and unadjusted for multiplicity. Panel (a) is not the exposure that produced panels (b--d).}
\label{fig:natural}
\end{figure*}
```

## Interpretation and limits

The controlled intervention identifies a model-specific dependence on hidden action values despite equality of the full initial policy. Three qualifications are central. First, incremental loss is relative to an intervention-specific baseline, and a low loss can coexist with a worse unshocked trajectory. Second, frozen-learning equality and a local threshold crossing do not establish punishment-specific causation of the population response. Third, natural support is an empirical requirement that can fail; obtaining legal support under a revised history does not itself guarantee a nonzero effect.

These experiments remain conditional on one selected operating point, a fixed network size and group size, and a contributor-reset shock. No second-point six-branch experiment, empirical-network test, independent prediction task, or full-sample mechanism distribution is reported here. Later mechanism and natural-history studies reuse main-experiment networks and are explicitly exploratory. The natural response uses only ten networks and one future each, which limits precision. Formal equivalence, long-run stability, and general improvements in resilience require different evidence from the finite-window comparisons presented here.
