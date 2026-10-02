# Model

## Public goods interactions and asynchronous learning

We consider a connected simple random hypergraph with $N$ nodes and $L$ distinct hyperedges of size $g$. Hyperedges are sampled uniformly without repeated nodes or duplicate groups; an entire graph is rejected if its projected graph is disconnected. Agent $i$ has public strategy $\sigma_i\in\{C,D,P\}$. Cooperators and punishers contribute $c=1$ to every group they enter. A punisher pays $\alpha$ for each defector in its group, and each defector pays a fine $\beta$ for each punisher. Let $n_C,n_D,n_P$ include the focal agent's newly selected action. With normalized multiplication factor $r=R/g$, group payoffs are

\begin{align}
\pi_C &= rc(n_C+n_P)-c,\\
\pi_D &= rc(n_C+n_P)-\beta n_P,\\
\pi_P &= rc(n_C+n_P)-c-\alpha n_D.
\label{eq:payoff}
\end{align}

The focal reward $u_i$ is the mean payoff across its incident groups, following @zou2026. This normalization is held fixed across experiments. Each agent stores a $3\times3$ table $Q_i(s,a)$; the state is its own strategy before the update and the next state is its chosen action. At each asynchronous event, one node is sampled uniformly with replacement. It explores uniformly among the three actions with probability $\epsilon$ and otherwise samples uniformly from the exact maximizers of its current Q-row. After computing the reward using the selected action, the update is

\begin{equation}
Q_i(s,a)\leftarrow Q_i(s,a)+\eta\left[u_i+\gamma\max_b Q_i(a,b)-Q_i(s,a)\right].
\label{eq:qlearning}
\end{equation}

The target uses the Q-table before this update, including when $s=a$. Public strategy and group counts are then updated. One Monte Carlo step (MCS) consists of $N$ such events. Training begins with zero Q-values and independently randomized strategies. We use $\eta=\gamma=0.8$ and $\epsilon=0.02$. These interacting learners generate a changing reward environment; no single-agent Q-learning convergence result is assumed to establish convergence of this population process.

The main observable is the contribution rate $q=(N_C+N_P)/N$, distinct from the cooperator fraction $f_C=N_C/N$. We also record the punisher fraction $f_P=N_P/N$ and group-membership-normalized snapshot welfare,

\begin{equation}
W=\frac{1}{gL}\sum_e\left[(gr-1)c(n_{C,e}+n_{P,e})-(\alpha+\beta)n_{P,e}n_{D,e}\right].
\label{eq:welfare}
\end{equation}

Fines are not redistributed. This quantity is an accounting function of a strategy snapshot, not the time-integrated rewards of asynchronously selected agents. Moreover, these linear group payoffs admit an exact weighted pairwise projection (\ref{sec:projection}). Hypergraph representation alone therefore does not establish an irreducible higher-order mechanism.

## Policy-preserving action-value interventions

At a learned checkpoint, define for each node and state row

\begin{equation}
M_{is}=\max\{Q_i(s,C),Q_i(s,D)\},\qquad d_{is}=M_{is}-Q_i(s,P).
\end{equation}

Rows with $d_{is}>10^{-10}$ are candidate intervention locations. In an eligible row, set

\begin{equation}
Q_i^{(\lambda)}(s,P)=M_{is}-\lambda d_{is},\qquad \lambda\in\{0.5,1,2\}.
\label{eq:gap}
\end{equation}

We call these levels near, sham, and far, respectively. They describe distance from the best non-punishment value, not observed punishment frequency. The $C$ and $D$ entries remain fixed. A single eligible-row set is used for all three levels. We exclude a row jointly if any proposed value lies outside the zero-initialization-compatible bound

\begin{equation}
\left[\min\left(0,\frac{\pi_{\min}}{1-\gamma}\right),\ \max\left(0,\frac{\pi_{\max}}{1-\gamma}\right)\right],
\end{equation}

where the extrema enumerate feasible focal group payoffs, or if floating-point representation removes strict suboptimality. No clipping is applied. The sham is an exact copy, avoiding rounding from an unnecessary arithmetic round trip.

**Initial-policy invariance.** Because $\lambda>0$, every edited $P$ entry remains below $M_{is}$ while both competing entries are unchanged. Thus the greedy set is unchanged. The exploration probability and tie rule are also fixed, so every node's policy is identical in all three states immediately after the intervention. Public strategies, graph structure, and other checkpoint variables are identical across intervention levels. This is stronger than equality of the actions currently expressed.

**Frozen-learning implication.** Disable every Q-update, keep strategy updates active, and expose the intervention branches to the same shock and event stream. At the first event they have the same public state and policy, and therefore select the same action from the same random inputs. Their next public states and rewards are equal. Repeating the argument proves equality of all later action and reward paths. It does not prove equality of the Q-tables, which remain deliberately different. Freezing only the $P$ column would not satisfy this argument, because changes to competing values could alter the greedy sets.

\input{figures/design.tex}

## Shocks, paired outcomes, and independent sampling

For each checkpoint and future replicate, a separate random stream selects $\lfloor\delta N\rfloor$ contributors, without replacement, to reset to $D$. It does not reset Q-values. The same selected nodes are used at every intervention level. Each level has an unshocked branch ($z=0$) and a shocked branch ($z=1$), giving six continuations (Fig. \ref{fig:design}). Every event consumes four shared uniform random numbers for node selection, exploration, exploratory action, and greedy tie-breaking, including slots that a branch does not use. This keeps random inputs aligned after behavioral divergence.

For $T=5000$ MCS, define the signed loss and the primary contrast as

\begin{align}
L_\lambda&=\frac{1}{T}\sum_{t=1}^{T}\left[q_{\lambda,0}(t)-q_{\lambda,1}(t)\right],\\
\theta&=L_{0.5}-L_2.
\label{eq:theta}
\end{align}

The branching-time snapshot is recorded but excluded from the sum. Negative losses are retained. Positive $\theta$ means that near has a larger incremental loss relative to its own unshocked continuation; it does not mean that far has a higher shocked contribution rate.

The main operating point is $N=1000$, $g=5$, $L=1600$, $r=0.55$, $\alpha=0.02$, $\beta=0.4$, and $\delta=0.30$. Preliminary public-state and sham-response screening fixed this point before near/far outcomes were examined. A documented protocol revision increased the main shock from $0.10$ to $0.30$ after the smaller shock showed rapid recovery. The confirmatory design then fixed 20 new networks, two independent training histories per network, and ten future replicates per history. This use of ``confirmatory'' denotes the locally locked design, not external preregistration.

Histories were checked at 100000, 200000, and 500000 MCS, selecting the first checkpoint with mean $q\geq0.80$ over the last 5000 MCS and changes of at most $0.02$ in both mean $q$ and mean $f_P$ between the last two 5000-MCS windows. Hidden diagnostics were recorded but were not checkpoint-selection criteria. The design permitted exclusion of wholly unqualified networks without replacement; all 40 histories qualified at 100000 MCS. Within each network we average futures and then histories. The 95% interval uses 10000 whole-network bootstrap resamples; histories, futures, and time points are not treated as independent observations. The sole primary contrast is Eq. \eqref{eq:theta}. The prespecified relevance threshold is $\Delta_{\min}=0.02$. All later decompositions and controls are exploratory, without multiplicity-adjusted claims.

## Mechanism controls and natural-history comparisons

We performed a full-horizon frozen-learning comparison for the first future of every main-experiment history. For local mechanism analysis, four prespecified networks (0--3), history 0, and future 0 were replayed under both shock conditions. The reference implementation logged events until the first near/far policy divergence and first action divergence were both located, with a 1000-MCS cap. Independent accelerated replays checked the full state at each prefix endpoint and the saved contribution and punishment trajectories over 1000 MCS. This samples eight onsets, not the population distribution of activation times.

Matched-action controls compare editable $P$ and $D$ entries in strata defined by current public strategy, Q-row state, and complete greedy set. Matched pairs have gaps within a 5% relative caliper; the two subsets have equal cardinality and total absolute rewrite magnitudes within 1% at the same intervention level. The response experiment uses history 0 from all 20 main networks and one fresh future per network. A corresponding $C$ control requires at least 30 matches and is not estimated when this gate fails. Matching does not hold topology, visit counts, or update ages fully fixed.

Natural-history experiments change the cost of punishment during learning. The original comparison reuses selected 100000-MCS low-cost ($\alpha=0.02$) checkpoints from the main experiment and trains high-cost ($\alpha=0.30$) histories for 100000 MCS from zero Q-values on the same ten graphs, using independent strategy initialization and training streams. Both histories originate from zero Q-values, but only the low-cost checkpoint is conditioned on passing the high-cooperation selection gate. A second comparison starts both branches from a common high-cooperation checkpoint and applies those costs for 10000 MCS. Both are followed by common-cost learning at $\alpha=0.02$ for washout times 0, 1000, and 10000 MCS. These designs must pass support checks before policy-preserving natural-gap comparisons are meaningful.

Following support failure in both long-history designs, we fixed a separate exploratory calibration procedure. Five preliminary-screening graphs were used to evaluate, in priority order, high-cost phases of ten MCS at $\alpha=0.30$, 100 MCS at $\alpha=0.10$, and 1000 MCS at $\alpha=0.04$ against a low-cost phase at $\alpha=0.02$. The gate required at least four of five graphs to have source $q\geq0.80$ at every washout, at least 20 donor rows per stratum, and at least 600 common eligible recipient rows across washouts. Calibration used no future-response outcomes. Each candidate clones a common checkpoint into low- and high-cost branches and uses shared random events for their cost phases. The first passing candidate was then evaluated on main-experiment networks 10--19, history 1, with fresh phase, washout, and response streams. The validation gate required at least eight of ten networks to pass all input criteria before any response outcome was simulated. These graphs are separate from calibration but are not new to the overall study.

For transplantation, empirical punishment gaps are pooled within the same exact strata as above. Recipient gaps in the original checkpoint are ranked within their strata; fixed empirical quantiles select actual donor gap values. These values replace only the recipient punishment gaps, preserving the recipient $C/D$ values, strategies, and policy. Rows must be jointly legal for both source histories at all three washouts. Thus the six branches comprise low-source, high-source, and sham transplants, each with and without shock. We also run shocked and unshocked continuations of the two raw source states; these do not isolate hidden values because their public states and other Q-values can differ. Here the contrast is $\theta_{\mathrm{nat}}=L_{\mathrm{low}}-L_{\mathrm{high}}$; low/high denotes history cost and does not imply near/far ordering. Ten validation networks and one future per network are analyzed with 10000 network resamples, sharing resampling indices across washouts. Naturally observed marginal gaps do not imply that the full transplanted Q-table is a naturally attained joint state.
