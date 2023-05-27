from treetime import TreeTime
from treetime.utils import parse_dates
import argparse
import numpy as np

def calc_node_timings(T, sigma_sq, mu, eps=0.2):
    for n in T.find_clades(order='postorder'):
        if n.is_terminal():
            prefactor = (n.observations/sigma_sq + mu**2/(n.nmuts+eps))
            n.a = (n.avg_date/sigma_sq + mu*n.nmuts/(n.nmuts+eps))/prefactor
        else:
            if n==T.root:
                tmp_children_1 = mu*np.sum([(mu*c.a-c.nmuts)/(eps+c.nmuts) for c in n])
                tmp_children_2 = mu**2*np.sum([(1-c.b)/(eps+c.nmuts) for c in n])
                prefactor = (n.observations/sigma_sq + tmp_children_2)
                n.a = (n.avg_date/sigma_sq + tmp_children_1)/prefactor
            else:
                tmp_children_1 = mu*np.sum([(mu*c.a-c.nmuts)/(eps+c.nmuts) for c in n])
                tmp_children_2 = mu**2*np.sum([(1-c.b)/(eps+c.nmuts) for c in n])
                prefactor = (n.observations/sigma_sq + mu**2/(n.nmuts+eps) + tmp_children_2)
                n.a = (n.avg_date/sigma_sq + mu*n.nmuts/(n.nmuts+eps)+tmp_children_1)/prefactor
        n.b = mu**2/(n.nmuts+eps)/prefactor
    T.root.tau = T.root.a

    for n in T.get_nonterminals(order='preorder'):
        for c in n:
            c.tau = c.a + c.b*n.tau

def calc_scores_to_optimize(x, T):
    mu, sigma = x
    calc_scores(T, sigma=sigma, mu=mu)

def calc_scores(T, sigma=None, mu=None):
    sigma_sq=sigma**2
    calc_node_timings(T, sigma_sq=sigma_sq, mu=mu)
    cost = 0
    n_tips = 0
    for n in T.find_clades():
        for x in n.tips.values():
            x['z'] = (x['date']-n.tau)/sigma
            cost += x['z']**2
        for c in n:
            cost += (mu*(c.tau-n.tau) - c.nmuts)**2/(c.nmuts+1)
        n_tips += n.observations

    res = 0.5*cost + np.log(2*np.pi*(sigma_sq+0.1))*n_tips*0.5
    return res

def prepare_tree(T):
    pruned_tips = []
    for n in T.get_nonterminals(order='preorder'):
        n.dates = []
        n.tips = {}
        children_to_prune = set()
        for c in n:
            if c.is_terminal():
                if c.raw_date_constraint is None: #len(c.mutations)==0:
                    if c.raw_date_constraint is not None:
                        n.dates.append(c.raw_date_constraint)
                        n.tips[c.name]={'date':c.raw_date_constraint}
                    children_to_prune.add(c)
                else:
                    c.dates = [c.raw_date_constraint]
                    c.observations = 1
                    c.avg_date = c.raw_date_constraint
                    c.tips = {c.name:{'date':c.raw_date_constraint}}
                    c.nmuts = len([m for m in c.mutations if m[-1] in 'ACGT'])

        n.clades = [c for c in n.clades if c not in children_to_prune]
        pruned_tips.extend(children_to_prune)
        n.nmuts = len(n.mutations)
        n.observations = len(n.dates)
        n.avg_date = np.mean(n.dates) if n.observations else 0
    return pruned_tips

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Flag outliers in a tree')
    parser.add_argument('--tree', type=str, help='tree file in newick format')
    parser.add_argument('--aln', type=str, help='alignment file in fasta format')
    parser.add_argument('--cutoff', type=float, default=2.0, help="z-score used to flag outliers")
    parser.add_argument('--optimize', action="store_true", help="optimize sigma and mu")
    parser.add_argument('--dates', type=str, help='csv file with dates for each sequence')

    args = parser.parse_args()
    dates = parse_dates(args.dates)
    tt = TreeTime(gtr='JC69', tree=args.tree, aln=args.aln, verbose=4, dates=dates)
    tt.clock_filter(n_iqd=4, reroot='least-squares')
    if args.aln:
        tt.infer_ancestral_sequences(prune_short=True, marginal=True)

    print(tt.tree.count_terminals())
    pruned_tips = prepare_tree(tt.tree)
    print(pruned_tips)

    mu = tt.clock_model['slope']*tt.data.full_length
    sigma = 3/mu
    if args.optimize:
        from scipy.optimize import minimize
        x0=(mu, sigma)
        print(calc_scores(x0, tt.tree))
        sol = minimize(calc_scores_to_optimize, x0=x0, args=(tt.tree,), method='Nelder-Mead')
        score = calc_scores(sol['x'], tt.tree)
    else:
        print(f"Calculating node timings using {mu=:1.3e} and {sigma=:1.3e}")
        score = calc_scores(tt.tree, mu=mu, sigma=sigma)

    print(f"\nflagged node\tz-score\ttau\tdate")
    z_dist = []
    for n in tt.tree.find_clades():
        for tip, s in n.tips.items():
            z_dist.append(s['z'])
            if np.abs(s['z'])>args.cutoff:
                print(f"{tip}\t{s['z']:1.2f}\t{n.tau:1.2f}\t{s['date']:1.2f}\t{n.nmuts}")
