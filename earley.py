from queue import PriorityQueue
import numpy as np

def get_RLC_matrix(grammar, NTs):
    # Compute the recursive left-corner matrix for a given grammar and set of NTs
    n = len(NTs)
    P_L = np.array([[0.0 for _ in range(n)] for _ in range(n)])
    
    nt2idx = dict(zip(NTs, range(n)))
    for (Y, *args), choices in grammar.items():
        for (p, X) in choices:
            if Y in NTs:
                P_L[nt2idx[X]][nt2idx[Y]] += p
            
    R_L = np.linalg.inv(np.eye(n) - P_L)
    
    return R_L, P_L, nt2idx
    
def print_chart(chart):
    # print a list of the columns of the chart in a way that looks ok in jupyter notebook
    for i, column in enumerate(chart):
        print(i)
        for entry in column:
            start, NT, rhs, dot, y, a = entry
            dot_str = " ".join(rhs[:dot] + ("|",) + rhs[dot:])
            print("({:2},{:3}-> {:9},{:.2},{:.2})".format(start, NT, dot_str, np.exp2(y), np.exp2(a)))
        print()

def fill_earley(get_complete_entry, get_scan_entry):
    def earley(tokens, grammar, NTs, debug = False):
        """ An Earley 'parser' that generates the prefix probabilitities under grammar for the
        sequence of tokens in tokens, implemented following Hale (2001) and Stolke (1995). This
        implementation has the has the correction for left-recursive rules, but DOES NOT have the
        fixes for empty categories/gaps/epsilon rules or for unit-production loops for the purposes
        of not complicating the code further. (That is, this is the minimal implementation that 
        works for this assignment. Read Stolke 1995 if you'd like to read about the missing 
        details) """
        
        # Recursive/Transitive Left Corner Probabilities - get around
        # an implementation problem with left-recursion in probabilistic
        # Early parsing. See Stolke (1995) section 4.5 for more info.
        R_L, _, nt2idx = get_RLC_matrix(grammar, NTs)
        
        # indices are for word boundaries rather than words
        n = len(tokens) + 1
        
        chart = [[] for _ in range(n)] 
        
        # Store indices of (start, lhs, rhs, dot_pos) so we don't have 
        # duplicate entries. We sum probabilities for duplicate entries
        # instead
        duplicate_check = [{} for _ in range(n)]
        
        # entries are (start_pos, NT, rhs, dot_pos, inside, forward) in column end_pos
        # which corresponds to a NT from start_pos to end_pos, of which
        # we've seent rhs[:dot_pos] and are predicting rhs[dot_pos:] in 
        # the future. the inside/gamma probability and forward/alpha
        # probabilities are as described in Hale/Stolke.
        
        # Initial entry - we are looking for an S
        chart[0].append((0, "", ("S",), 0, 0.0, 0.0))
        
        # Store all of the indexes of things that might need to be completed
        # in this priority queue (products of scans from the last column
        # and completes in the current column). We do this in a priority queue
        # so we complete the more recent start pos first to do all of the summing for
        # an entry before completing it (See Stolke (1995) appendix on complete)
        complete_queue = PriorityQueue() 
        
        prefixes = [0.0]
        
        # Grammar uses raw probabilities (for computation of the RLC matrix),
        # but we want to store our (small) probabilities as log-probs
        
        for end, column in enumerate(chart):
            # For each column in linear order
            
            while not complete_queue.empty():
                _, i  = complete_queue.get()
                start, NT, rhs, dot, y, a = completed = chart[end][i]
                
                if dot == len(rhs): # If the dot is at the end (e.g. complete-able)
                    # Loop through column for customers
                    for customer in chart[start]: 
                    
                        start_, NT_, rhs_, dot_, y_, a_ = customer
                        
                        # check if it's a customer (it wants what we've produced.)
                        if dot_ < len(rhs_) and rhs_[dot_] == NT:
                            idx = duplicate_check[end].get((start_, NT_, rhs_, dot_ + 1), None)
                            
                            # Do a complete - y += y' * y, a += a' * y
                            if idx: 
                                y__, a__ = chart[end][idx][-2:]
                                chart[end][idx] = get_complete_entry(y__, a__, completed, customer, end)
                            else:
                                chart[end].append(get_complete_entry(-np.inf, -np.inf, completed, customer, end))
                                duplicate_check[end][(start_, NT_, rhs_, dot_ + 1)] = len(chart[end]) - 1
                                complete_queue.put((-start_, len(chart[end]) - 1))

            # Do a batch-predict (to avoid recursive loops)
            n = len(column) # only iterate through what was there from the start
            for i in range(n):
                start, NT, rhs, dot, y, a = column[i]
                if dot >= len(rhs): continue
                next_const = rhs[dot]
                if next_const in NTs: # If theres a NT after the dot, predict
                    # for each rule, see if there's probability that this can be generated from the NT
                    for rhs_, nts in grammar.items():
                        for p, NT_ in nts:
                            r = R_L[nt2idx[next_const]][nt2idx[NT_]]
                            if r > 0:
                                # if so, predict it!
                                idx = duplicate_check[end].get((end, NT_, rhs_, 0), None)
                                if idx:
                                    y_, a_ = chart[end][idx][-2:]
                                    chart[end][idx] = chart[end][idx][:-2] + (np.log2(p), 
                                                          np.logaddexp2(a + np.log2(r) + np.log2(p), a_))
                                else:
                                    chart[end].append((end, NT_, rhs_, 0, np.log2(p), a + np.log2(r) + np.log2(p)))
                                    duplicate_check[end][(end, NT_, rhs_, 0)] = len(chart[end]) - 1

            prefix = -np.inf                        
            # See what things that we can scan
            for scanned_entry in column:
                start, NT, rhs, dot, y, a = scanned_entry
                
                # don't scan if the entry is complete or we have no more words in the input
                if dot >= len(rhs) or end >= len(tokens): continue
                next_const = rhs[dot]
                if next_const not in NTs: # if there's a terminal after the dot, scan!
                    if tokens[end] == next_const: # See if it matches the real next token
                        idx = duplicate_check[end + 1].get((start, NT, rhs, dot + 1), None)
                        # update the entry if it's already in the table
                        if idx:
                            
                            # the gamma and alpha probabilities in the existing entry
                            y_, a_ = chart[end + 1][idx][-2:] 

                            chart[end + 1][idx] = get_scan_entry(y_, a_, scanned_entry, end)
                    
                        # add a brand new entry otherwise
                        else:
                            chart[end + 1].append(get_scan_entry(-np.inf, -np.inf, scanned_entry, end)) 
                    
                            duplicate_check[end + 1][(start, NT, rhs, dot + 1)] = len(chart[end + 1]) - 1
                            complete_queue.put((-start, len(chart[end + 1]) - 1))
        
                        # Compute prefix probability:
                        # As in Stolke (1995), lemma 3c, the prefix probability is the sum over all of the
                        # scanned states 
                        prefix = np.logaddexp2(a, prefix)
            prefixes.append(prefix)
        
        if debug:    
            print_chart(chart)
        return prefixes[:-1]
    return earley
