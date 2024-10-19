# Sailing League Problems

The provided programs are created with the intention to help on solving ***sailing league problems (SLPs)*** for various 
parameters.  

The observed sailing leagues consists of $n$ teams competing at $r$ flights. A flight is then split into races with 
$k$ participants (we demand $k$ to be a divisor of $n$). 

## Mathematical Model
Given a $SLP(n,k,r)$ with $n=tk$. A race is a $k$-subset of the set of teams $N=\{0,1,2,3,n-1\}$, a flight is a 
$t$-Partition of $N$ and a Schedule $S$ is a set of $tr$ races forming $r$ flights.  

We define the amount of times team $i$ competed against team $j$ by  
$$\lambda_{ij}(S) := \#\{R\in S : \{i,j\}\subset R\}, \quad i,j\in N$$
and the minimally and maximally amount of matches between two teams are given by
$$\lambda^+(S) := \max_{0\leq i < j < n} \lambda_{ij}(S), \qquad \lambda^-(S) := \min_{0\leq i < j < n}$$

To determine how fair a schedule is we give each a score $\Delta(S):=\lambda^+(S)-\lambda^-(S)$. A schedule is 
called a perfect schedule if $\Delta(S) = 0$. 

The goal of this work is to find optimal values for $\Delta(S)$ given a $SLP$. We therefore define a function
$$f(n,k,r) := \min\{\Delta(S) : S \text{schedule for } SLP(n,k,r)\}$$

We start by setting $t=2$ (each flight consists of two races) for further simplifications of our model.